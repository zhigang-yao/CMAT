"""CMAT trajectory metrics, paired aggregation, bootstrap, and fixed motion bounds.

XYZ input units are metres. Rare labels and validity masks are shared across
methods; acceleration is measured in mm/frame^2 on the complete trajectory.
"""

from math import erfc, sqrt

import numpy as np

# Metrics
# Paper metrics on complete XYZ trajectories; input units are metres.
#
# Select rare centers only after computing temporal errors on the complete clip.
# Acceleration uses frame differences, not time derivatives, hence mm/frame^2.

METRIC_NAMES = ("mpjpe_mm", "pa_mpjpe_mm", "accel_error_mm_frame2")


def _trajectory(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 3 or value.shape[-1] != 3 or value.shape[1] < 1:
        raise ValueError(f"{name} must have shape (frames, joints, 3)")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite coordinates")
    return value


def root_center(positions, root_indices=(0,)):
    """Center independently, using joint 0 for AMASS or hips (2, 3) for EFT."""
    positions = _trajectory(positions, "positions")
    roots = np.asarray(root_indices)
    if roots.ndim != 1 or len(roots) == 0 or roots.dtype.kind not in "iu":
        raise ValueError("root_indices must be a nonempty sequence of integers")
    if len(np.unique(roots)) != len(roots):
        raise ValueError("root_indices must be distinct")
    if np.any(roots < 0) or np.any(roots >= positions.shape[1]):
        raise ValueError("root index outside the joint array")
    return positions - positions[:, roots].mean(axis=1, keepdims=True)


def pa_mpjpe(prediction, target):
    """Per-frame similarity alignment with a proper rotation and nonnegative scale."""
    prediction = _trajectory(prediction, "prediction")
    target = _trajectory(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    x = prediction - prediction.mean(axis=1, keepdims=True)
    y = target - target.mean(axis=1, keepdims=True)
    u, singular, vh = np.linalg.svd(x.transpose(0, 2, 1) @ y)
    sign = np.where(np.linalg.det(u @ vh) < 0, -1.0, 1.0)
    u[:, :, -1] *= sign[:, None]
    numerator = singular[:, :2].sum(axis=1) + sign * singular[:, -1]
    scale = np.maximum(numerator, 0) / np.maximum((x * x).sum((1, 2)), 1e-30)
    aligned = scale[:, None, None] * (x @ (u @ vh))
    return 1000 * np.linalg.norm(aligned - y, axis=2).mean(axis=1)


def acceleration_centers(length, frame_ids=None):
    """Valid centers; absent IDs mean consecutive samples on the analysis grid.

    AMASS was resampled to 30 Hz, so use analysis-grid IDs (or omit them),
    not raw mocap source IDs. EFT uses its original image-frame IDs.
    """
    mask = np.zeros(length, dtype=bool)
    if frame_ids is None:
        mask[1:-1] = True
        return mask
    ids = np.asarray(frame_ids)
    if ids.shape != (length,) or ids.dtype.kind not in "iu":
        raise ValueError("frame_ids must contain one integer per frame")
    if np.any(np.diff(ids) <= 0):
        raise ValueError("frame IDs must be strictly increasing within a track")
    mask[1:-1] = (np.diff(ids)[:-1] == 1) & (np.diff(ids)[1:] == 1)
    return mask


def frame_errors(prediction, target, root_indices=(0,), frame_ids=None):
    """Return (frames, 3): MPJPE, PA-MPJPE, and acceleration error.

    Invalid acceleration centers are NaN rather than artificial zero errors.
    No method-specific mask, endpoint padding, or fps multiplier is applied.
    """
    prediction = _trajectory(prediction, "prediction")
    target = _trajectory(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    residual = root_center(prediction, root_indices) - root_center(target, root_indices)
    mp = 1000 * np.linalg.norm(residual, axis=-1).mean(axis=-1)
    pa = pa_mpjpe(prediction, target)
    ac = np.full(len(prediction), np.nan)
    if len(prediction) >= 3:
        ac[1:-1] = 1000 * np.linalg.norm(np.diff(residual, n=2, axis=0), axis=-1).mean(-1)
    ac[~acceleration_centers(len(prediction), frame_ids)] = np.nan
    return np.column_stack((mp, pa, ac))


# Aggregation
# Pool valid frame observations, retaining subject clusters and noise repeats.


def window_totals(errors, rare):
    """Sum (clips, repeats, methods, frames, metrics) errors over rare centers.

    Rare labels refer to clean configurations and are shared by methods and
    noise repeats. The neighboring frames of a rare acceleration center do not
    themselves have to be rare. All methods must have identical valid entries.
    """
    errors = np.asarray(errors, dtype=np.float64)
    rare = np.asarray(rare)
    if errors.ndim != 5 or rare.shape != (errors.shape[0], errors.shape[3]):
        raise ValueError("expected five error axes and a (clips, frames) rare mask")
    if rare.dtype.kind != "b":
        raise ValueError("rare must be a Boolean mask")
    if np.isinf(errors).any():
        raise ValueError("errors cannot contain infinity")
    finite = np.isfinite(errors)
    if not np.array_equal(finite, np.broadcast_to(finite[:, :, :1], finite.shape)):
        raise ValueError("methods do not share the same valid frame observations")
    selected = finite & rare[:, None, None, :, None]
    sums = np.where(selected, errors, 0.0).sum(axis=(1, 3))
    counts = selected.sum(axis=(1, 3))
    return sums, counts


def subject_totals(sums, counts, subjects, keep=None):
    """Combine all clips and repeats of a subject before cluster resampling."""
    sums, counts = np.asarray(sums, float), np.asarray(counts, float)
    subjects = np.asarray(subjects)
    if sums.shape != counts.shape or sums.ndim < 2 or subjects.shape != sums.shape[:1]:
        raise ValueError("sums/counts need matching first clip axis and subject IDs")
    if not np.isfinite(sums).all() or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("finite sums and nonnegative counts are required")
    if np.any((counts == 0) & (sums != 0)):
        raise ValueError("zero-count cells must have zero sums")
    if keep is None:
        keep = np.ones(len(subjects), bool)
    keep = np.asarray(keep)
    if keep.shape != subjects.shape or keep.dtype.kind != "b" or not keep.any():
        raise ValueError("keep must be a nonempty Boolean clip mask")
    names, inverse = np.unique(subjects[keep], return_inverse=True)
    shape = (len(names),) + sums.shape[1:]
    grouped_sums, grouped_counts = np.zeros(shape), np.zeros(shape)
    np.add.at(grouped_sums, inverse, sums[keep])
    np.add.at(grouped_counts, inverse, counts[keep])
    return names, grouped_sums, grouped_counts


def pooled_mean(sums, counts):
    """Ratio of sums, not the unweighted mean of clip means."""
    sums, counts = np.asarray(sums, float), np.asarray(counts, float)
    if sums.shape != counts.shape:
        raise ValueError("sums and counts must have equal shapes")
    denominator = counts.sum(axis=0)
    return np.divide(
        sums.sum(axis=0), denominator, out=np.full_like(denominator, np.nan), where=denominator > 0
    )


# Bootstrap
# Paired subject bootstrap and the paper's component-comparison family.
#
# Intervals condition on fixed labels, clips, noise and the selected motion mask.
# They do not correct for selecting that mask after examining results.


def subject_bootstrap(sums, counts, n_resamples=10000, seed=20260926):
    """Return point means and shared bootstrap means for subject-level totals.

    Arrays have shape (subjects, methods, metrics). One multiplicity draw is
    shared by every method/metric, keeping a subject's clips and repeats intact.
    Multinomial sampling exactly follows the frozen current-paper analysis.
    """
    sums, counts = np.asarray(sums, float), np.asarray(counts, float)
    if sums.shape != counts.shape or sums.ndim != 3 or len(sums) == 0:
        raise ValueError("expected matching (subjects, methods, metrics) arrays")
    if not np.isfinite(sums).all() or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("finite totals and nonnegative counts are required")
    if np.any((counts == 0) & (sums != 0)):
        raise ValueError("zero-count cells must have zero sums")
    if not np.array_equal(counts, np.broadcast_to(counts[:, :1], counts.shape)):
        raise ValueError("method denominators differ; comparisons must be paired")
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 2:
        raise ValueError("n_resamples must be an integer of at least two")
    n_subjects = len(sums)
    draws = np.random.default_rng(seed).multinomial(
        n_subjects, np.full(n_subjects, 1 / n_subjects), n_resamples
    )
    numerator = np.einsum("bs,smk->bmk", draws, sums)
    denominator = np.einsum("bs,smk->bmk", draws, counts)
    if np.any(denominator <= 0):
        raise ValueError("a bootstrap replicate has no valid observations")
    return pooled_mean(sums, counts), numerator / denominator


def paired_comparison(point, samples, comparator, reference):
    """Comparator minus reference: positive differences favor the reference."""
    point, samples = np.asarray(point, float), np.asarray(samples, float)
    if samples.ndim != 3 or point.shape != samples.shape[1:]:
        raise ValueError("expected (methods, metrics) point and matching samples")
    difference = point[comparator] - point[reference]
    draws = samples[:, comparator] - samples[:, reference]
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    standard_error = np.std(draws, axis=0, ddof=1)
    # This floor matches the archived component analysis, including null rows.
    z = np.abs(difference) / np.maximum(standard_error, 1e-12)
    p_values = np.array([erfc(float(v) / sqrt(2)) for v in z])
    return {
        "difference": difference,
        "ci_low": low,
        "ci_high": high,
        "bootstrap_sd": standard_error,
        "p_normal_bootstrap_sd": p_values,
    }


def benjamini_hochberg(p_values):
    """Adjust the entire predeclared family (19 variants x 3 metrics = 57)."""
    p = np.asarray(p_values, float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be a finite probability vector")
    if len(p) == 0:
        return p.copy()
    order = np.argsort(p)
    adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1)
    return result


def wilson_interval(accepted, attempted, z=1.96):
    """Generation yield uncertainty; rejected/fallback attempts stay counted."""
    if attempted <= 0 or not 0 <= accepted <= attempted or z <= 0:
        raise ValueError("require 0 <= accepted <= attempted and attempted > 0")
    p = accepted / attempted
    denominator = 1 + z**2 / attempted
    center = p + z**2 / (2 * attempted)
    radius = z * sqrt(p * (1 - p) / attempted + z**2 / (4 * attempted**2))
    return (center - radius) / denominator, (center + radius) / denominator


# Motion Bounds
# Reproduce the fixed retrospective AMASS motion-amplitude restriction.
#
# These are descriptors of the noisy input, not anatomical validity tests.
# The target body scale is for experimental cohort description only; inference
# must normalize with its observed-data scale and must not read the target.


def motion_descriptors(target, observed, fps=30.0, root_indices=(0,)):
    """Return body scale plus one speed/amplitude value per noise realization.

    target: (T,J,3); observed: (R,T,J,3). Speed is in body scales/s;
    amplitude is the RMS temporal deviation in body scales.
    """
    clean = root_center(target, root_indices)
    observations = np.asarray(observed, dtype=np.float64)
    if observations.ndim != 4 or observations.shape[1:] != clean.shape:
        raise ValueError("observed must have shape (replicates, frames, joints, 3)")
    if len(clean) < 2 or len(observations) == 0 or fps <= 0:
        raise ValueError("require at least two frames, one replicate and positive fps")
    scale = np.median(np.sqrt(np.mean(np.sum(clean**2, axis=-1), axis=-1)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("the target must have a positive finite body scale")
    speed, amplitude = [], []
    for observation in observations:
        centered = root_center(observation, root_indices)
        speed.append(fps * np.linalg.norm(np.diff(centered, axis=0), axis=-1).mean() / scale)
        temporal = centered - centered.mean(axis=0, keepdims=True)
        amplitude.append(np.sqrt(np.mean(np.sum(temporal**2, axis=-1))) / scale)
    return {
        "body_scale_m": float(scale),
        "speed": np.asarray(speed),
        "amplitude": np.asarray(amplitude),
    }


def rounded_upper_quantile(values, quantile):
    """Equal-clip linear quantile, rounded upward to the 0.01 grid."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a nonempty finite vector")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must lie in [0, 1]")
    return float(np.ceil(100 * np.quantile(values, quantile, method="linear")) / 100)


def fixed_motion_mask(speed, amplitude, speed_upper=6.34, amplitude_upper=0.73):
    """Apply frozen paper bounds to replicate-averaged clip descriptors.

    Accept (clips,) averaged descriptors or (clips, replicates) arrays.
    The speed limit is the rounded sample maximum and excludes no paper clip.
    Do not re-estimate a quantile from each new test set to improve results.
    """
    speed, amplitude = np.asarray(speed, dtype=float), np.asarray(amplitude, dtype=float)
    if speed.shape != amplitude.shape or speed.ndim not in (1, 2):
        raise ValueError("speed and amplitude must have equal one- or two-dimensional shapes")
    if speed.ndim == 2:
        if speed.shape[1] == 0:
            raise ValueError("at least one replicate is required")
        speed, amplitude = speed.mean(1), amplitude.mean(1)
    if not np.isfinite(speed).all() or not np.isfinite(amplitude).all():
        raise ValueError("motion descriptors must be finite")
    if np.any(speed < 0) or np.any(amplitude < 0):
        raise ValueError("motion descriptors must be nonnegative")
    return (speed <= speed_upper) & (amplitude <= amplitude_upper)
