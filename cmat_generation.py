"""Rarity-preserving CMAT generation with shared screening and resumable blocks.

Rotation-space baselines are untrained implementations. Final geometry checks
use finite joint pairs; they do not perform mesh collision detection.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from cmat_geometry import chart, fit, project
from cmat_rotations import (
    cap_joint, canonicalize, distance, exp_map, relative_log, to_axis_angle,
)
from cmat_rarity import weighted_quantile

__all__ = [
    "METHODS",
    "PARTS",
    "LIMB_ROOTS",
    "JOINT_PAIRS",
    "generate_block",
    "prepare_anchor_pool",
    "run_blocks",
    "candidate_bank",
    "anchor_bound",
    "local_deviation",
    "refinement_trials",
    "select_candidate",
    "gaussian",
    "part_mix",
    "posetrans_3d",
    "poseaug_3d",
    "normalized_pair_distances",
    "calibrate_pair_thresholds",
    "final_screen",
    "yield_summary",
]


# Six candidates and fixed refinement grid

def local_deviation(pose, neighbors) -> float:
    residuals = relative_log(pose[None], neighbors).reshape(len(neighbors), -1)
    local = chart(residuals, 12)
    normal_mean = local.mean - project(local.mean, local.basis)
    return float(np.linalg.norm(normal_mean) / max(np.sqrt(local.scale_squared), 1e-12))


def anchor_bound(pose, anchor, neighborhood_scale):
    """Bound the actual composed displacement, then each joint's rotation."""
    displacement = relative_log(anchor, pose).reshape(-1)
    displacement *= min(1, 0.65 * neighborhood_scale
                        / max(np.linalg.norm(displacement), 1e-12))
    return exp_map(anchor, cap_joint(displacement, 12).reshape(21, 3))


def candidate_bank(anchor, partner, neighbors, seed: int) -> dict:
    """One attempt's six candidates, each followed by one CMAT update.

    Sampling order (normal then uniform per candidate) matches saved runs.
    The same anchor neighbors are used for every candidate comparison.
    """
    rng = np.random.default_rng(seed)
    residuals = relative_log(anchor[None], neighbors).reshape(len(neighbors), -1)
    local = chart(residuals, 12)
    scale = np.sqrt(local.scale_squared)
    edge = relative_log(anchor, partner).reshape(-1)
    poses, errors, diagnostics = [], [], []
    for _ in range(6):
        exploration = (rng.normal(size=12) * local.singular[:12]) @ local.basis
        increment = rng.uniform(0.25, 0.75) * project(edge, local.basis) + 0.34 * exploration
        initial = anchor_bound(exp_map(anchor, increment.reshape(21, 3)), anchor, scale)
        current = relative_log(initial[None], neighbors).reshape(len(neighbors), -1)
        fitted = fit(current, 12)
        proposal = exp_map(initial, cap_joint(fitted["step"], 12).reshape(21, 3))
        proposal = anchor_bound(proposal, anchor, scale)
        poses.append(proposal)
        errors.append(local_deviation(proposal, neighbors))
        diagnostics.append([fitted[key] for key in (
            "gate", "radius", "half_length", "support", "effective_neighbors", "enlargement",
        )])
    return dict(poses=np.asarray(poses), deviation=np.asarray(errors),
                diagnostics=np.asarray(diagnostics))


def refinement_trials(pose, neighbors) -> dict:
    residuals = relative_log(pose[None], neighbors).reshape(len(neighbors), -1)
    fitted = fit(residuals, 12)
    increment = cap_joint(fitted["step"], 12).reshape(21, 3)
    poses = np.asarray([exp_map(pose, fraction * increment)
                        for fraction in (0, 0.25, 0.5, 0.75, 1)])
    return dict(poses=poses, deviation=np.asarray([
        local_deviation(candidate, neighbors) for candidate in poses
    ]))


def select_candidate(percentiles, novel, deviations, previous_percentile=None):
    """Feasible minimum deviation, earliest tie; first candidate on failure.

    The fallback is only passed to the next stage, never automatically accepted.
    Inputs have shape [attempt, candidates].
    """
    percentiles, novel = np.asarray(percentiles), np.asarray(novel)
    deviations = np.asarray(deviations)
    feasible = (percentiles >= 0.95) & novel & (deviations <= 0.55)
    if previous_percentile is not None:
        feasible &= percentiles >= np.asarray(previous_percentile)[:, None] - 0.005
    score = np.where(feasible, deviations, np.inf)
    selected = np.argmin(score, axis=1)
    selected[~feasible.any(axis=1)] = 0
    return selected, feasible


# Rotation-space baselines

PARTS = {
    "torso": (2, 5, 8, 11, 14), "left_arm": (12, 15, 17, 19),
    "right_arm": (13, 16, 18, 20), "left_leg": (0, 3, 6, 9),
    "right_leg": (1, 4, 7, 10),
}
LIMB_ROOTS = (0, 1, 3, 4, 15, 16, 17, 18)


def gaussian(anchor, rng, degrees=8.0):
    direction = rng.normal(size=(len(anchor), 21, 3))
    direction /= np.maximum(np.linalg.norm(direction, axis=2, keepdims=True), 1e-12)
    magnitude = abs(rng.normal(scale=np.deg2rad(degrees), size=(len(anchor), 21, 1)))
    return exp_map(anchor, direction * magnitude)


def part_mix(anchor, partner, rng):
    result = np.array(anchor, copy=True)
    parts = tuple(PARTS.values())
    for row in range(len(result)):
        count = int(rng.integers(1, 3))
        selected = rng.choice(len(parts), count, replace=False)
        for part in selected:
            joints = np.asarray(parts[int(part)])
            result[row, joints] = partner[row, joints]
    return canonicalize(result)


def _random_axis_angle(rng, shape, degrees):
    direction = rng.normal(size=shape)
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
    magnitude = rng.normal(scale=np.deg2rad(degrees), size=shape[:-1] + (1,))
    return direction * magnitude


def posetrans_3d(anchor, rng, degrees=18.0):
    tangent = np.zeros((len(anchor), 21, 3))
    for joint in LIMB_ROOTS:
        active = rng.random(len(anchor)) < 0.5
        if active.any():
            tangent[active, joint] = _random_axis_angle(rng, (int(active.sum()), 3), degrees)
    return exp_map(anchor, tangent)


def poseaug_3d(anchor, rng, degrees=12.0):
    tangent = np.zeros((len(anchor), 21, 3))
    for name, joints in PARTS.items():
        active = rng.random(len(anchor)) < 0.65
        if not active.any():
            continue
        scale = 0.55 if name == "torso" else 1.0
        shared = _random_axis_angle(rng, (int(active.sum()), 3), degrees * scale)
        for joint in joints:
            local = _random_axis_angle(rng, (int(active.sum()), 3), 0.25 * degrees * scale)
            tangent[active, joint] = shared + local
    return exp_map(anchor, tangent)


# Common final screening

JOINT_PAIRS = np.asarray([
    (4, 7), (5, 8), (7, 10), (8, 11), (10, 11), (15, 16), (15, 20), (16, 19),
    (17, 18), (19, 20), (4, 15), (5, 16), (7, 17), (8, 18), (10, 19), (11, 20),
], dtype=np.int64)


def normalized_pair_distances(joints):
    joints = np.asarray(joints)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError("Expected neutral SMPL joints shaped [N,22,3]")
    scale = np.linalg.norm(joints[:, 2] - joints[:, 1], axis=1)
    scale += np.linalg.norm(joints[:, 3] - joints[:, 2], axis=1)
    first, second = joints[:, JOINT_PAIRS[:, 0]], joints[:, JOINT_PAIRS[:, 1]]
    return np.linalg.norm(first - second, axis=2) / np.maximum(scale, 1e-6)[:, None]


def calibrate_pair_thresholds(rarity_model, decoder):
    """Decoder consumes [N,21,3] axis angles and returns neutral SMPL [N,22,3]."""
    joints = decoder(to_axis_angle(rarity_model.calibration_poses))
    pairs = normalized_pair_distances(joints)
    thresholds = {}
    for source, weights in rarity_model.weights.items():
        mask = rarity_model.sources == source
        thresholds[source] = np.asarray([
            max(0, weighted_quantile(pairs[mask, column], weights, 0.002) - 0.002)
            for column in range(len(JOINT_PAIRS))
        ])
    return thresholds


def final_screen(poses, sources, subjects, rarity_model, decoder, pair_thresholds):
    """Fresh neighbors and all final checks; no isolation-percentile rejection."""
    sources = np.broadcast_to(sources, (len(poses),))
    scores = rarity_model.classify(poses, sources, subjects)
    reference = rarity_model.retriever.reference
    deviations = np.asarray([
        local_deviation(pose, reference[rows[:64]])
        for pose, rows in zip(poses, scores["neighbors"])
    ])
    axis_angle = to_axis_angle(poses)
    joints = decoder(axis_angle)
    pairs = normalized_pair_distances(joints)
    valid = np.isfinite(axis_angle).all((1, 2)) & np.isfinite(joints).all((1, 2))
    for source in np.unique(sources):
        mask = sources == source
        valid[mask] &= (pairs[mask] >= pair_thresholds[source]).all(1)
    accepted = scores["rare"] & scores["novel"] & (deviations <= 0.55) & valid
    return dict(body_pose=axis_angle, joints=joints, rho=scores["rho"],
                score=scores["score"], nearest=scores["nearest"], rare=scores["rare"],
                novel=scores["novel"], chart=deviations, geometry=valid, accepted=accepted)


def yield_summary(accepted):
    """All attempts, including infeasible fallbacks, remain in the denominator."""
    accepted = np.asarray(accepted, dtype=bool)
    if not accepted.size:
        raise ValueError("At least one attempt is required")
    return dict(attempts=int(accepted.size), accepted=int(accepted.sum()),
                yield_fraction=float(accepted.mean()))


# Deterministic generation and atomic resume

METHODS = (
    "rare_replay", "axis_gaussian", "part_mix", "posetrans_3d",
    "poseaug_3d", "rarity_preserving_cmat",
)


def prepare_anchor_pool(rarity_model, reference_sources, search_size=250000,
                        seed=20260915):
    """Screen first, then retain only subjects with a rare partner."""
    reference_sources = np.asarray(reference_sources)
    retriever = rarity_model.retriever
    eligible = np.flatnonzero(np.isin(reference_sources, list(rarity_model.weights)))
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(eligible, min(search_size, len(eligible)), replace=False))
    scores = rarity_model.classify(retriever.reference[rows], reference_sources[rows],
                                   retriever.subject_ids[rows])
    keep = scores["rare"].copy()
    subjects = retriever.subject_ids[rows]
    for subject in np.unique(subjects):
        if np.sum(keep & (subjects == subject)) < 2:
            keep[subjects == subject] = False
    return dict(pool_rows=rows, anchor_rows=rows[keep], anchor_rho=scores["rho"][keep],
                anchor_subject=subjects[keep], pool_rho=scores["rho"])


def generate_block(rarity_model, anchor_rows, reference_sources, decoder,
                   pair_thresholds, start=0, count=250):
    """Run one deterministic block with the saved experiment's RNG streams.

    Use starts 0,250,... for paper reproduction. Changing block boundaries or
    count changes later draws within a block and is a new run, not a resume.
    decoder(axis_angles) must return neutral SMPL's first 22 joints. Tests may
    supply a toy decoder, but such outputs are never paper results.
    """
    retriever = rarity_model.retriever
    reference = retriever.reference
    anchor_rows = np.asarray(anchor_rows, dtype=np.int64)
    if len(anchor_rows) < 2 or count < 1:
        raise ValueError("Need a nonempty attempt block and paired rare anchors")
    pool_subjects = retriever.subject_ids[anchor_rows]
    by_subject = {subject: np.flatnonzero(pool_subjects == subject)
                  for subject in np.unique(pool_subjects)}
    if any(len(rows) < 2 for rows in by_subject.values()):
        raise ValueError("Every retained anchor subject needs at least two anchors")
    rng = np.random.default_rng(20260915 + start)
    anchor_indices = rng.integers(len(anchor_rows), size=count)
    anchors = reference[anchor_rows[anchor_indices]]
    subjects = pool_subjects[anchor_indices]
    sources = np.asarray(reference_sources)[anchor_rows[anchor_indices]]
    partner_indices = []
    for index in anchor_indices:
        eligible = by_subject[pool_subjects[index]]
        eligible = eligible[eligible != index]
        distances = distance(reference[anchor_rows[index]], reference[anchor_rows[eligible]])
        closest = eligible[np.argsort(distances)[:12]]
        partner_indices.append(int(rng.choice(closest)))
    partners = reference[anchor_rows[np.asarray(partner_indices)]]
    geometry = retriever.batch(anchors)
    local_rows = geometry["neighbors"][:, 1:65]
    seeds = rng.integers(2**32, size=count)
    banks = [candidate_bank(a, p, reference[rows], int(seed))
             for a, p, rows, seed in zip(anchors, partners, local_rows, seeds)]
    bank_poses = np.stack([bank["poses"] for bank in banks])
    bank_error = np.stack([bank["deviation"] for bank in banks])
    bank_scores = rarity_model.classify(bank_poses.reshape(-1, 21, 4),
                                        np.repeat(sources, 6), np.repeat(subjects, 6))
    bank_rho = bank_scores["rho"].reshape(count, 6)
    selected, bank_feasible = select_candidate(
        bank_rho, bank_scores["novel"].reshape(count, 6), bank_error,
    )
    selected_poses = bank_poses[np.arange(count), selected]
    selected_rho = bank_rho[np.arange(count), selected]
    selected_neighbors = retriever.batch(selected_poses)["neighbors"][:, :64]
    lines = [refinement_trials(q, reference[rows])
             for q, rows in zip(selected_poses, selected_neighbors)]
    line_poses = np.stack([line["poses"] for line in lines])
    line_error = np.stack([line["deviation"] for line in lines])
    line_scores = rarity_model.classify(line_poses.reshape(-1, 21, 4),
                                        np.repeat(sources, 5), np.repeat(subjects, 5))
    line_rho = line_scores["rho"].reshape(count, 5)
    line_selected, line_feasible = select_candidate(
        line_rho, line_scores["novel"].reshape(count, 5), line_error, selected_rho,
    )
    candidates = {
        "rare_replay": anchors,
        "axis_gaussian": gaussian(anchors, np.random.default_rng(20270000 + start)),
        "part_mix": part_mix(anchors, partners, np.random.default_rng(20280000 + start)),
        "posetrans_3d": posetrans_3d(anchors, np.random.default_rng(20290000 + start)),
        "poseaug_3d": poseaug_3d(anchors, np.random.default_rng(20300000 + start)),
        "rarity_preserving_cmat": line_poses[np.arange(count), line_selected],
    }
    saved = dict(
        anchor_rows=anchor_rows[anchor_indices],
        partner_rows=anchor_rows[np.asarray(partner_indices)], subject=subjects,
        bank_quaternion=bank_poses, bank_rho=bank_rho, bank_chart=bank_error,
        bank_feasible=bank_feasible, bank_selected=selected,
        bank_diagnostics=np.stack([bank["diagnostics"] for bank in banks]),
        line_quaternion=line_poses, line_rho=line_rho, line_chart=line_error,
        line_feasible=line_feasible, line_selected=line_selected,
    )
    for name, poses in candidates.items():
        scores = final_screen(poses, sources, subjects, rarity_model, decoder, pair_thresholds)
        saved.update({f"{name}__{key}": value for key, value in scores.items()})
    return saved


def run_blocks(output_directory, rarity_model, anchor_rows, reference_sources,
               decoder, pair_thresholds, starts=range(0, 50000, 250), block_size=250):
    """Atomic NPZ blocks allow a remote job to resume after disconnection.

    Use a new output directory whenever inputs, index, calibration or settings
    change. Existing blocks must have the requested attempt count and fields.
    The experiment runner records full input hashes beside these blocks.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    for start in starts:
        destination = output_directory / f"block_{start:05d}.npz"
        if destination.exists():
            with np.load(destination, allow_pickle=False) as saved:
                if len(saved["anchor_rows"]) != block_size:
                    raise ValueError("Saved block size differs: use a fresh directory")
                for method in METHODS:
                    if f"{method}__accepted" not in saved:
                        raise ValueError("Existing block is incomplete")
            continue
        saved = generate_block(rarity_model, anchor_rows, reference_sources, decoder,
                               pair_thresholds, start, block_size)
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **saved)
        temporary.replace(destination)
        yield start, {method: int(saved[f"{method}__accepted"].sum()) for method in METHODS}
