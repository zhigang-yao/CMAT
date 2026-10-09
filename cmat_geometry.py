"""Weighted PCA, finite cylinders and generation attention (Supplement C.2–3)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from cmat_rotations import (
    cap_joint, canonicalize, distance, exp_map, from_axis_angle,
    relative_log, scalar_first, scalar_last, to_axis_angle,
)

__all__ = [
    "LocalChart",
    "chart",
    "cylinder",
    "fit",
    "project",
    "canonicalize",
    "from_axis_angle",
    "to_axis_angle",
    "relative_log",
    "exp_map",
    "distance",
    "scalar_last",
    "scalar_first",
    "cap_joint",
]


# Local manifold chart

HEAD_COEFFICIENTS = np.array([
    [1.0, 1.0, 0.25], [4.0, 1.0, 0.25],
    [1.0, 4.0, 0.25], [0.25, 0.25, 4.0],
])


@dataclass
class LocalChart:
    mean: np.ndarray
    basis: np.ndarray
    singular: np.ndarray
    weights: np.ndarray
    scale_squared: float


def project(value: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Orthogonal projection; basis stores orthonormal directions as rows."""
    return (value @ basis.T) @ basis


def chart(residuals: np.ndarray, rank: int = 12) -> LocalChart:
    x = np.asarray(residuals, dtype=np.float64)
    if (x.ndim != 2 or rank < 1 or len(x) < rank + 2
            or x.shape[1] < rank or not np.isfinite(x).all()):
        raise ValueError("The finite local chart requires K >= rank+2")
    squared = np.einsum("kd,kd->k", x, x)
    scale_squared = max(float(np.median(squared)), 1e-16)
    weights = np.exp(-np.minimum(squared / scale_squared, 700))
    weights /= weights.sum()
    mean = weights @ x
    _, singular, basis = np.linalg.svd(
        (x - mean) * np.sqrt(weights[:, None]), full_matrices=False,
    )
    return LocalChart(mean, basis[:rank], singular, weights, scale_squared)


def cylinder(residuals: np.ndarray, local: LocalChart,
             minimum_support: int | None = None) -> dict:
    """Build one bounded cylinder from the preliminary spherical mean.

    The axis is not assumed to be a tangent or normal direction. Radius and
    half-length expand together until the requested comparison count fits.
    """
    x = np.asarray(residuals, dtype=np.float64)
    count, dimension = x.shape
    rank = len(local.basis)
    scale = np.sqrt(local.scale_squared)
    centered = x - local.mean
    normal = centered - project(centered, local.basis)
    variance = np.sum(local.weights * np.sum(normal * normal, axis=1))
    sigma = np.clip(np.sqrt(variance / max(dimension - rank, 1)) / scale,
                    1e-5, 0.95)
    radius0 = 5 * sigma / np.log10(max(count, 2)) * scale
    half0 = 10 * sigma * np.sqrt(np.log(1 / sigma)) / np.log10(max(count, 2)) * scale
    distances = np.sqrt(np.sum(x * x, axis=1))
    fifth = np.partition(distances, min(5, count) - 1)[min(5, count) - 1]
    sphere = max(radius0, 1.1 * fifth, 1e-12)
    spherical_weights = np.maximum(1 - (distances / sphere) ** 2, 0) ** 3
    spherical_weights /= max(spherical_weights.sum(), 1e-300)
    initial_mean = spherical_weights @ x
    norm = np.linalg.norm(initial_mean)
    degenerate = norm <= 1e-12 * scale
    axis = local.basis[0] if degenerate else initial_mean / max(norm, 1e-300)
    axial = x @ axis
    radial = np.sqrt(np.maximum(np.sum(x * x, axis=1) - axial * axial, 0))
    ratio = np.maximum(radial / max(radius0, 1e-15),
                       abs(axial) / max(half0, 1e-15))
    requested = max(11, rank + 2) if minimum_support is None else minimum_support
    minimum = min(requested, count)
    if minimum < 1:
        raise ValueError("minimum_support must be positive")
    enlargement = max(1.0, 1.1 * np.partition(ratio, minimum - 1)[minimum - 1])
    radius, half_length = radius0 * enlargement, half0 * enlargement
    inside = (radial < radius) & (abs(axial) < half_length)
    return dict(inside=inside, radius=float(radius), half_length=float(half_length),
                enlargement=float(enlargement), sigma=float(sigma),
                axis_degenerate=bool(degenerate), axis=axis)


def fit(residuals: np.ndarray, rank: int = 12, gain: float = 0.7,
        tangent_fraction: float = 0.1) -> dict:
    """Four fixed geometric heads and one gated CMAT generation increment.

    This is generation's operator. Correction has its own head refitting,
    covariance and trajectory solve; it must not call this update in place of
    those operations.
    """
    x = np.asarray(residuals, dtype=np.float64)
    local = chart(x, rank)
    support = cylinder(x, local)
    tangent = project(x, local.basis)
    normal = x - tangent
    parts = np.column_stack([
        (tangent * tangent).sum(1), (normal * normal).sum(1), (x * x).sum(1),
    ])
    scales = np.maximum(np.median(parts, axis=0), local.scale_squared * 1e-8)
    logits = -(parts / scales) @ HEAD_COEFFICIENTS.T
    logits[~support["inside"]] = -np.inf
    logits -= np.max(logits, axis=0)
    weights = np.exp(logits)
    weights /= weights.sum(axis=0)
    combined = weights @ np.full(4, 0.25)
    attended = combined @ x
    tangent_mean = project(attended, local.basis)
    normal_mean = attended - tangent_mean
    effective = 1 / (combined @ combined)
    identifiable = local.singular[rank - 1] > 1e-10 * max(local.singular[0], 1e-12)
    gate = gain * min(1, effective / (rank + 2)) if identifiable else 0.0
    step = gate * (normal_mean + tangent_fraction * tangent_mean)
    return dict(
        **support, step=step, basis=local.basis, mean=local.mean, weights=weights,
        attended=attended, gate=gate, normal_step=gate * normal_mean,
        tangent_step=gate * tangent_fraction * tangent_mean,
        effective_neighbors=float(effective), support=int(support["inside"].sum()),
        identifiable=bool(identifiable), scales=scales,
        flatness=float(np.sum(local.singular[:rank] ** 2)
                       / max(np.sum(local.singular ** 2), 1e-16)),
    )
