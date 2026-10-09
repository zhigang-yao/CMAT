"""Product SO(3) operations for 21 local joints; quaternions are (w,x,y,z).

Float32 conversions preserve the original generation computation; distances
normalize and accumulate in float64 in the original rarity-retrieval order.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "canonicalize",
    "from_axis_angle",
    "to_axis_angle",
    "multiply",
    "relative_log",
    "exp_map",
    "distance",
    "scalar_last",
    "scalar_first",
    "cap_joint",
]


# Quaternion operations

def canonicalize(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float32)
    if value.shape[-1] != 4 or not np.isfinite(value).all():
        raise ValueError("Expected finite quaternions with final dimension 4")
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("A zero quaternion is not a rotation")
    value = value / np.maximum(norm, 1e-12)
    return np.where(value[..., :1] < 0, -value, value)


def from_axis_angle(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape[-1] != 3:
        raise ValueError("Expected rotation vectors with final dimension 3")
    angle = np.linalg.norm(vector, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.where(
        angle > 1e-9,
        np.sin(half) / np.maximum(angle, 1e-12),
        0.5 - angle * angle / 48.0,
    )
    return canonicalize(np.concatenate([np.cos(half), vector * scale], axis=-1))


def to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = canonicalize(quaternion)
    vector = quaternion[..., 1:]
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2 * np.arctan2(norm, np.maximum(quaternion[..., :1], 1e-12))
    scale = np.where(norm > 1e-9, angle / np.maximum(norm, 1e-12), 2.0)
    return vector * scale


def multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    aw, av = left[..., :1], left[..., 1:]
    bw, bv = right[..., :1], right[..., 1:]
    return np.concatenate(
        [aw * bw - np.sum(av * bv, axis=-1, keepdims=True),
         aw * bv + bw * av + np.cross(av, bv)], axis=-1,
    )


def relative_log(anchor: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Principal Log(R_anchor.T R_target), independently at every joint."""
    inverse = np.concatenate([anchor[..., :1], -anchor[..., 1:]], axis=-1)
    return to_axis_angle(multiply(inverse, target))


def exp_map(anchor: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Right-compose each joint with Exp(tangent)."""
    return canonicalize(multiply(anchor, from_axis_angle(tangent)))


def distance(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """One pose to an array of poses; output is product angular distance."""
    # Sum in the frozen scalar-last retrieval order as well as float64. Merely
    # reordering summands can perturb exact-neighbor ties at machine precision.
    query = np.asarray(scalar_last(query), dtype=np.float64)
    reference = np.asarray(scalar_last(reference), dtype=np.float64)
    query = query / np.linalg.norm(query, axis=-1, keepdims=True)
    reference = reference / np.linalg.norm(reference, axis=-1, keepdims=True)
    dot = np.abs(np.sum(query[None] * reference, axis=-1))
    return np.sqrt(np.sum((2 * np.arccos(np.clip(dot, 0, 1))) ** 2, axis=-1))


def scalar_last(quaternion: np.ndarray) -> np.ndarray:
    """Explicit conversion for SciPy/FAISS rotation-matrix embedding."""
    return np.asarray(quaternion)[..., [1, 2, 3, 0]]


def scalar_first(quaternion: np.ndarray) -> np.ndarray:
    return np.asarray(quaternion)[..., [3, 0, 1, 2]]


def cap_joint(step: np.ndarray, degrees: float = 12.0) -> np.ndarray:
    if degrees < 0:
        raise ValueError("Joint cap must be nonnegative")
    step = np.asarray(step)
    rows = step.reshape(-1, 3)
    norm = np.linalg.norm(rows, axis=-1)
    factors = np.minimum(1, np.deg2rad(degrees) / np.maximum(norm, 1e-15))
    return (rows * factors[:, None]).reshape(step.shape)
