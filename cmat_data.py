"""Explicit-input CMAT data preparation, frozen noise, and optional SMPL decoding.

No data downloads or filesystem discovery are performed. SMPL weights are
separately licensed, and torch/smplx are needed only when SMPLDecoder is used.
"""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np

# Noise
# The paper's sole synthetic noise recipe, with its frozen random stream.

NOISE_SEED_TAG = "cmat-expanded-30subjects-20260918"
PAPER_SPLIT = "reference_rare_test"


def noise_seed_words(subject, source_file, source_start, replicate, split=PAPER_SPLIT):
    """Preserve original manifest strings, including path separators and case.

    The old tag is retained only because it identifies the frozen random stream;
    it does not select the earlier 30-subject cohort.
    """
    if replicate not in (0, 1):
        raise ValueError("the paper uses noise replicates 0 and 1")
    replicate = int(replicate)
    if int(source_start) != source_start or source_start < 0:
        raise ValueError("source_start must be a nonnegative integer")
    if not subject or not source_file:
        raise ValueError("subject and source_file must be nonempty manifest strings")
    key = f"{subject}|{source_file}|{int(source_start)}"
    digest = hashlib.sha256(
        f"{NOISE_SEED_TAG}|noise|{split}|{replicate}|{key}".encode("utf-8")
    ).hexdigest()
    return [int(digest[i : i + 8], 16) for i in range(0, 32, 8)]


def correlated_tangent_noise(seed_words, frames=240):
    """Radians, (frames, 21, 3); 10-degree coordinate SD, 80% AR variance.

    AR states are stationary from frame zero. Generate the complete AR path
    before the IID array to reproduce the frozen observation cache exactly.
    """
    if not isinstance(frames, (int, np.integer)) or frames < 1:
        raise ValueError("frames must be a positive integer")
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))
    shape = (frames, 21, 3)
    rho = np.exp(-1 / 15)
    ar = np.empty(shape, dtype=np.float64)
    ar[0] = rng.normal(size=(21, 3))
    for t in range(1, frames):
        ar[t] = rho * ar[t - 1] + np.sqrt(1 - rho * rho) * rng.normal(size=(21, 3))
    return np.deg2rad(10.0) * (np.sqrt(0.2) * rng.normal(size=shape) + np.sqrt(0.8) * ar)


def corrupt_rotations(pose, subject, source_file, source_start, replicate):
    """Right-compose body rotations with noise, preserving root and end joints.

    Input is (240,24,3) SMPL axis angles; output retains the input floating dtype.
    Returns (noisy_pose, tangent_noise_float32), matching the archived cache.
    Neither target geometry nor correction results enter the random seed.
    """
    from scipy.spatial.transform import Rotation

    pose = np.asarray(pose)
    if pose.shape != (240, 24, 3) or pose.dtype.kind != "f" or not np.isfinite(pose).all():
        raise ValueError("pose must be a finite floating (240,24,3) array")
    seed = noise_seed_words(subject, source_file, source_start, replicate)
    tangent = correlated_tangent_noise(seed)
    clean = Rotation.from_rotvec(pose[:, 1:22].reshape(-1, 3).astype(float))
    error = Rotation.from_rotvec(tangent.reshape(-1, 3))
    noisy = pose.copy()
    noisy[:, 1:22] = (clean * error).as_rotvec().reshape(240, 21, 3)
    return noisy, tangent.astype(np.float32)


# Selection
# Rare-first clip selection from already calibrated reference anchors.
#
# No rarity fitting, database discovery, motion-error ranking or test-time tuning
# occurs here. Callers supply the eligible, identity-excluded anchor pool.

SELECTION_TAG = "amass-reference-rarity-first-20260921"


@dataclass(frozen=True)
class SourceSequence:
    subject: str
    subject_id: int
    source_file: str
    source_fps: float
    source_frames: int
    file_id: int = -1
    source_sha256: str = ""


@dataclass(frozen=True)
class RareAnchor:
    reference_row: int
    source_file: str
    source_frame: int
    percentile: float
    score: float = float("nan")


def _rank(value):
    return hashlib.sha256(f"{SELECTION_TAG}|{value}".encode("utf-8")).hexdigest()


def frame_window(anchor, source_fps, source_frames):
    """240 samples at 30 Hz, preserving the exact anchor source frame.

    Prefer anchor position 119, then the closest feasible position. Use NumPy
    round-to-even source sampling, exactly as in the frozen selection code.
    Return (integer source indices, anchor position), or None if too short.
    """
    if not np.isfinite(source_fps) or source_fps < 30:
        raise ValueError("source fps must be finite and at least 30")
    if int(anchor) != anchor or int(source_frames) != source_frames:
        raise ValueError("anchor and source_frames must be integers")
    if not 0 <= anchor < source_frames:
        raise ValueError("anchor is outside the source sequence")
    anchor, source_frames = int(anchor), int(source_frames)
    stride = source_fps / 30
    for left in sorted(range(240), key=lambda i: (abs(i - 119), i)):
        frames = anchor + np.rint((np.arange(240) - left) * stride).astype(int)
        if frames[0] >= 0 and frames[-1] < source_frames:
            if not np.all(np.diff(frames) > 0):
                raise ValueError("sampling would duplicate source frames")
            return frames, left
    return None


def select_rare_windows(anchors, sequences):
    """Return manifest rows and rejected-sequence reasons.

    `sequences` contains SourceSequence records; subject_id preserves the
    original library subject ordering. `anchors` contains RareAnchor records
    from the eligible pool; percentile < .95 rows are ignored. At most four
    clips per subject, different files first, then nonoverlapping extra clips.
    """
    sequences = list(sequences)
    metadata = {s.source_file: s for s in sequences}
    if len(metadata) != len(sequences):
        raise ValueError("source_file must identify one unique source sequence")
    subject_ids, subjects = {}, {}
    for sequence in sequences:
        if sequence.subject_id in subjects and subjects[sequence.subject_id] != sequence.subject:
            raise ValueError("subject_id refers to multiple canonical subjects")
        if sequence.subject in subject_ids and subject_ids[sequence.subject] != sequence.subject_id:
            raise ValueError("canonical subject has multiple subject IDs")
        subjects[sequence.subject_id] = sequence.subject
        subject_ids[sequence.subject] = sequence.subject_id
    pools = defaultdict(lambda: defaultdict(list))
    seen = set()
    for anchor in sorted(anchors, key=lambda a: a.reference_row):
        if anchor.reference_row in seen:
            raise ValueError("duplicate reference row")
        seen.add(anchor.reference_row)
        if not np.isfinite(anchor.percentile) or not 0 <= anchor.percentile <= 1:
            raise ValueError("anchor percentile must be finite and in [0,1]")
        if anchor.source_file not in metadata:
            raise ValueError("anchor has no source metadata")
        if anchor.percentile >= 0.95:
            sequence = metadata[anchor.source_file]
            pools[sequence.subject_id][anchor.source_file].append(anchor)
    selected, rejected = [], []
    for subject_id in sorted(pools):
        queues = {}
        for filename in sorted(pools[subject_id], key=lambda p: _rank(f"file|{p}")):
            sequence = metadata[filename]
            queue = sorted(
                pools[subject_id][filename],
                key=lambda a: _rank(f"anchor|{filename}|{a.source_frame}"),
            )
            if (
                frame_window(queue[0].source_frame, sequence.source_fps, sequence.source_frames)
                is None
            ):
                rejected.append(
                    {
                        "subject": sequence.subject,
                        "source_file": filename,
                        "reason": "less than 240 samples at 30 Hz",
                    }
                )
            else:
                queues[filename] = queue
        occupied = defaultdict(list)
        chosen = 0
        while chosen < 4 and queues:
            made = False
            for filename in list(queues):
                sequence = metadata[filename]
                while queues[filename]:
                    anchor = queues[filename].pop(0)
                    candidate = frame_window(
                        anchor.source_frame, sequence.source_fps, sequence.source_frames
                    )
                    if candidate is None:
                        continue
                    frames, left = candidate
                    if any(
                        frames[0] <= stop and frames[-1] >= start
                        for start, stop in occupied[filename]
                    ):
                        continue
                    selected.append(
                        {
                            "index": len(selected),
                            "subject": sequence.subject,
                            "subject_id": subject_id,
                            "source": sequence.subject.split("/")[0],
                            "source_file": filename,
                            "file_id": sequence.file_id,
                            "source_sha256": sequence.source_sha256,
                            "anchor_reference_row": anchor.reference_row,
                            "anchor_source_frame": anchor.source_frame,
                            "anchor_position": int(left),
                            "anchor_percentile": anchor.percentile,
                            "anchor_score": anchor.score,
                            "source_start": int(frames[0]),
                            "source_last": int(frames[-1]),
                            "stride": sequence.source_fps / 30,
                            "source_fps": sequence.source_fps,
                            "source_frames": sequence.source_frames,
                        }
                    )
                    occupied[filename].append((int(frames[0]), int(frames[-1])))
                    chosen += 1
                    made = True
                    break
                if not queues[filename]:
                    del queues[filename]
                if chosen == 4:
                    break
            if not made:
                break
    return selected, rejected


# Smpl
# Optional SMPL decoding with user-supplied, separately licensed models.


def load_amass_clip(path, source_indices, expected_sha256=None):
    """Read one already-selected source file; do not discover or resample data.

    Keep AMASS root + 21 body rotations; the two unused SMPL joints stay zero.
    Return float32 (frames,24,3) axis angles and (frames,10) shape coefficients.
    """
    path = Path(path)
    if expected_sha256:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("source file does not match the frozen manifest hash")
    indices = np.asarray(source_indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) == 0:
        raise ValueError("source_indices must be a nonempty integer vector")
    if np.any(indices < 0) or np.any(np.diff(indices) <= 0):
        raise ValueError("source indices must be nonnegative and increasing")
    with np.load(path, allow_pickle=False) as archive:
        raw = archive["poses"]
        if raw.ndim != 2 or raw.shape[1] < 66 or indices[-1] >= len(raw):
            raise ValueError("source poses do not contain the requested body frames")
        pose = np.zeros((len(indices), 24, 3), np.float32)
        pose[:, :22] = raw[indices, :66].reshape(len(indices), 22, 3)
        beta = np.zeros(10, np.float32)
        if "betas" in archive:
            raw_beta = np.asarray(archive["betas"]).reshape(-1)
            beta[: min(10, len(raw_beta))] = raw_beta[:10]
    if not np.isfinite(pose).all() or not np.isfinite(beta).all():
        raise ValueError("source clip has nonfinite rotations or shape")
    return pose, np.broadcast_to(beta, (len(indices), 10)).copy()


class SMPLDecoder:
    """Decode to the first 22 joints in metres using a neutral SMPL model.

    Optional dependencies: torch and smplx. Model weights are never bundled.
    (N,21,3) inputs are local body rotations: zero root, beta and final joints.
    (N,24,3) inputs preserve their root; provide shape for controlled AMASS.
    """

    def __init__(self, model_path, device=None, batch_size=256):
        try:
            import torch
            import smplx
        except ImportError as error:
            raise ImportError("SMPL decoding needs optional packages torch and smplx") from error
        if batch_size < 1 or int(batch_size) != batch_size:
            raise ValueError("batch_size must be a positive integer")
        if not Path(model_path).exists():
            raise FileNotFoundError("supply the path to your licensed neutral SMPL models")
        self.torch = torch
        self.smplx = smplx
        self.model_path = str(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = int(batch_size)
        self.models = {}

    def __call__(self, pose, shape=None):
        pose = np.asarray(pose)
        if pose.ndim != 3 or pose.shape[1:] not in ((21, 3), (24, 3)):
            raise ValueError("axis angles must have shape (N,21,3) or (N,24,3)")
        if not np.isfinite(pose).all():
            raise ValueError("axis angles must be finite")
        if pose.shape[1] == 21:
            full = np.zeros((len(pose), 24, 3), np.float32)
            full[:, 1:22] = pose
            pose = full
        if shape is None:
            shape = np.zeros((len(pose), 10), np.float32)
        shape = np.asarray(shape)
        if shape.shape == (10,):
            shape = np.broadcast_to(shape, (len(pose), 10))
        if shape.shape != (len(pose), 10) or not np.isfinite(shape).all():
            raise ValueError("shape must be finite (N,10) or (10,)")
        if len(pose) == 0:
            return np.empty((0, 22, 3), np.float32)
        blocks = []
        with self.torch.no_grad():
            for start in range(0, len(pose), self.batch_size):
                end = min(start + self.batch_size, len(pose))
                n = end - start
                if n not in self.models:
                    model = self.smplx.create(
                        self.model_path,
                        model_type="smpl",
                        gender="neutral",
                        use_pca=False,
                        batch_size=n,
                    )
                    self.models[n] = model.to(self.device).eval()
                p = self.torch.as_tensor(
                    pose[start:end], dtype=self.torch.float32, device=self.device
                )
                b = self.torch.as_tensor(
                    shape[start:end].copy(), dtype=self.torch.float32, device=self.device
                )
                output = self.models[n](
                    global_orient=p[:, 0], body_pose=p[:, 1:].reshape(n, 69), betas=b
                )
                blocks.append(output.joints[:, :22].cpu().numpy().astype(np.float32))
        return np.concatenate(blocks)


# Eft
# Convert independently obtained official SmoothNet 3DPW--EFT arrays.

JOINT_ORDER = (6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10)


def _unpack_tracks(value):
    if isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype.kind == "O":
        value = value.item()
    if isinstance(value, dict):
        return value
    if isinstance(value, np.ndarray) and value.ndim >= 1:
        return value
    raise ValueError("Expected an array of tracks or a keyed track dictionary")


def eft_records(detected, groundtruth, trust_pickle=False):
    """Yield paired 14-joint tracks, preserving original row order and IDs.

    The official variable-length release uses object arrays. Enable
    trust_pickle only for the authenticated official files you obtained.
    Newly exported standard CMAT NPZ files never require pickle.
    """
    with np.load(detected, allow_pickle=trust_pickle) as archive:
        observations, names = archive["joints_3d"], archive["imgname"]
    with np.load(groundtruth, allow_pickle=trust_pickle) as archive:
        targets, target_names = archive["joints_3d"], archive["imgname"]
    observations, targets, names, target_names = map(
        _unpack_tracks, (observations, targets, names, target_names)
    )
    if not len(observations) == len(targets) == len(names) == len(target_names):
        raise ValueError("Detected/target sequence counts differ")
    containers = (observations, targets, names, target_names)
    if any(isinstance(value, dict) for value in containers):
        if not all(isinstance(value, dict) for value in containers):
            raise ValueError("Cannot pair keyed and unkeyed track containers")
        keys = list(observations)
        if any(set(value) != set(keys) for value in containers):
            raise ValueError("Detected/target track dictionary keys differ")
    else:
        keys = list(range(len(observations)))
    for index, key in enumerate(keys):
        observed, target, images, paired_images = (value[key] for value in containers)
        images = np.asarray(images, dtype=str).reshape(-1)
        paired_images = np.asarray(paired_images, dtype=str).reshape(-1)
        if not np.array_equal(images, paired_images):
            raise ValueError(f"Track {index} has unpaired image identifiers")
        x = np.asarray(observed, dtype=float).reshape(len(observed), -1, 3)
        y = np.asarray(target, dtype=float).reshape(len(target), -1, 3)
        if x.shape != y.shape or len(images) != len(x):
            raise ValueError(f"Track {index} has inconsistent coordinates")
        if (
            len(x) == 0
            or x.shape[1] <= max(JOINT_ORDER)
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError(
                f"Track {index} requires finite coordinates for the official 17-joint layout"
            )
        ids = []
        for name in images:
            match = re.search(r"(\d+)(?:\.[^.]+)?$", name.replace("\\", "/").rsplit("/", 1)[-1])
            if match is None:
                raise ValueError("Cannot recover original frame ID")
            ids.append(int(match.group(1)))
        if np.any(np.diff(ids) <= 0):
            raise ValueError(f"Track {index} frame IDs are not strictly increasing")
        yield dict(
            index=index,
            observed=x[:, JOINT_ORDER],
            target=y[:, JOINT_ORDER],
            frame_ids=np.asarray(ids),
            names=images,
        )
