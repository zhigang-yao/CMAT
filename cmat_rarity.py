"""Exact reranking and independent subject/sequence-weighted rarity calibration.

Public quaternions are scalar-first. Optional FAISS search embeds rotation
matrices before exact SO(3) reranking. Geometry uses the complete reference;
rarity and calibration use matched subject exclusion when requested.
"""
from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from cmat_rotations import distance, scalar_last

__all__ = [
    "hierarchy_weights",
    "sequence_score",
    "weighted_cdf",
    "weighted_quantile",
    "rotation_embedding",
    "RarityModel",
    "ExhaustiveCandidates",
    "FaissCandidates",
    "SequenceRetriever",
]


# Independent weighted calibration

def hierarchy_weights(subjects: np.ndarray, sequences: np.ndarray) -> np.ndarray:
    """Equal subjects, then equal sequences, then equal sampled poses."""
    subjects, sequences = np.asarray(subjects), np.asarray(sequences)
    if subjects.shape != sequences.shape or subjects.ndim != 1 or not len(subjects):
        raise ValueError("Nonempty aligned subject/sequence vectors are required")
    weights = np.zeros(len(subjects))
    unique_subjects = np.unique(subjects)
    for subject in unique_subjects:
        subject_mask = subjects == subject
        unique_sequences = np.unique(sequences[subject_mask])
        for sequence in unique_sequences:
            mask = subject_mask & (sequences == sequence)
            weights[mask] = 1 / len(unique_subjects) / len(unique_sequences) / mask.sum()
    return weights


def _distribution(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (values.ndim != 1 or values.shape != weights.shape or not len(values)
            or np.isnan(values).any() or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Invalid weighted empirical distribution")
    order = np.argsort(values)
    return values[order], np.cumsum(weights[order] / weights.sum())


def weighted_cdf(values, weights, query) -> np.ndarray:
    ordered, cumulative = _distribution(values, weights)
    count = np.searchsorted(ordered, query, side="right")
    return np.r_[0.0, cumulative][count]


def weighted_quantile(values, weights, probability: float) -> float:
    if not 0 <= probability <= 1:
        raise ValueError("Quantile probability must lie in [0,1]")
    ordered, cumulative = _distribution(values, weights)
    index = min(np.searchsorted(cumulative, probability), len(ordered) - 1)
    return float(ordered[index])


def sequence_score(ordered_sequence_distances: np.ndarray) -> float:
    distances = np.asarray(ordered_sequence_distances, dtype=np.float64)
    if len(distances) < 16 or np.any(distances < 0) or np.isnan(distances).any():
        raise ValueError("At least 16 nonnegative distinct-sequence distances required")
    means = [distances[:count].mean() for count in (4, 8, 16)]
    # The formula has no upper-percentile rejection or arbitrary distance floor.
    # If an entire scale consists of exact duplicates, log(0) is -infinity.
    with np.errstate(divide="ignore"):
        return float(np.mean(np.log(means)))


# Reference retrieval

def rotation_embedding(quaternions: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    values = Rotation.from_quat(scalar_last(quaternions).reshape(-1, 4))
    return values.as_matrix().reshape(len(quaternions), 189).astype("float32")


class ExhaustiveCandidates:
    """Reference implementation for small tests, not a substitute for IVF timing."""
    def __init__(self, reference_size: int):
        self.reference_size = reference_size

    def search(self, query: np.ndarray, count: int) -> np.ndarray:
        return np.arange(self.reference_size, dtype=np.int64)

    def search_batch(self, queries: np.ndarray, count: int):
        return [self.search(query, count) for query in queries]


class FaissCandidates:
    """Paper backend: PCA64, IVF1024, 64 probed cells; exact SO(3) reranking.

    Loading the frozen index and its row map gives the closest reproduction.
    Rebuilding an approximate index can change boundary scores; it must be
    followed by recalibration, not combined with a different saved calibration.
    """
    def __init__(self, index, pca, reference_rows, nprobe: int = 64):
        self.index, self.pca = index, pca
        self.reference_rows = np.asarray(reference_rows, dtype=np.int64)
        self.index.nprobe = min(nprobe, self.index.nlist)
        if self.index.ntotal != len(self.reference_rows):
            raise ValueError("Index and row mapping have different lengths")

    def search(self, query: np.ndarray, count: int) -> np.ndarray:
        embedding = self.pca.apply_py(rotation_embedding(query[None]))
        _, local = self.index.search(embedding, min(count, len(self.reference_rows)))
        return self.reference_rows[local[0][local[0] >= 0]]

    def search_batch(self, queries: np.ndarray, count: int):
        embedding = self.pca.apply_py(rotation_embedding(queries))
        _, local = self.index.search(embedding, min(count, len(self.reference_rows)))
        return [self.reference_rows[row[row >= 0]] for row in local]

    @classmethod
    def load(cls, index_path, pca_path, row_path, nprobe=64):
        import faiss
        return cls(faiss.read_index(str(index_path)),
                   faiss.read_VectorTransform(str(pca_path)),
                   np.load(row_path, allow_pickle=False), nprobe)

    @classmethod
    def build(cls, reference, directory, reference_permutation=None,
              seed=20260915, nprobe=64):
        """Build the documented frozen design without any evaluation queries."""
        import faiss
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        count = len(reference)
        if count < 500000:
            raise ValueError("Paper IVF1024 backend expects >=500,000 poses")
        permutation = (np.random.default_rng(seed).permutation(count)
                       if reference_permutation is None else reference_permutation)
        rows = np.arange(count, dtype=np.int64)
        pca = faiss.PCAMatrix(189, 64)
        pca.train(rotation_embedding(reference[np.asarray(permutation[:20000])]))
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(64), 64, 1024, faiss.METRIC_L2)
        index.cp.niter, index.cp.seed = 12, seed
        training = rows[np.random.default_rng(42).choice(count, min(60000, count), replace=False)]
        index.train(pca.apply_py(rotation_embedding(reference[training])))
        for start in range(0, count, 20000):
            index.add(pca.apply_py(rotation_embedding(reference[start:start + 20000])))
        faiss.write_index(index, str(directory / "reference.faiss"))
        faiss.write_VectorTransform(pca, str(directory / "pca.faiss"))
        np.save(directory / "reference_rows.npy", rows)
        return cls(index, pca, rows, nprobe)


class SequenceRetriever:
    """Keep complete-reference geometry separate from subject-excluded rarity."""
    def __init__(self, reference, sequence_ids, subject_ids, candidates=None, workers=1):
        self.reference = reference
        self.sequence_ids = np.asarray(sequence_ids)
        self.subject_ids = np.asarray(subject_ids)
        if len(reference) < 65:
            raise ValueError("At least 65 reference poses are needed")
        if len(reference) != len(self.sequence_ids) or len(reference) != len(self.subject_ids):
            raise ValueError("Reference and identifiers must align")
        # A global sequence identifier must not silently combine two subjects.
        order = np.argsort(self.sequence_ids)
        same = self.sequence_ids[order][1:] == self.sequence_ids[order][:-1]
        if np.any(same & (self.subject_ids[order][1:] != self.subject_ids[order][:-1])):
            raise ValueError("Sequence IDs must be globally unique across subjects")
        self.candidates = candidates or ExhaustiveCandidates(len(reference))
        self.workers = workers

    def _ordered(self, query, count):
        rows = self.candidates.search(query, count)
        distances = distance(query, self.reference[rows])
        order = np.argsort(distances)
        return rows[order], distances[order]

    def query(self, query, excluded_subject=None, candidate_rows=None) -> dict:
        if candidate_rows is None:
            rows, distances = self._ordered(query, 2048)
        else:
            distances = distance(query, self.reference[candidate_rows])
            order = np.argsort(distances)
            rows, distances = candidate_rows[order], distances[order]
        # Geometry and separation use this complete-reference search, even when
        # rarity has to expand search to recover 16 nonexcluded sequences.
        neighbors, nearest = rows[:65], float(distances[0])
        if len(neighbors) < 65:
            raise RuntimeError("Search returned fewer than 65 geometric neighbors")
        for budget in (2048, 8192, 32768, 65536):
            if budget != 2048:
                rows, distances = self._ordered(query, budget)
            keep = (np.ones(len(rows), dtype=bool) if excluded_subject is None
                    else self.subject_ids[rows] != excluded_subject)
            eligible_rows, eligible_distances = rows[keep], distances[keep]
            _, first = np.unique(self.sequence_ids[eligible_rows], return_index=True)
            first = np.sort(first)
            if len(first) >= 16:
                distinct = eligible_distances[first[:16]]
                return dict(score=sequence_score(distinct), nearest=nearest,
                            neighbors=neighbors, sequence_distances=distinct)
        raise RuntimeError("Fewer than 16 distinct sequences after subject exclusion")

    def batch(self, queries, excluded_subjects=None) -> dict:
        if not len(queries):
            raise ValueError("At least one query is required")
        excluded = ([None] * len(queries) if excluded_subjects is None
                    else np.broadcast_to(excluded_subjects, (len(queries),)))
        results = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for start in range(0, len(queries), 128):
                current = queries[start:start + 128]
                candidates = self.candidates.search_batch(current, 2048)
                items = zip(current, excluded[start:start + 128], candidates)
                results.extend(pool.map(lambda item: self.query(*item), items))
        return {key: np.stack([row[key] for row in results]) for key in results[0]}


# Source-specific calibrated rarity

class RarityModel:
    """Calibrate each source on independent subjects and never tune on queries.

    Subject identifiers must have one global meaning in reference, calibration
    and queries. Use separately resolved true identities, not directory names
    or shape-proxy clusters that can overlap across splits.
    """
    def __init__(self, retriever, calibration_poses, subjects, sequences, sources,
                 complete_reference_scores=None):
        self.retriever = retriever
        self.calibration_poses = np.asarray(calibration_poses)
        self.subjects, self.sequences = np.asarray(subjects), np.asarray(sequences)
        self.sources = np.asarray(sources)
        if any(len(v) != len(calibration_poses)
               for v in (self.subjects, self.sequences, self.sources)):
            raise ValueError("Calibration metadata and poses must align")
        if np.intersect1d(self.subjects, retriever.subject_ids).size:
            raise ValueError("Calibration subjects overlap the reference")
        self.cache = {}
        self.weights, self.separation = {}, {}
        complete = (retriever.batch(self.calibration_poses)
                    if complete_reference_scores is None else complete_reference_scores)
        for key in ("score", "nearest"):
            if np.asarray(complete[key]).shape != (len(calibration_poses),):
                raise ValueError(f"Precomputed calibration {key} has the wrong shape")
        for source in np.unique(self.sources):
            mask = self.sources == source
            weights = hierarchy_weights(self.subjects[mask], self.sequences[mask])
            self.weights[source] = weights
            self.separation[source] = max(
                weighted_quantile(complete["nearest"][mask], weights, 0.01), 1e-6,
            )
            self.cache[(str(source), None)] = complete["score"][mask]

    def load_matched_calibrations(self, path):
        """Load frozen ``source__subject`` score vectors after input-hash checks.

        The caller must verify cache provenance against the reference, index,
        calibration row order and search settings before using this shortcut.
        This method validates dimensions and the presence of excluded subjects.
        In the historical archive, suffix -1 denotes the complete reference.
        """
        with np.load(path, allow_pickle=False) as saved:
            for name in saved.files:
                source, identifier = name.rsplit("__", 1)
                subject = None if identifier == "-1" else int(identifier)
                if source not in self.weights:
                    continue
                if subject is not None and subject not in self.retriever.subject_ids:
                    raise ValueError(f"Unknown cached excluded subject {subject}")
                values = saved[name]
                expected = int(np.sum(self.sources == source))
                if values.shape != (expected,) or np.isnan(values).any():
                    raise ValueError(f"Invalid matched calibration array {name}")
                self.cache[(source, subject)] = values

    def calibration(self, source: str, excluded_subject=None) -> np.ndarray:
        source = str(source)
        if source not in self.weights:
            raise KeyError(f"No independent calibration for source {source!r}")
        key = source, excluded_subject
        if key not in self.cache:
            mask = self.sources == source
            # Score calibration against precisely the same reduced reference as
            # the generated query. Never reuse the full-reference distribution.
            self.cache[key] = self.retriever.batch(
                self.calibration_poses[mask], excluded_subject,
            )["score"]
        return self.cache[key]

    def classify(self, poses, sources, excluded_subjects=None) -> dict:
        sources = np.broadcast_to(sources, (len(poses),))
        excluded = ([None] * len(poses) if excluded_subjects is None
                    else np.broadcast_to(excluded_subjects, (len(poses),)))
        raw = self.retriever.batch(poses, excluded_subjects)
        percentiles = np.empty(len(poses))
        novel = np.empty(len(poses), dtype=bool)
        groups = {}
        for row, (source, subject) in enumerate(zip(sources, excluded)):
            groups.setdefault((str(source), subject), []).append(row)
        for (source, subject), rows in groups.items():
            rows = np.asarray(rows)
            scores = self.calibration(source, subject)
            percentiles[rows] = weighted_cdf(scores, self.weights[source], raw["score"][rows])
            novel[rows] = raw["nearest"][rows] > self.separation[source]
        return dict(**raw, rho=percentiles, rare=percentiles >= 0.95, novel=novel)
