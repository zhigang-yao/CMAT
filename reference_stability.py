"""Independently recalibrated reference-size experiments with resumable FAISS indices."""
from __future__ import annotations


# Sampling and comparisons
from itertools import combinations

import numpy as np
from scipy.stats import rankdata

from cmat_rarity import hierarchy_weights, weighted_cdf

PAPER_SIZES = (100000, 500000, 1000000, 2000000, 2500000, 3000000)


def _percentiles(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("percentiles must be a nonempty finite vector")
    if np.any((values < -1e-12) | (values > 1 + 1e-12)):
        raise ValueError("percentiles must lie in [0,1], up to rounding error")
    return values


def calibrate_scores(calibration_scores, calibration_subjects, calibration_sequences,
                     calibration_sources, evaluation_scores, evaluation_sources):
    """Recalibrate one draw using independent, source-matched calibration rows.

    Pass freshly computed calibration scores for *this* reference draw. Never
    combine a subset's evaluation scores with the full-reference calibration.
    Subject-disjointness must be checked against reference/query metadata by
    the caller; scalar scores cannot establish who generated a pose.
    """
    calibration_scores = np.asarray(calibration_scores, dtype=float)
    evaluation_scores = np.asarray(evaluation_scores, dtype=float)
    subjects, sequences, sources = map(np.asarray, (
        calibration_subjects, calibration_sequences, calibration_sources))
    evaluation_sources = np.asarray(evaluation_sources)
    if any(v.shape != calibration_scores.shape for v in (subjects, sequences, sources)):
        raise ValueError("calibration scores and metadata must align")
    if calibration_scores.ndim != 1 or evaluation_scores.shape != evaluation_sources.shape:
        raise ValueError("scores and source metadata must be one-dimensional and aligned")
    if evaluation_scores.ndim != 1 or not np.isfinite(evaluation_scores).all():
        raise ValueError("evaluation scores must be a finite vector")
    if not np.isfinite(calibration_scores).all():
        raise ValueError("calibration scores must be finite")
    result = np.empty(len(evaluation_scores))
    for source in np.unique(evaluation_sources):
        calibration = sources == source
        evaluation = evaluation_sources == source
        if not calibration.any():
            raise ValueError(f"no independent calibration for source {source!r}")
        weights = hierarchy_weights(subjects[calibration], sequences[calibration])
        result[evaluation] = weighted_cdf(calibration_scores[calibration], weights,
                                          evaluation_scores[evaluation])
    return result


def compare_percentiles(first, second, subjects, sequences):
    """One fixed evaluation population, weighted labels and unweighted ranks.

    Empty rare-set union makes Jaccard undefined (NaN), not perfect agreement.
    Constant percentile ranks similarly give undefined Spearman correlation.
    """
    first, second = _percentiles(first), _percentiles(second)
    weights = hierarchy_weights(subjects, sequences)
    if first.shape != second.shape or first.shape != weights.shape:
        raise ValueError("percentile arrays and evaluation metadata must align")
    a, b = first >= 0.95, second >= 0.95
    rank_a, rank_b = rankdata(first, method="average"), rankdata(second, method="average")
    rank_a, rank_b = rank_a - rank_a.mean(), rank_b - rank_b.mean()
    denominator = np.linalg.norm(rank_a) * np.linalg.norm(rank_b)
    union = float(weights @ (a | b))
    return {
        "rare_fraction_a": float(weights @ a), "rare_fraction_b": float(weights @ b),
        "label_agreement": float(weights @ (a == b)),
        "rare_jaccard": float(weights @ (a & b) / union) if union > 0 else float("nan"),
        "spearman": float(rank_a @ rank_b / denominator) if denominator > 0 else float("nan"),
        "median_percentile_shift": float(np.median(np.abs(first - second))),
    }


def reference_draws(pool_rows, sizes=PAPER_SIZES, repeats=3, seed=20260915):
    """Yield (size, repeat, actual rows) under the historical nested draw rule.

    Each repeat uses permutation(seed + repeat*1009); sizes are nested within
    that repeat. At size equal to the finite pool there is exactly one unique
    subset, so yield it only once. No particular numeric size is privileged.
    """
    pool_rows = np.asarray(pool_rows)
    if pool_rows.ndim != 1 or pool_rows.dtype.kind not in "iu" or len(pool_rows) == 0:
        raise ValueError("pool_rows must be a nonempty integer vector")
    if len(np.unique(pool_rows)) != len(pool_rows):
        raise ValueError("pool rows must be distinct")
    sizes = tuple(sizes)
    if len(set(sizes)) != len(sizes) or not sizes:
        raise ValueError("reference sizes must be nonempty and distinct")
    if any(int(n) != n or not 0 < n <= len(pool_rows) for n in sizes):
        raise ValueError("each reference size must be a positive integer within the pool")
    if int(repeats) != repeats or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    sizes, repeats = tuple(int(n) for n in sizes), int(repeats)
    pool_rows = np.sort(pool_rows)
    for repeat in range(repeats):
        permutation = np.random.default_rng(seed + repeat * 1009).permutation(len(pool_rows))
        for size in sizes:
            if size == len(pool_rows) and repeat != 0:
                continue
            yield int(size), repeat, np.sort(pool_rows[permutation[:size]])


def summarize_runs(runs, subjects, sequences):
    """Compare all distinct pairs within each size, never a run with itself.

    Each run has reference_size, repeat, reference_rows and rho. Identical
    sampled row sets are explicitly recorded as skipped, even if their index
    training or ordering changed. They are not independent frame draws.
    """
    runs = list(runs)
    weights = hierarchy_weights(subjects, sequences)
    keys = [(run["reference_size"], run["repeat"]) for run in runs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate size/repeat identifier")
    summaries, pairs, skipped = [], [], []
    for run in runs:
        rho = _percentiles(run["rho"])
        rows = np.asarray(run["reference_rows"])
        if rho.shape != weights.shape or rows.ndim != 1 or len(rows) != run["reference_size"]:
            raise ValueError("run dimensions do not match its size and evaluation population")
        if len(np.unique(rows)) != len(rows):
            raise ValueError("a reference draw repeats rows")
        summaries.append({"reference_size": int(run["reference_size"]), "repeat": int(run["repeat"]),
                          "rare_fraction": float(weights @ (rho >= 0.95))})
    for size in sorted({run["reference_size"] for run in runs}):
        for a, b in combinations([run for run in runs if run["reference_size"] == size], 2):
            pair = {"reference_size": int(size), "repeat_a": int(a["repeat"]), "repeat_b": int(b["repeat"])}
            if np.array_equal(np.sort(a["reference_rows"]), np.sort(b["reference_rows"])):
                skipped.append({**pair, "reason": "identical reference row sets"})
                continue
            pairs.append({**pair, **compare_percentiles(a["rho"], b["rho"], subjects, sequences),
                          "interpretation": "different frame draws from the same finite library"})
    return {"runs": summaries, "pairs": pairs, "skipped_pairs": skipped}


# Per-draw recalibration
import hashlib
from pathlib import Path

import numpy as np


def array_digest(*values):
    """Fingerprint aligned inputs without serializing arrays to long text files."""
    digest = hashlib.sha256()
    for value in values:
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError("object arrays cannot be safely cached; use numeric or string arrays")
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def save_npz(path, **arrays):
    """Atomic binary cache update; incomplete work never becomes a finished run."""
    path = Path(path)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def run_stability(pool_rows, sizes, evaluation_poses, subjects, sequences, sources,
                  model_factory, output, provenance, repeats=3, seed=20260915):
    """Factory(reference_rows) must return a freshly calibrated RarityModel.

    Use StabilityFactory for the historical PCA/IVF backend or provide an
    explicit exhaustive model for small tests. `provenance` must identify the
    reference, calibration, search settings and code version; changing it
    invalidates saved runs. Evaluation cohorts remain fixed for all draws.
    No server path, database scan, threshold search or score reuse across
    different reference subsets is performed.
    """
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("provide nonempty reference/calibration/search provenance")
    subjects, sequences, sources = map(np.asarray, (subjects, sequences, sources))
    if any(v.shape != (len(evaluation_poses),) for v in (subjects, sequences, sources)):
        raise ValueError("evaluation metadata must align with poses")
    signature = array_digest(evaluation_poses, subjects, sequences, sources,
                             provenance, getattr(model_factory, "signature", "custom_factory"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for size, repeat, rows in reference_draws(pool_rows, sizes, repeats, seed):
        path = output / f"scores_{size}_{repeat}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if str(saved["signature"]) != signature or not np.array_equal(saved["reference_rows"], rows):
                    raise ValueError(f"cached run does not match current inputs: {path.name}")
                result = {key: saved[key].copy() for key in saved.files}
        else:
            model = model_factory(rows)
            if np.intersect1d(subjects, model.subjects).size:
                raise ValueError("evaluation subjects overlap independent calibration")
            if np.intersect1d(subjects, model.retriever.subject_ids).size:
                raise ValueError("evaluation subjects overlap reference subjects")
            classification = model.classify(evaluation_poses, sources)
            # Save the exact independent source calibrations used in this run.
            calibration_scores = np.empty(len(model.subjects), dtype=float)
            for source in np.unique(model.sources):
                calibration_scores[model.sources == source] = model.calibration(str(source))
            result = dict(reference_rows=rows, rho=np.asarray(classification["rho"]),
                          evaluation_score=np.asarray(classification["score"]),
                          calibration_score=calibration_scores,
                          calibration_subjects=model.subjects, calibration_sequences=model.sequences,
                          calibration_sources=model.sources, signature=np.asarray(signature),
                          reference_size=np.asarray(size), repeat=np.asarray(repeat))
            save_npz(path, **result)
        result["reference_size"], result["repeat"] = int(size), int(repeat)
        results.append(result)
    summary = summarize_runs(results, subjects, sequences)
    summary["interpretation"] = "same-library frame subsampling; not independently acquired libraries"
    return summary


# PCA and index factory
from pathlib import Path

import numpy as np

from cmat_rarity import FaissCandidates, RarityModel, SequenceRetriever
from cmat_rarity import rotation_embedding


class StabilityFactory:
    """Build/load each draw's index and independent calibration from explicit arrays.

    `reference_id` is a caller-verified reference file hash or manifest identity.
    Public quaternions are scalar-first, including calibration poses. Metadata
    must use global canonical identities, as required by RarityModel.
    No raw calibration scores are shared between different draw row sets.
    """

    def __init__(self, reference, sequence_ids, subject_ids, calibration_poses,
                 calibration_subjects, calibration_sequences, calibration_sources,
                 directory, reference_id, seed=20260915, nprobe=64, workers=1):
        if not reference_id:
            raise ValueError("provide verified reference provenance")
        self.reference = reference
        self.sequence_ids, self.subject_ids = np.asarray(sequence_ids), np.asarray(subject_ids)
        self.calibration_poses = np.asarray(calibration_poses)
        self.calibration_subjects = np.asarray(calibration_subjects)
        self.calibration_sequences = np.asarray(calibration_sequences)
        self.calibration_sources = np.asarray(calibration_sources)
        if len(reference) < 20000:
            raise ValueError("historical PCA factory needs at least 20,000 reference rows; use exhaustive models for small tests")
        if self.sequence_ids.shape != (len(reference),) or self.subject_ids.shape != (len(reference),):
            raise ValueError("reference metadata must align")
        if any(v.shape != (len(self.calibration_poses),) for v in
               (self.calibration_subjects, self.calibration_sequences, self.calibration_sources)):
            raise ValueError("calibration metadata must align")
        if np.intersect1d(self.subject_ids, self.calibration_subjects).size:
            raise ValueError("calibration subjects overlap the complete reference pool")
        self.seed, self.nprobe, self.workers = seed, nprobe, workers
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.signature = array_digest(
            reference_id, np.asarray(reference).shape, self.sequence_ids, self.subject_ids,
            self.calibration_poses, self.calibration_subjects,
            self.calibration_sequences, self.calibration_sources,
            seed, nprobe, "pca64_20k_shared_ivf512below500k_else1024_niter12_train60k_seed42",
        )
        self.pca = None

    def _shared_pca(self):
        import faiss
        if self.pca is not None:
            return self.pca
        path, metadata = self.directory / "shared_pca.faiss", self.directory / "shared_pca.npz"
        training_rows = np.random.default_rng(self.seed).permutation(len(self.reference))[:20000]
        if path.exists() and metadata.exists():
            with np.load(metadata, allow_pickle=False) as saved:
                if str(saved["signature"]) != self.signature or not np.array_equal(saved["training_rows"], training_rows):
                    raise ValueError("shared PCA cache does not match reference/calibration provenance")
            pca = faiss.read_VectorTransform(str(path))
        else:
            pca = faiss.PCAMatrix(189, 64)
            pca.train(rotation_embedding(self.reference[training_rows]))
            temporary = path.with_suffix(".partial.faiss")
            faiss.write_VectorTransform(pca, str(temporary))
            temporary.replace(path)
            save_npz(metadata, signature=np.asarray(self.signature), training_rows=training_rows)
        self.pca = pca
        return pca

    def __call__(self, reference_rows):
        import faiss
        rows = np.asarray(reference_rows)
        if rows.ndim != 1 or rows.dtype.kind not in "iu" or len(rows) < 512:
            raise ValueError("historical IVF draws need at least 512 distinct integer rows")
        if np.any(rows < 0) or np.any(rows >= len(self.reference)) or len(np.unique(rows)) != len(rows):
            raise ValueError("reference rows must be distinct and within the pool")
        rows = np.sort(rows)
        pca = self._shared_pca()
        draw_signature = array_digest(self.signature, rows)
        folder = self.directory / f"reference_{len(rows)}_{draw_signature[:16]}"
        folder.mkdir(exist_ok=True)
        index_path, row_path = folder / "reference.faiss", folder / "reference_rows.npz"
        if index_path.exists() and row_path.exists():
            with np.load(row_path, allow_pickle=False) as saved:
                if str(saved["signature"]) != draw_signature or not np.array_equal(saved["rows"], rows):
                    raise ValueError("reference subset cache mismatch")
            index = faiss.read_index(str(index_path))
        else:
            nlist = 512 if len(rows) < 500000 else 1024
            index = faiss.IndexIVFFlat(faiss.IndexFlatL2(64), 64, nlist, faiss.METRIC_L2)
            index.cp.niter, index.cp.seed = 12, self.seed
            local_train = np.random.default_rng(42).choice(len(rows), min(60000, len(rows)), replace=False)
            index.train(pca.apply_py(rotation_embedding(self.reference[rows[local_train]])))
            for start in range(0, len(rows), 20000):
                index.add(pca.apply_py(rotation_embedding(self.reference[rows[start:start + 20000]])))
            temporary = index_path.with_suffix(".partial.faiss")
            faiss.write_index(index, str(temporary))
            temporary.replace(index_path)
            save_npz(row_path, rows=rows, signature=np.asarray(draw_signature), nlist=np.asarray(nlist))
        candidates = FaissCandidates(index, pca, np.arange(len(rows)), nprobe=self.nprobe)
        retriever = SequenceRetriever(self.reference[rows], self.sequence_ids[rows], self.subject_ids[rows],
                                      candidates=candidates, workers=self.workers)
        calibration_path = folder / "calibration.npz"
        if calibration_path.exists():
            with np.load(calibration_path, allow_pickle=False) as saved:
                if str(saved["signature"]) != draw_signature:
                    raise ValueError("calibration cache provenance mismatch")
                complete = {"score": saved["score"], "nearest": saved["nearest"]}
        else:
            complete = retriever.batch(self.calibration_poses)
            save_npz(calibration_path, score=complete["score"], nearest=complete["nearest"],
                     signature=np.asarray(draw_signature))
        return RarityModel(retriever, self.calibration_poses, self.calibration_subjects,
                           self.calibration_sequences, self.calibration_sources,
                           complete_reference_scores=complete)


# Command
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table(folder, name, rows):
    if not rows:
        return
    for start in range(0, len(rows), 400):
        path = folder / f"{name}_{start // 400 + 1:03d}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows[start:start + 400])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sequences", type=Path, required=True)
    parser.add_argument("--reference-subjects", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=PAPER_SIZES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    reference = np.load(args.reference, mmap_mode="r", allow_pickle=False)
    sequences = np.load(args.reference_sequences, allow_pickle=False)
    subjects = np.load(args.reference_subjects, allow_pickle=False)
    with np.load(args.calibration, allow_pickle=False) as archive:
        calibration = {key: archive[key] for key in ("poses", "subjects", "sequences", "sources")}
    with np.load(args.evaluation, allow_pickle=False) as archive:
        evaluation = {key: archive[key] for key in ("poses", "subjects", "sequences", "sources")}
    factory = StabilityFactory(
        reference, sequences, subjects, calibration["poses"], calibration["subjects"],
        calibration["sequences"], calibration["sources"], args.output / "indices",
        reference_id=sha256(args.reference), seed=args.seed, workers=args.workers,
    )
    result = run_stability(
        np.arange(len(reference)), args.sizes, evaluation["poses"], evaluation["subjects"],
        evaluation["sequences"], evaluation["sources"], factory, args.output / "scores",
        provenance=factory.signature, repeats=args.repeats, seed=args.seed,
    )
    for name in ("runs", "pairs", "skipped_pairs"):
        table(args.output, name, result[name])
    summary = dict(runs=len(result["runs"]), within_size_pairs=len(result["pairs"]),
                   skipped_identical_pairs=len(result["skipped_pairs"]),
                   interpretation=result["interpretation"],
                   full_pool="one unique subset; no self-comparison reported",
                   index="shared PCA64; IVF512 below 500k, otherwise IVF1024; nprobe64",
                   calibration="source-specific independent subjects, recalibrated for every draw",
                   provenance=factory.signature)
    (args.output / "STABILITY_RUN.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
