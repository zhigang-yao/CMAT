"""Analyze correction totals or complete rare-pose generation blocks."""
from __future__ import annotations


# Correction statistics
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from cmat_evaluation import METRIC_NAMES, benjamini_hochberg, paired_comparison, subject_bootstrap, subject_totals

MAIN_METHODS = ("input", "time_only", "full", "yao", "smoothnet")
VARIANTS = (
    "time_only", "no_guide", "no_cylinder", "uniform_attention", "no_reliability",
    "no_balance", "no_gls", "no_balance_gls", "normal_only", "full_displacement_only",
    "no_cap", "no_null", "no_risk_margin", "equal_head_mix", "no_final_temporal",
    "no_final_robust_data", "no_overlap_normalization", "temporal_neighbors", "pose_only_neighbors",
)
LABELS = {
    "input": "Perturbed input", "time_only": "Temporal correction alone",
    "full": "Self-Action CMAT", "yao": "YY-based", "smoothnet": "SmoothNet",
    "no_guide": "No preliminary guide", "no_cylinder": "No cylinder boundary",
    "uniform_attention": "Reliability-only weights", "no_reliability": "No reliability factor",
    "no_balance": "No balance head", "no_gls": "No GLS head", "no_balance_gls": "No balance or GLS heads",
    "normal_only": "Rank-three normal projection only", "full_displacement_only": "Full-center constraint only",
    "no_cap": "No displacement coefficient", "no_null": "No null option",
    "no_risk_margin": "No score margin", "equal_head_mix": "Equal candidate mixture",
    "no_final_temporal": "No final temporal penalty", "no_final_robust_data": "Quadratic final observation loss",
    "no_overlap_normalization": "No overlap normalization", "temporal_neighbors": "Temporal neighbors",
    "pose_only_neighbors": "Position-only matching",
}


def csv_shards(folder, stem, rows):
    """Keep every CSV at most 400 data records; do not truncate results."""
    for start in range(0, len(rows), 400):
        path = folder / f"{stem}_{start // 400 + 1:03d}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows[start:start + 400])


def reproduce(archive_path, output, n_resamples=10000, seed=20260926):
    """Return compact validation metadata after writing three small tables."""
    archive_path, output = Path(archive_path), Path(output)
    with np.load(archive_path, allow_pickle=False) as z:
        sums, counts, subjects = z["sums"], z["counts"], z["subjects"]
        methods, metrics, keep = z["methods"].tolist(), z["metrics"].tolist(), z["keep"]
    if metrics != list(METRIC_NAMES):
        raise ValueError("metric labels/order must match the paper's three metrics")
    missing = set(MAIN_METHODS + VARIANTS) - set(methods)
    if missing or len(set(methods)) != len(methods):
        raise ValueError(f"missing paper methods or repeated method labels: {sorted(missing)}")
    names, numerator, denominator = subject_totals(sums, counts, subjects, keep)
    point, samples = subject_bootstrap(numerator, denominator, n_resamples, seed)
    reference = methods.index("full")
    means, comparisons, components = [], [], []
    for method in MAIN_METHODS:
        m = methods.index(method)
        low, high = np.quantile(samples[:, m], [0.025, 0.975], axis=0)
        for k, metric in enumerate(metrics):
            means.append({"method": method, "label": LABELS[method], "metric": metric, "mean": float(point[m, k]),
                          "ci_low": float(low[k]), "ci_high": float(high[k])})
    def comparison_rows(method):
        result = paired_comparison(point, samples, methods.index(method), reference)
        rows = []
        for k, metric in enumerate(metrics):
            row = {"method": method, "label": LABELS[method], "reference": "full", "metric": metric}
            row.update({key: float(value[k]) for key, value in result.items()})
            rows.append(row)
        return rows
    for method in MAIN_METHODS:
        if method != "full":
            comparisons.extend(comparison_rows(method))
    for method in VARIANTS:
        components.extend(comparison_rows(method))
    if len(components) != 57:
        raise ValueError("the component family must contain all 19 x 3 comparisons")
    adjusted = benjamini_hochberg([r["p_normal_bootstrap_sd"] for r in components])
    for row, q_value in zip(components, adjusted):
        row["q_bh"] = float(q_value)
    output.mkdir(parents=True, exist_ok=True)
    csv_shards(output, "main_means", means)
    csv_shards(output, "paired_comparisons", comparisons)
    csv_shards(output, "component_ablations", components)
    summary = {
        "input_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "clips": int(keep.sum()), "subjects": len(names),
        "rare_observations": int(denominator[:, reference, 0].sum()),
        "rare_acceleration_centers": int(denominator[:, reference, 2].sum()),
        "bootstrap_replicates": n_resamples, "bootstrap_seed": seed,
        "main_rows": len(means), "paired_rows": len(comparisons), "component_rows": len(components),
        "estimand": "pooled valid frame observations; shared subject bootstrap",
        "scope": "fixed retrospective mask; no threshold search; inference not rerun",
        "positive_difference": "comparator or variant minus full CMAT",
    }
    (output / "REPRODUCTION.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# Generation statistics
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np

METHODS = (
    "rare_replay", "axis_gaussian", "part_mix", "posetrans_3d",
    "poseaug_3d", "rarity_preserving_cmat",
)
GENERATION_LABELS = (
    "Rare replay", "Axis-angle Gaussian", "Part mix", "PoseTrans-inspired 3D",
    "PoseAug-style 3D", "Rarity-preserving CMAT",
)
CRITERIA = ("geometry", "rarity", "non_memorization", "local_support")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    """Keep at most 400 data records per file without truncating the manifest."""
    paths = []
    for start in range(0, len(rows), 400):
        chunk = rows[start:start + 400]
        target = path if len(rows) <= 400 else path.with_name(f"{path.stem}_{start // 400 + 1:03d}.csv")
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(chunk[0]))
            writer.writeheader()
            writer.writerows(chunk)
        paths.append(target.name)
    return paths


def wilson(accepted, attempted):
    z = 1.959963984540054
    proportion = accepted / attempted
    denominator = 1 + z * z / attempted
    center = (proportion + z * z / (2 * attempted)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / attempted
                       + z * z / (4 * attempted * attempted)) / denominator
    return max(0, min(proportion, center - half)), min(1, max(proportion, center + half))


def block_files(directory):
    files = []
    for path in directory.glob("block_*.npz"):
        match = re.fullmatch(r"block_(\d+)\.npz", path.name)
        if match:
            files.append((int(match.group(1)), path))
    files.sort()
    if not files or len({start for start, _ in files}) != len(files):
        raise ValueError("Need nonempty uniquely numbered complete block files")
    return files


def summarize(directory, output, expected_attempts=None):
    files = block_files(directory)
    counts = {method: dict(attempts=0, accepted=0, individual=np.zeros(4, dtype=np.int64),
                            cumulative=np.zeros(4, dtype=np.int64)) for method in METHODS}
    manifest, expected_start = [], files[0][0]
    for start, path in files:
        if start != expected_start:
            raise ValueError(f"Missing or overlapping attempt block before {path.name}")
        with np.load(path, allow_pickle=False) as saved:
            attempted = len(saved["anchor_rows"])
            if attempted < 1:
                raise ValueError("Empty generation block")
            expected_start += attempted
            for method, record in counts.items():
                prefix = method + "__"
                arrays = [saved[prefix + name] for name in ("geometry", "rare", "novel")]
                chart = saved[prefix + "chart"]
                accepted = saved[prefix + "accepted"]
                rho = saved[prefix + "rho"]
                if any(array.shape != (attempted,) for array in arrays + [chart, accepted, rho]):
                    raise ValueError(f"Unaligned block arrays for {method} in {path.name}")
                if not np.array_equal(arrays[1], rho >= 0.95):
                    raise ValueError(f"Stored rarity flags differ from rho>=0.95 for {method}")
                masks = [np.asarray(array, dtype=bool) for array in arrays] + [chart <= 0.55]
                final = np.logical_and.reduce(masks)
                if not np.array_equal(final, accepted):
                    raise ValueError(f"Stored acceptance violates common final screening for {method}")
                record["attempts"] += attempted
                record["accepted"] += int(accepted.sum())
                record["individual"] += np.asarray([mask.sum() for mask in masks])
                retained = np.ones(attempted, dtype=bool)
                for index, mask in enumerate(masks):
                    retained &= mask
                    record["cumulative"][index] += int(retained.sum())
        manifest.append(dict(file=path.name, first_attempt=start, attempts=attempted,
                             bytes=path.stat().st_size, sha256=file_hash(path)))
    attempts = counts[METHODS[0]]["attempts"]
    if expected_attempts is not None and attempts != expected_attempts:
        raise ValueError(f"Expected {expected_attempts} attempts per method; found {attempts}")
    yields, screening = [], []
    for method, label in zip(METHODS, GENERATION_LABELS):
        record = counts[method]
        low, high = wilson(record["accepted"], attempts)
        yields.append(dict(method=method, label=label, attempts=attempts,
                           accepted=record["accepted"], yield_fraction=record["accepted"] / attempts,
                           yield_percent=100 * record["accepted"] / attempts,
                           wilson_95_low=low, wilson_95_high=high))
        for mode in ("individual", "cumulative"):
            for index, criterion in enumerate(CRITERIA):
                passed = int(record[mode][index])
                screening.append(dict(method=method, mode=mode, order=index + 1, criterion=criterion,
                                      attempts=attempts, passed=passed, fraction=passed / attempts,
                                      percent=100 * passed / attempts))
        screening.append(dict(method=method, mode="joint", order=4, criterion="all_criteria",
                              attempts=attempts, passed=record["accepted"],
                              fraction=record["accepted"] / attempts,
                              percent=100 * record["accepted"] / attempts))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "generation_yields.csv", yields)
    write_csv(output / "generation_screening.csv", screening)
    manifests = write_csv(output / "generation_block_manifest.csv", manifest)
    digest = hashlib.sha256()
    for row in manifest:
        digest.update((row["file"] + ":" + row["sha256"] + "\n").encode("ascii"))
    report = dict(
        computation="Aggregate recomputation from existing saved NPZ blocks; no pose inference in this script",
        attempts_per_method=attempts, methods=list(METHODS), block_count=len(files),
        first_attempt=files[0][0], after_last_attempt=expected_start,
        stored_rarity_verified=True, common_acceptance_intersection_verified=True,
        cumulative_criterion_order=list(CRITERIA), source_block_digest=digest.hexdigest(),
        source_block_manifests=manifests,
        intervals="Wilson 95%; conditional Monte Carlo yield under fixed reference and screening",
        screening_note="individual is each criterion alone; cumulative intersects criteria in the stated order",
    )
    (output / "generation_recomputation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return yields, screening, report


def plot(yields, screening, path):
    """Optional scientific summary; entirely driven by recomputed table rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    proportions = np.asarray([row["yield_percent"] for row in yields])
    low = np.asarray([100 * row["wilson_95_low"] for row in yields])
    high = np.asarray([100 * row["wilson_95_high"] for row in yields])
    values = np.asarray([[next(row["percent"] for row in screening
                               if row["method"] == method and row["mode"] == "individual"
                               and row["criterion"] == criterion)
                          for criterion in CRITERIA] for method in METHODS])
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    locations = np.arange(len(METHODS))
    axes[0].barh(locations, proportions, color=["#7b9db8"] * 5 + ["#ca683a"])
    axes[0].errorbar(proportions, locations, xerr=[proportions - low, high - proportions],
                     fmt="none", ecolor="black", capsize=3)
    axes[0].set_yticks(locations, GENERATION_LABELS)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Accepted per attempt (%)")
    axes[0].set_title("(a) Yield and Wilson 95% intervals")
    heat = axes[1].imshow(values, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    axes[1].set_yticks(locations, GENERATION_LABELS)
    axes[1].set_xticks(np.arange(4), ["Joint checks", "Rarity", "Separation", "Local support"], rotation=25)
    axes[1].set_title("(b) Each criterion separately (%)")
    for row in range(len(METHODS)):
        for column in range(4):
            axes[1].text(column, row, f"{values[row, column]:.2f}", ha="center", va="center",
                         color="white" if values[row, column] > 55 else "black", fontsize=8)
    figure.colorbar(heat, ax=axes[1], shrink=0.7)
    figure.savefig(path)
    plt.close(figure)


# Commands
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("correction", help="Means, paired intervals and all 57 component tests")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260926)
    p = sub.add_parser("generation", help="Yields and screening from generation block NPZs")
    p.add_argument("--blocks", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-attempts", type=int)
    p.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.command == "correction":
        result = reproduce(args.input, args.output, args.bootstrap, args.seed)
    else:
        yields, screening, result = summarize(args.blocks, args.output, args.expected_attempts)
        if args.plot:
            plot(yields, screening, args.output / "generation_summary.pdf")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
