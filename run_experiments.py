"""Batch AMASS/EFT correction, component interventions and paired metric totals."""
from __future__ import annotations

import os
for _thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_thread_variable] = "1"


# Controlled component interventions
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cmat_correction import correct, published_config
from cmat_correction import normalize_observations, recover_prepared
from cmat_correction import VARIANTS, REBUILD
from cmat_correction import prepare


def run_ablation(observed, variant, seed=0):
    """Run one fixed AMASS intervention, without any tuning or truth input."""
    return correct(observed, published_config("cmat", "amass"), seed, intervention=variant)


def iter_ablations(observed, seed=0, variants=None):
    """Yield ``(name, CorrectionResult)`` without keeping all outputs in memory.

    Repeated calls have no shared mutable state. ``variants`` supports resuming
    an interrupted run from its missing files without rerunning completed ones.
    The default includes full CMAT and all 19 published interventions.
    """
    names = tuple(VARIANTS) if variants is None else tuple(variants)
    if len(set(names)) != len(names) or any(name not in VARIANTS for name in names):
        raise ValueError("variants must contain unique names from VARIANTS")
    config = published_config("cmat", "amass")
    raw, y, root, scale = normalize_observations(observed, config)
    if len(raw) < 16:
        for name in names:
            yield name, correct(raw, config, seed, intervention=name)
        return
    need_full = any(name not in REBUILD and name != "time_only" for name in names)
    full_state, full_trials = prepare(y, config, seed) if need_full else (None, None)
    for name in names:
        if name in REBUILD:
            state, trials = prepare(y, config, seed, intervention=name)
        else:
            state, trials = full_state, full_trials
        result = recover_prepared(y, root, scale, config, seed, state, trials, name)
        yield name, result


def run_all(observed, seed=0, variants=None):
    """Return a mapping for a small sequence; use iter_ablations for batching."""
    return dict(iter_ablations(observed, seed, variants))


# Case inference and aggregation
import hashlib
import json
from pathlib import Path

import numpy as np

from cmat_correction import correct, published_config
from cmat_evaluation import METRIC_NAMES, frame_errors
from run_cmat import file_hash, json_save, npz_save, package_hash


_WORKER = {}


def read_cohort(path):
    if path is None:
        return {}
    with np.load(path, allow_pickle=False) as data:
        indices = data["indices"]
        subjects = data["subjects"]
        keep = data["keep"]
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("Cohort indices must be a one-dimensional integer array")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Duplicate cohort indices")
        if subjects.shape != indices.shape or keep.shape != indices.shape:
            raise ValueError("Cohort indices, subjects and keep must have equal shapes")
        if keep.dtype.kind != "b":
            raise ValueError("Cohort keep must be Boolean")
        rare = data["rare"] if "rare" in data else None
        if rare is not None and (rare.ndim != 2 or len(rare) != len(indices)):
            raise ValueError("Cohort rare must have shape [clips, frames]")
        return {
            int(index): dict(
                subject=str(subjects[row]),
                keep=bool(keep[row]),
                rare=None if rare is None else rare[row].copy(),
            )
            for row, index in enumerate(indices)
        }


def clip_record(path, dataset, cohort):
    """Read the explicit standard schema; never infer a rarity label."""
    with np.load(path, allow_pickle=False) as data:
        if "index" in data:
            value = data["index"]
            if value.ndim != 0 or value.dtype.kind not in "iu":
                raise ValueError(f"{path.name}: index must be a scalar integer")
            index = int(value)
        else:
            try:
                index = int(path.stem)
            except ValueError as error:
                raise ValueError(f"{path.name}: provide a scalar integer index") from error
        if index < 0:
            raise ValueError("Clip indices must be nonnegative")
        if cohort and index not in cohort:
            raise ValueError(f"Clip {index} is absent from the supplied fixed cohort")
        row = cohort.get(index, {})
        if "units" not in data or str(data["units"]) != "meters":
            raise ValueError(f"{path.name}: units='meters' is required")
        observed, target = data["observed"], data["target"]
        repeats = 2 if dataset == "amass" else 1
        valid_shape = (
            observed.ndim == 4 and observed.shape[0] == 2
            if dataset == "amass"
            else observed.ndim == 3 and observed.shape[1] == 14
        )
        sample_shape = observed.shape[1:] if dataset == "amass" else observed.shape
        if not valid_shape or len(sample_shape) != 3 or sample_shape[-1] != 3:
            raise ValueError("AMASS needs [2,T,J,3]; EFT needs [T,14,3]")
        if target.shape != sample_shape or min(sample_shape) < 1:
            raise ValueError(f"{path.name}: target must be nonempty [T,J,3]")
        if not np.isfinite(observed).all() or not np.isfinite(target).all():
            raise ValueError(f"{path.name}: coordinates must be finite")
        frames = len(target)
        frame_ids = data["frame_ids"] if "frame_ids" in data else np.arange(frames)
        if dataset == "eft" and "frame_ids" not in data:
            raise ValueError("EFT requires original frame_ids to exclude acceleration gaps")
        if frame_ids.shape != (frames,) or frame_ids.dtype.kind not in "iu":
            raise ValueError("frame_ids must be an integer [T] array")
        if np.any(np.diff(frame_ids) <= 0):
            raise ValueError("frame_ids must be strictly increasing")
        rare = data["rare"] if "rare" in data else row.get("rare")
        if dataset == "amass":
            if rare is None or rare.shape != (frames,) or rare.dtype.kind != "b":
                raise ValueError("AMASS requires a Boolean clean-pose rare[T] mask")
            if row.get("rare") is not None and not np.array_equal(rare, row["rare"]):
                raise ValueError(f"Clip {index}: rare labels differ from fixed cohort")
        else:
            # The final paper's EFT comparison evaluates all supplied frames.
            rare = np.ones(frames, dtype=bool)
        subject = str(data["subject"]) if "subject" in data else row.get("subject")
        if "subject" in data and data["subject"].ndim != 0:
            raise ValueError("subject must be a scalar string")
        if row.get("subject") is not None and subject != row["subject"]:
            raise ValueError(f"Clip {index}: subject differs from fixed cohort")
        if not subject and dataset == "amass":
            raise ValueError("AMASS requires subject identity or --cohort")
        subject = subject or f"track:{index}"
        source = str(data["source"]) if "source" in data else subject.split("/")[0]
    return dict(
        index=index,
        path=str(path.resolve()),
        filename=path.name,
        input_sha256=file_hash(path),
        subject=subject,
        source=source,
        keep=row.get("keep", True),
        rare=rare.copy(),
        frames=frames,
        repeats=repeats,
    )


def method_names(components, smoothnet):
    names = ["input", "full", "yao", "time_only"]
    if components:
        names.extend(name for name in VARIANTS if name not in ("full", "time_only"))
    if smoothnet:
        names.append("smoothnet")
    return names


def initialize_worker(settings, expected_code):
    if package_hash() != expected_code:
        raise RuntimeError("Package changed after run manifest creation")
    _WORKER.clear()
    _WORKER.update(settings=settings, model=None)


def prediction_iterator(observed, dataset, seed, components):
    yield "input", observed
    if components:
        for name, result in iter_ablations(observed, seed):
            yield name, result.positions
    else:
        yield "full", correct(observed, published_config("cmat", dataset), seed).positions
        yield "time_only", correct(
            observed, published_config("temporal", dataset), seed
        ).positions
    yield "yao", correct(observed, published_config("yy", dataset), seed).positions
    settings = _WORKER["settings"]
    if settings["smoothnet_repository"]:
        if _WORKER["model"] is None:
            from cmat_smoothnet import SmoothNet32

            _WORKER["model"] = SmoothNet32(
                settings["smoothnet_repository"],
                settings["smoothnet_checkpoint"],
                device=settings["device"],
            )
        # SmoothNet receives raw XYZ, as in the frozen comparison, not a guide.
        yield "smoothnet", _WORKER["model"].predict(observed)


def run_case(task):
    record, replicate, output, signature = task
    settings = _WORKER["settings"]
    dataset, methods = settings["dataset"], settings["methods"]
    index = record["index"]
    label = "amass" if dataset == "amass" else "aist"
    seed = int(hashlib.sha256(f"r4|{label}|{index}".encode()).hexdigest()[:8], 16)
    destination = Path(output) / "cases" / f"{index:06d}_r{replicate}.npz"
    receipt = destination.with_suffix(".json")
    provenance = dict(
        run_signature=signature,
        input_sha256=record["input_sha256"],
        index=index,
        replicate=replicate,
        seed=seed,
    )
    if destination.exists() and receipt.exists():
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        if saved != dict(provenance, output_sha256=file_hash(destination)):
            raise ValueError(f"Case {index}/{replicate}: cached data or receipt changed")
        return dict(index=index, replicate=replicate, resumed=True)
    if file_hash(record["path"]) != record["input_sha256"]:
        raise ValueError(f"Clip {index}: input changed after inventory")
    with np.load(record["path"], allow_pickle=False) as data:
        observed = data["observed"]
        if dataset == "amass":
            observed = observed[replicate]
        target = data["target"]
        frame_ids = data["frame_ids"] if "frame_ids" in data else np.arange(len(target))
    positions, errors = {}, {}
    roots = published_config("cmat", dataset).root_indices
    for name, result in prediction_iterator(observed, dataset, seed, settings["components"]):
        positions[name] = np.asarray(result, dtype=np.float32)
        errors[name] = frame_errors(positions[name], target, roots, frame_ids)
    predictions = np.stack([positions[name] for name in methods])
    all_errors = np.stack([errors[name] for name in methods])
    finite = np.isfinite(all_errors)
    if not np.all(finite == finite[0:1]):
        raise ValueError("Methods have unequal valid metric masks; cannot compare them")
    selected = finite & record["rare"][None, :, None]
    sums = np.where(selected, all_errors, 0.0).sum(axis=1)
    counts = selected.sum(axis=1)
    npz_save(
        destination,
        prediction=predictions,
        target=target,
        errors=all_errors,
        frame_ids=frame_ids,
        rare=record["rare"],
        methods=np.asarray(methods),
        metrics=np.asarray(METRIC_NAMES),
        sums=sums,
        counts=counts,
        units="meters",
        run_signature=signature,
    )
    json_save(receipt, dict(provenance, output_sha256=file_hash(destination)))
    return dict(index=index, replicate=replicate, resumed=False)


def aggregate(records, output, methods):
    """Pool sums/counts across repeats, without averaging sequence means."""
    clip_sums, clip_counts = [], []
    for record in records:
        sums = np.zeros((len(methods), len(METRIC_NAMES)), dtype=np.float64)
        counts = np.zeros_like(sums)
        for replicate in range(record["repeats"]):
            path = Path(output) / "cases" / f"{record['index']:06d}_r{replicate}.npz"
            with np.load(path, allow_pickle=False) as data:
                if list(data["methods"]) != methods:
                    raise ValueError("Stored method labels do not match run")
                sums += data["sums"]
                counts += data["counts"]
        clip_sums.append(sums)
        clip_counts.append(counts)
    arrays = dict(
        indices=np.asarray([row["index"] for row in records]),
        subjects=np.asarray([row["subject"] for row in records]),
        sources=np.asarray([row["source"] for row in records]),
        keep=np.asarray([row["keep"] for row in records]),
        methods=np.asarray(methods),
        metrics=np.asarray(METRIC_NAMES),
        sums=np.stack(clip_sums),
        counts=np.stack(clip_counts),
        rare_values=np.concatenate([row["rare"] for row in records]),
        rare_offsets=np.r_[0, np.cumsum([row["frames"] for row in records])],
    )
    if len({row["frames"] for row in records}) == 1:
        arrays["rare"] = np.stack([row["rare"] for row in records])
    npz_save(Path(output) / "clip_totals.npz", **arrays)
    return arrays


# Batch runner
import os

# One numerical thread per worker avoids multiplying BLAS threads by workers.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from cmat_smoothnet import PAPER_CHECKPOINT_SHA256
from cmat_correction import published_config
from run_cmat import file_hash, json_save, npz_save, run_manifest


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Directory containing one standard NPZ per clip")
    parser.add_argument("output", type=Path, help="Separate output directory; may resume")
    parser.add_argument("--dataset", choices=("amass", "eft"), required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cohort", type=Path, help="Fixed indices, subjects, keep, optional rare")
    parser.add_argument("--components", action="store_true", help="AMASS: full plus 19 interventions")
    parser.add_argument("--smoothnet-repository", type=Path)
    parser.add_argument("--smoothnet-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu", help="SmoothNet device; CUDA needs one worker")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.components and args.dataset != "amass":
        parser.error("The 19 frozen component interventions are defined for AMASS")
    if bool(args.smoothnet_repository) != bool(args.smoothnet_checkpoint):
        parser.error("Supply both SmoothNet repository and checkpoint")
    if args.smoothnet_repository and args.device != "cpu" and args.workers != 1:
        parser.error("Non-CPU SmoothNet inference requires --workers 1")
    args.input, args.output = args.input.resolve(), args.output.resolve()
    if args.input == args.output or args.output in args.input.parents:
        parser.error("Output must be separate from, and not contain, the input directory")
    if not args.input.is_dir():
        parser.error("Input must be a directory of standard NPZ clips")
    return args


def prepare_run(args, records, cohort):
    methods = method_names(args.components, bool(args.smoothnet_repository))
    inventory = [
        (row["index"], row["filename"], row["input_sha256"]) for row in records
    ]
    inventory_hash = hashlib.sha256(json.dumps(inventory).encode()).hexdigest()
    settings = dict(
        dataset=args.dataset,
        methods=methods,
        components=args.components,
        device=args.device,
        seed_template="r4|amass|index" if args.dataset == "amass" else "r4|aist|index",
        repeats_per_clip=2 if args.dataset == "amass" else 1,
        metric_selection="clean rare centers" if args.dataset == "amass" else "all frames",
        keep_policy="supplied fixed cohort" if cohort else "all supplied clips",
        inventory_sha256=inventory_hash,
        configs={
            method: asdict(published_config(method, args.dataset))
            for method in ("cmat", "yy", "temporal")
        },
    )
    inputs = {
        "runner": Path(__file__),
        
    }
    if args.cohort:
        inputs["cohort"] = args.cohort
    if args.smoothnet_repository:
        inputs["smoothnet_source"] = args.smoothnet_repository / "lib/models/smoothnet.py"
        inputs["smoothnet_checkpoint"] = args.smoothnet_checkpoint
        if file_hash(args.smoothnet_checkpoint) != PAPER_CHECKPOINT_SHA256:
            raise ValueError("SmoothNet checkpoint does not match the frozen paper checkpoint")
    if args.output.exists() and not (args.output / "manifest.json").exists():
        if any(args.output.iterdir()):
            raise ValueError("Nonempty output has no manifest; choose a new output directory")
    manifest = run_manifest(args.output, inputs, settings)
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    npz_save(
        args.output / "input_inventory.npz",
        indices=np.asarray([row["index"] for row in records]),
        filenames=np.asarray([row["filename"] for row in records]),
        sha256=np.asarray([row["input_sha256"] for row in records]),
    )
    worker_settings = dict(
        dataset=args.dataset,
        methods=methods,
        components=args.components,
        device=args.device,
        smoothnet_repository=(
            str(args.smoothnet_repository.resolve()) if args.smoothnet_repository else None
        ),
        smoothnet_checkpoint=(
            str(args.smoothnet_checkpoint.resolve()) if args.smoothnet_checkpoint else None
        ),
    )
    return methods, manifest, signature, worker_settings


def main(argv=None):
    args = arguments(argv)
    cohort = read_cohort(args.cohort)
    if args.cohort and not cohort:
        raise ValueError("Supplied cohort is empty")
    records = sorted(
        (clip_record(path, args.dataset, cohort) for path in args.input.glob("*.npz")),
        key=lambda row: row["index"],
    )
    if not records:
        raise ValueError("No standard NPZ clips found")
    if len({row["index"] for row in records}) != len(records):
        raise ValueError("Duplicate clip indices")
    methods, manifest, signature, worker_settings = prepare_run(args, records, cohort)
    tasks = [
        (row, replicate, str(args.output), signature)
        for row in records
        for replicate in range(row["repeats"])
    ]
    completed, resumed = 0, 0

    def record_completion(result):
        nonlocal completed, resumed
        completed += 1
        resumed += int(result["resumed"])
        json_save(
            args.output / "progress.json",
            dict(run_signature=signature, cases_completed=completed, cases_total=len(tasks)),
        )
        if completed % 10 == 0 or completed == len(tasks):
            print(f"Cases {completed}/{len(tasks)}; reused {resumed}", flush=True)

    if args.workers == 1:
        initialize_worker(worker_settings, manifest["code_sha256"])
        for task in tasks:
            record_completion(run_case(task))
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker,
            initargs=(worker_settings, manifest["code_sha256"]),
        ) as pool:
            futures = [pool.submit(run_case, task) for task in tasks]
            for future in as_completed(futures):
                record_completion(future.result())
    totals = aggregate(records, args.output, methods)
    keep = totals["keep"]
    sums = totals["sums"][keep].sum(axis=0)
    counts = totals["counts"][keep].sum(axis=0)
    means = {
        name: [float(s / n) if n else None for s, n in zip(sums[row], counts[row])]
        for row, name in enumerate(methods)
    }
    summary = dict(
        run_signature=signature,
        clips=len(records),
        kept_clips=int(keep.sum()),
        cases=len(tasks),
        cases_reused=resumed,
        methods=methods,
        metrics=totals["metrics"].tolist(),
        kept_pooled_means=means,
        fixed_cohort_provided=bool(cohort),
        complete_fixed_cohort=bool(cohort) and len(records) == len(cohort),
        all_paper_table_methods_present=args.components and bool(args.smoothnet_repository),
        grouping="subject" if args.dataset == "amass" else "person-track unless subject supplied",
        totals_sha256=file_hash(args.output / "clip_totals.npz"),
        note="Metrics are newly computed; cohort input supplies labels and inclusion only.",
    )
    json_save(args.output / "COMPLETE.json", summary)
    print(f"Saved {len(records)} clip totals and {len(tasks)} case archives", flush=True)


if __name__ == "__main__":
    main()
