"""Run pose correction, evaluation, rarity classification or rare-pose generation."""
from __future__ import annotations


# Atomic inputs and outputs
from pathlib import Path
import hashlib
import json
import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def package_hash():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def json_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    if len(text.splitlines()) > 500:
        raise ValueError('Metadata exceeds 500 lines; store arrays as NPZ')
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text + '\n', encoding='utf-8')
    temporary.replace(path)


def npz_save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(np.asarray(value).dtype.kind == 'O' for value in arrays.values()):
        raise ValueError('Object/pickled arrays are not supported in standard CMAT files')
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def save_diagnostics(path, value):
    """Keep readable scalars in JSON and long/ragged histories in a linked NPZ."""
    path = Path(path)
    binary = path.with_suffix('.npz')
    arrays = {}

    def pack(item, keys=()):
        if isinstance(item, dict):
            return {key: pack(child, keys + (str(key),)) for key, child in item.items()}
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, (list, tuple, np.ndarray)):
            name = '.'.join(keys) or 'values'
            try:
                array = np.asarray(item)
            except ValueError:
                array = None
            if array is not None and array.ndim == 0:
                return array.item()
            if array is not None and not array.dtype.hasobject and array.size > 32:
                arrays[name] = array
                return dict(npz=binary.name, key=name, shape=list(array.shape))
            if len(item) > 32 and all(isinstance(row, (list, tuple, np.ndarray)) for row in item):
                rows = [np.asarray(row) for row in item]
                if all(row.ndim == 1 and not row.dtype.hasobject for row in rows):
                    nonempty = [row for row in rows if len(row)]
                    dtype = np.result_type(*[row.dtype for row in nonempty]) if nonempty else np.int64
                    arrays[name + '.values'] = np.concatenate(rows).astype(dtype)
                    arrays[name + '.offsets'] = np.r_[0, np.cumsum([len(row) for row in rows])]
                    return dict(npz=binary.name, format='ragged_rows',
                                values=name + '.values', offsets=name + '.offsets')
            return [pack(child, keys + (str(index),)) for index, child in enumerate(item)]
        return item

    compact = pack(value)
    if arrays:
        npz_save(binary, **arrays)
    json_save(path, compact)
    return compact


def run_manifest(directory, inputs, settings):
    """Refuse to resume with changed data, code, or settings."""
    directory = Path(directory)
    manifest = dict(inputs={name:file_hash(path) for name,path in inputs.items()},
                    settings=settings, code_sha256=package_hash())
    # JSON changes tuples to lists. Canonicalize before comparing a resumed
    # dataclass config to its saved representation, not only when writing it.
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    path = directory/'manifest.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved != manifest:
            raise ValueError('Run inputs/settings/code changed; use a new output directory')
    else:
        json_save(path, manifest)
    return manifest


def sequence(path, key='observed', replicate=0):
    if not isinstance(replicate, (int, np.integer)) or replicate < 0:
        raise ValueError('replicate must be a nonnegative integer')
    with np.load(path, allow_pickle=False) as archive:
        value = archive[key]
        if value.ndim == 4:
            if replicate >= len(value):
                raise ValueError('Requested noise replicate is not present')
            value = value[replicate]
        elif replicate != 0:
            raise ValueError('A single trajectory has only replicate 0')
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError('Trajectory must be [T,J,3] or [R,T,J,3]')
        if len(value) == 0 or value.shape[1] == 0 or not np.isfinite(value).all():
            raise ValueError('Trajectory must be nonempty and finite')
        if 'units' in archive and str(archive['units']) != 'meters':
            raise ValueError('Standard files must store coordinates in meters')
        metadata = {name:archive[name] for name in ('target','frame_ids','rare') if name in archive}
        if 'target' in metadata and metadata['target'].shape != value.shape:
            raise ValueError('Target and observed trajectory shapes differ')
    return value, metadata


# Reference and calibration input
import hashlib
import json
from pathlib import Path
import numpy as np

from cmat_rarity import FaissCandidates, RarityModel, SequenceRetriever


def add_parsers(subparsers):
    """Register both geometry commands; each sets ``args.func(args)``."""
    add_classify_parser(subparsers)
    add_generate_parser(subparsers)


def add_reference_arguments(parser):
    parser.add_argument("--reference", type=Path, required=True,
                        help="NPZ: quaternions, subject_ids, sequence_ids, sources")
    parser.add_argument("--calibration", type=Path, required=True,
                        help="Independent-subject NPZ with the same fields")
    parser.add_argument("--retrieval", choices=("exhaustive", "faiss"), default="exhaustive")
    parser.add_argument("--faiss-index", type=Path)
    parser.add_argument("--faiss-pca", type=Path)
    parser.add_argument("--faiss-rows", type=Path)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)


def load_pose_table(path, require_sequences=True):
    """No implicit rotation conversion, identity renaming or pickle loading."""
    required = ["quaternions", "subject_ids", "sources"]
    if require_sequences:
        required.append("sequence_ids")
    with np.load(path, allow_pickle=False) as archive:
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"Missing NPZ fields: {sorted(missing)}")
        values = {key: archive[key] for key in required}
    poses = values["quaternions"]
    if (poses.ndim != 3 or poses.shape[1:] != (21, 4) or not len(poses)
            or poses.dtype.kind != "f" or not np.isfinite(poses).all()):
        raise ValueError("quaternions must be a finite nonempty floating array [N,21,4]")
    norms = np.linalg.norm(poses, axis=-1)
    if np.max(abs(norms - 1)) > 1e-5:
        raise ValueError("quaternions must already be normalized scalar-first (w,x,y,z)")
    for key in required[1:]:
        vector = values[key]
        if vector.shape != (len(poses),) or vector.dtype.kind not in "iuUS":
            raise ValueError(f"{key} must be an aligned integer or Unicode/string vector")
        if vector.dtype.kind == "S":
            values[key] = vector.astype(str)
    values["sources"] = values["sources"].astype(str)
    return values


def validate_source_identity(reference, calibration):
    """A subject belongs to one collection, and calibration cannot share identities."""
    if reference["subject_ids"].dtype.kind != calibration["subject_ids"].dtype.kind:
        numeric = reference["subject_ids"].dtype.kind in "iu" and calibration["subject_ids"].dtype.kind in "iu"
        if not numeric:
            raise ValueError("Reference and calibration must use the same subject-ID type")
    for table in (reference, calibration):
        subjects, sources = table["subject_ids"], table["sources"]
        for subject in np.unique(subjects):
            if len(np.unique(sources[subjects == subject])) != 1:
                raise ValueError("One subject ID is assigned to multiple sources")
    if np.intersect1d(reference["subject_ids"], calibration["subject_ids"]).size:
        raise ValueError("Calibration subjects overlap reference subjects")


def build_model(args, reference=None, calibration=None):
    reference = load_pose_table(args.reference) if reference is None else reference
    calibration = load_pose_table(args.calibration) if calibration is None else calibration
    validate_source_identity(reference, calibration)
    if args.workers < 1 or args.nprobe < 1:
        raise ValueError("workers and nprobe must be positive")
    backend = None
    if args.retrieval == "faiss":
        if not all((args.faiss_index, args.faiss_pca, args.faiss_rows)):
            raise ValueError("FAISS needs --faiss-index, --faiss-pca and --faiss-rows together")
        backend = FaissCandidates.load(args.faiss_index, args.faiss_pca,
                                       args.faiss_rows, args.nprobe)
        if np.any(backend.reference_rows < 0) or np.any(backend.reference_rows >= len(reference["quaternions"])):
            raise ValueError("FAISS row map references rows outside the supplied reference")
    elif any((args.faiss_index, args.faiss_pca, args.faiss_rows)):
        raise ValueError("Choose --retrieval faiss to use the supplied index files")
    elif len(reference["quaternions"]) > 100000:
        raise ValueError("Use explicit FAISS retrieval for more than 100,000 reference poses")
    retriever = SequenceRetriever(reference["quaternions"], reference["sequence_ids"],
                                  reference["subject_ids"], backend, args.workers)
    model = RarityModel(retriever, calibration["quaternions"], calibration["subject_ids"],
                        calibration["sequence_ids"], calibration["sources"])
    return model, reference, calibration


def file_identity(path):
    """Content hashes record provenance without storing absolute filesystem paths."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(name=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest())


def input_manifest(args):
    inputs = {key: file_identity(getattr(args, key)) for key in ("reference", "calibration")}
    if args.retrieval == "faiss":
        for key in ("faiss_index", "faiss_pca", "faiss_rows"):
            path = getattr(args, key)
            if path is None:
                raise ValueError("FAISS requires index, PCA and row map files")
            inputs[key] = file_identity(path)
    package_root = Path(__file__).parent
    code_files = list(package_root.glob("*.py"))
    code_digest = hashlib.sha256()
    for path in sorted(code_files):
        code_digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        code_digest.update(path.read_bytes())
    return dict(schema="cmat-geometry-v1", quaternion_order="wxyz", inputs=inputs,
                retrieval=args.retrieval, nprobe=args.nprobe,
                numpy_version=np.__version__, code_sha256=code_digest.hexdigest())


def write_json(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if len(text.splitlines()) > 500:
        raise ValueError("JSON exceeds 500 lines; put large numerical detail in NPZ")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


# Rarity command
from pathlib import Path
import numpy as np


def add_classify_parser(subparsers):
    parser = subparsers.add_parser("classify", help="Score poses using independent rarity calibration")
    add_reference_arguments(parser)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output NPZ; must not exist")
    parser.add_argument("--subject-mode", choices=("independent", "exclude"),
                        default="independent", help="Held-out queries or reference-subject queries")
    parser.set_defaults(func=classify)


def classify(args):
    if args.output.suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz")
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError("Use a new output path; existing rarity results are retained")
    queries = load_pose_table(args.queries, require_sequences=False)
    reference, calibration = load_pose_table(args.reference), load_pose_table(args.calibration)
    for table in (reference, calibration):
        numeric = queries["subject_ids"].dtype.kind in "iu" and table["subject_ids"].dtype.kind in "iu"
        if not numeric and queries["subject_ids"].dtype.kind != table["subject_ids"].dtype.kind:
            raise ValueError("Queries must use the same global subject-ID type as calibration/reference")
    if args.subject_mode == "independent":
        known = np.union1d(reference["subject_ids"], calibration["subject_ids"])
        if np.intersect1d(queries["subject_ids"], known).size:
            raise ValueError("Independent query subjects overlap reference or calibration subjects")
        excluded = None
    else:
        if not np.isin(queries["subject_ids"], reference["subject_ids"]).all():
            raise ValueError("exclude mode requires reference-subject IDs for every query")
        excluded = queries["subject_ids"]
        for subject in np.unique(excluded):
            expected = np.unique(reference["sources"][reference["subject_ids"] == subject])
            actual = np.unique(queries["sources"][excluded == subject])
            if not np.array_equal(expected, actual):
                raise ValueError("Excluded subject's query source differs from its reference source")
    model, _, _ = build_model(args, reference, calibration)
    if not np.isin(queries["sources"], model.sources).all():
        raise ValueError("A query source has no independent calibration cohort")
    scores = model.classify(queries["quaternions"], queries["sources"], excluded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **scores, subject_ids=queries["subject_ids"], sources=queries["sources"])
    temporary.replace(args.output)
    manifest = input_manifest(args)
    manifest["inputs"]["queries"] = file_identity(args.queries)
    manifest.update(subject_mode=args.subject_mode, query_count=len(queries["quaternions"]),
                    rare_count=int(scores["rare"].sum()), threshold=0.95,
                    calibration_pose_count=len(calibration["quaternions"]),
                    calibration_subject_count=len(np.unique(calibration["subject_ids"])),
                    calibration_source_count=len(np.unique(calibration["sources"])),
                    upper_percentile_rejection=False,
                    interpretation="Empirical source-specific percentile, not a physical-validity probability")
    write_json(summary_path, manifest)
    print(f"Classified {len(queries['quaternions'])} poses; rare={int(scores['rare'].sum())}")
    return manifest


# Generation command
import hashlib
import json
from pathlib import Path
import numpy as np

from cmat_data import SMPLDecoder
from cmat_generation import METHODS, calibrate_pair_thresholds, prepare_anchor_pool, run_blocks


def add_generate_parser(subparsers):
    parser = subparsers.add_parser("generate", help="Run six generators with common final screening")
    add_reference_arguments(parser)
    parser.add_argument("--anchors", type=Path,
                        help="Optional NPZ anchor_rows; otherwise screen a fixed search sample")
    parser.add_argument("--search-size", type=int, default=250000)
    parser.add_argument("--smpl-models", type=Path, required=True,
                        help="Your licensed neutral SMPL file/directory, never bundled")
    parser.add_argument("--device", default=None, help="SMPL torch device, e.g. cpu or cuda:0")
    parser.add_argument("--smpl-batch-size", type=int, default=256)
    parser.add_argument("--attempts", type=int, default=50000)
    parser.add_argument("--block-size", type=int, default=250)
    parser.add_argument("--start", type=int, default=0, help="First attempt-block seed offset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.set_defaults(func=generate)


def model_identity(path):
    """Hash explicit model assets, excluding unrelated directory contents."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(
        item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in (".pkl", ".npz")
    )
    if not files:
        raise ValueError("No SMPL .pkl/.npz assets found at --smpl-models")
    digest, total_bytes = hashlib.sha256(), 0
    for item in files:
        identity = file_identity(item)
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(identity["sha256"].encode("ascii"))
        total_bytes += identity["bytes"]
    return dict(asset_count=len(files), total_bytes=total_bytes, sha256=digest.hexdigest())


def guard_manifest(directory, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Resume manifest mismatch; inputs/code/settings changed. Use a new output directory")
    elif any(directory.iterdir()):
        raise ValueError("Nonempty output directory has no verified run_manifest.json; use a new directory")
    else:
        write_json(path, manifest)


def load_anchor_rows(path, model, reference):
    with np.load(path, allow_pickle=False) as archive:
        if "anchor_rows" not in archive.files:
            raise ValueError("Anchor NPZ needs anchor_rows")
        rows = archive["anchor_rows"]
    if rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows):
        raise ValueError("anchor_rows must be a nonempty integer vector")
    if np.any(rows < 0) or np.any(rows >= len(reference["quaternions"])):
        raise ValueError("Anchor row outside the supplied reference")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("anchor_rows must be unique; repeated draws occur only during attempts")
    sources, subjects = reference["sources"][rows], reference["subject_ids"][rows]
    scores = model.classify(reference["quaternions"][rows], sources, subjects)
    if not scores["rare"].all():
        raise ValueError("Supplied anchor pool includes nonrare rows under this matched calibration")
    for subject in np.unique(subjects):
        if np.sum(subjects == subject) < 2:
            raise ValueError("Every anchor subject needs at least two rare anchors for pairing")
    return rows


def summarize(directory, starts, count):
    totals = {method: dict(attempts=0, accepted=0, rare=0, novel=0,
                           local_support=0, geometry=0) for method in METHODS}
    for start in starts:
        path = directory / f"block_{start:05d}.npz"
        with np.load(path, allow_pickle=False) as saved:
            if len(saved["anchor_rows"]) != count:
                raise ValueError("Unexpected block length while summarizing")
            for method, values in totals.items():
                values["attempts"] += count
                for key in ("accepted", "rare", "novel", "geometry"):
                    values[key] += int(saved[f"{method}__{key}"].sum())
                values["local_support"] += int((saved[f"{method}__chart"] <= 0.55).sum())
    for values in totals.values():
        values["yield_fraction"] = values["accepted"] / values["attempts"]
        proportion, count, z = values["yield_fraction"], values["attempts"], 1.959963984540054
        denominator = 1 + z * z / count
        center = (proportion + z * z / (2 * count)) / denominator
        half = z * np.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
        values["wilson_95_lower"], values["wilson_95_upper"] = center - half, center + half
    result = dict(methods=totals,
                  interpretation="Common-screening yield over all attempts, including infeasible fallbacks",
                  reference_scope="Exactly the reference/calibration/assets hashed in run_manifest.json")
    write_json(directory / "summary.json", result)
    return result


def generate(args):
    if (args.attempts < 1 or args.block_size < 1 or args.attempts % args.block_size
            or args.start < 0 or args.start % args.block_size):
        raise ValueError("Positive attempts must be a block-size multiple; start must be a nonnegative block boundary")
    if args.search_size < 1 or args.smpl_batch_size < 1:
        raise ValueError("search-size and smpl-batch-size must be positive")
    manifest = input_manifest(args)
    if args.anchors is not None:
        manifest["inputs"]["anchors"] = file_identity(args.anchors)
    manifest.update(methods=list(METHODS), attempts=args.attempts, block_size=args.block_size,
                    start=args.start, anchor_search_size=args.search_size, anchor_seed=20260915,
                    smpl_assets=model_identity(args.smpl_models), device=args.device or "auto",
                    smpl_batch_size=args.smpl_batch_size)
    guard_manifest(args.output_dir, manifest)
    model, reference, _ = build_model(args)
    if args.anchors is None:
        anchor_path = args.output_dir / "anchor_pool.npz"
        if anchor_path.exists():
            pool_manifest = args.output_dir / "anchor_pool_manifest.json"
            if not pool_manifest.exists() or json.loads(pool_manifest.read_text()) != file_identity(anchor_path):
                raise ValueError("Generated anchor-pool hash is missing or changed; use a fresh output directory")
            rows = load_anchor_rows(anchor_path, model, reference)
        else:
            pool = prepare_anchor_pool(model, reference["sources"], args.search_size)
            rows = pool["anchor_rows"]
            temporary = anchor_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **pool)
            temporary.replace(anchor_path)
            write_json(args.output_dir / "anchor_pool_manifest.json", file_identity(anchor_path))
    else:
        rows = load_anchor_rows(args.anchors, model, reference)
    if len(rows) < 2:
        raise ValueError("No usable same-subject rare-anchor pairs in this reference/calibration")
    decoder = SMPLDecoder(args.smpl_models, device=args.device, batch_size=args.smpl_batch_size)
    thresholds = calibrate_pair_thresholds(model, decoder)
    source_order = sorted(thresholds)
    threshold_path = args.output_dir / "joint_pair_thresholds.npz"
    if not threshold_path.exists():
        np.savez_compressed(threshold_path, sources=np.asarray(source_order),
                            thresholds=np.stack([thresholds[source] for source in source_order]))
    starts = range(args.start, args.start + args.attempts, args.block_size)
    for start, accepted in run_blocks(args.output_dir, model, rows, reference["sources"],
                                     decoder, thresholds, starts, args.block_size):
        print(f"Completed block {start}: {accepted}", flush=True)
    result = summarize(args.output_dir, starts, args.block_size)
    print("Completed generation; all six methods are recorded in summary.json", flush=True)
    return result


# Correction, evaluation and command dispatch
import argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np

from cmat_correction import correct, published_config
from cmat_evaluation import frame_errors, METRIC_NAMES


def evaluate_arrays(prediction, metadata, roots):
    if 'target' not in metadata:
        raise ValueError('Evaluation requires a target trajectory')
    errors = frame_errors(prediction, metadata['target'], roots,
                          metadata.get('frame_ids'))
    mask = np.asarray(metadata.get('rare', np.ones(len(errors), dtype=bool)))
    if mask.shape != (len(errors),) or mask.dtype.kind != 'b':
        raise ValueError('rare must be one Boolean label per frame')
    selected = errors[mask]
    counts = np.isfinite(selected).sum(0)
    sums = np.nansum(selected, axis=0)
    means = np.divide(sums, counts, out=np.full(3, np.nan), where=counts > 0)
    summary = {name: {'mean': None if not np.isfinite(value) else float(value),
                      'count': int(count)}
               for name, value, count in zip(METRIC_NAMES, means, counts)}
    return errors, summary


def run_correct(args):
    config = published_config(args.method, args.dataset)
    folder = Path(args.output)
    run_manifest(folder, {'sequence': args.input},
                 dict(command='correct', config=asdict(config), seed=args.seed,
                      replicate=args.replicate, evaluate=args.evaluate))
    observed, metadata = sequence(args.input, replicate=args.replicate)
    result = correct(observed, config, seed=args.seed)
    arrays = dict(prediction=result.positions.astype(np.float32), units='meters')
    if args.evaluate:
        errors, summary = evaluate_arrays(arrays['prediction'], metadata,
                                          config.root_indices)
        arrays['errors'] = errors
        json_save(folder / 'metrics.json', summary)
    npz_save(folder / 'prediction.npz', **arrays)
    save_diagnostics(folder / 'diagnostics.json', result.diagnostics)
    print(f'Saved {len(observed)} frames to {folder / "prediction.npz"}')


def run_evaluate(args):
    _, metadata = sequence(args.input, replicate=args.replicate)
    with np.load(args.prediction, allow_pickle=False) as archive:
        prediction = archive['prediction']
        if 'units' in archive and str(archive['units']) != 'meters':
            raise ValueError('Predictions must be in meters')
    folder = Path(args.output)
    run_manifest(folder, {'sequence': args.input, 'prediction': args.prediction},
                 dict(command='evaluate', dataset=args.dataset,
                      replicate=args.replicate))
    errors, summary = evaluate_arrays(
        prediction, metadata, published_config(dataset=args.dataset).root_indices)
    npz_save(folder / 'errors.npz', errors=errors, metrics=np.array(METRIC_NAMES))
    json_save(folder / 'metrics.json', summary)
    print(f'Saved metrics to {folder / "metrics.json"}')


def run_export_eft(args):
    from cmat_data import eft_records
    folder = Path(args.output)
    run_manifest(folder, {'detected': args.detected, 'target': args.target},
                 dict(command='export-eft', joint_layout='paper_14', units='meters'))
    count = 0
    for record in eft_records(
            args.detected, args.target, trust_pickle=args.trust_pickle):
        index = record.pop('index')
        npz_save(folder / f'{index:04d}.npz', **record, units='meters')
        count += 1
    print(f'Exported {count} EFT tracks')


def run_smoothnet(args):
    from cmat_smoothnet import SmoothNet32
    model = SmoothNet32(args.repository, args.checkpoint, device=args.device)
    folder = Path(args.output)
    run_manifest(folder, {'sequence': args.input, 'checkpoint': args.checkpoint,
                         'external_model': Path(args.repository) / 'lib/models/smoothnet.py'},
                 dict(command='smoothnet', replicate=args.replicate,
                      device=args.device, checkpoint=model.metadata))
    observed, _ = sequence(args.input, replicate=args.replicate)
    npz_save(folder / 'prediction.npz', prediction=model.predict(observed), units='meters')
    print(f'Saved SmoothNet predictions to {folder}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='CMAT paper inference and evaluation')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('correct', help='Correct one full joint trajectory')
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--method', choices=('cmat', 'yy', 'temporal'), default='cmat')
    p.add_argument('--dataset', choices=('amass', 'eft'), default='amass')
    p.add_argument('--replicate', type=int, default=0)
    p.add_argument('--seed', type=int, required=True, help='Same seed for both methods')
    p.add_argument('--evaluate', action='store_true')
    p.set_defaults(func=run_correct)
    p = sub.add_parser('evaluate', help='Compute metrics before selecting rare centers')
    p.add_argument('input', type=Path)
    p.add_argument('prediction', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--dataset', choices=('amass', 'eft'), default='amass')
    p.add_argument('--replicate', type=int, default=0)
    p.set_defaults(func=run_evaluate)
    p = sub.add_parser('export-eft', help='Convert official trusted EFT object archives')
    p.add_argument('detected', type=Path)
    p.add_argument('target', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--trust-pickle', action='store_true',
                   help='Only for official archives you independently trust')
    p.set_defaults(func=run_export_eft)
    p = sub.add_parser('smoothnet', help='Inference with the external paper checkpoint')
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--repository', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--replicate', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.set_defaults(func=run_smoothnet)
    add_parsers(sub)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
