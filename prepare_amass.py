"""Rebuild fixed AMASS clips from your source manifest, labels and licensed SMPL model."""
from __future__ import annotations


# Data preparation
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

from cmat_data import load_amass_clip, SMPLDecoder
from cmat_data import frame_window
from cmat_data import corrupt_rotations
from run_cmat import file_hash, json_save, run_manifest, npz_save


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amass-root', type=Path, required=True)
    parser.add_argument('--smpl', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--indices', type=int, nargs='+')
    args = parser.parse_args(argv)
    model_files = {'model_'+str(i):p for i,p in enumerate(sorted(args.smpl.rglob('*')))
                   if p.is_file()} if args.smpl.is_dir() else {'model':args.smpl}
    if not model_files:
        raise ValueError('No SMPL model files found')
    manifest = run_manifest(args.output, dict(manifest=args.manifest, labels=args.labels,
                                             preparation_script=Path(__file__), **model_files),
                            dict(command='prepare_amass', noise_repeats=2, fps=30,
                                 frames=240, device=args.device))
    with args.manifest.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    with np.load(args.labels, allow_pickle=False) as archive:
        rare, indices = archive['rare'], archive['indices']
    if not np.array_equal(indices, [int(row['index']) for row in rows]):
        raise ValueError('Manifest and fixed label row orders differ')
    if rare.shape != (len(rows), 240) or rare.dtype.kind != 'b':
        raise ValueError('Fixed labels must be a Boolean [clips,240] array')
    if len(set(indices.tolist())) != len(indices):
        raise ValueError('Manifest clip indices must be unique')
    if args.indices is not None and set(args.indices) - set(indices.tolist()):
        raise ValueError('Requested clip index is not in the frozen manifest')
    provenance = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    decoder = None
    for row, mask in zip(rows, rare):
        index = int(row['index'])
        if args.indices is not None and index not in args.indices:
            continue
        destination = args.output / f'{index:04d}.npz'
        source = (args.amass_root / row['source_file']).resolve()
        if args.amass_root.resolve() not in source.parents:
            raise ValueError('Source file must stay within AMASS root')
        window = frame_window(int(row['anchor_source_frame']),
                              float(row['source_fps']), int(row['source_frames']))
        if window is None:
            raise ValueError(f'Selected clip {index} no longer fits its source sequence')
        frames, left = window
        if (frames[0] != int(row['source_start']) or frames[-1] != int(row['source_last'])
                or left != int(row['anchor_position'])):
            raise ValueError(f'Selected clip {index} frame window differs from its frozen manifest')
        if file_hash(source) != row['source_sha256']:
            raise ValueError(f'Source hash changed for clip {index}')
        receipt = destination.with_suffix('.json')
        if destination.exists():
            if receipt.exists():
                saved = json.loads(receipt.read_text(encoding='utf-8'))
                if saved != dict(index=index, provenance=provenance, output_sha256=file_hash(destination)):
                    raise ValueError(f'Cached clip {index} or its provenance changed')
            else:
                # Recover an interruption after atomic NPZ creation but before
                # its receipt; never accept an unrelated preexisting file.
                with np.load(destination, allow_pickle=False) as saved:
                    valid = (str(saved['preparation_provenance']) == provenance
                             and str(saved['source_sha256']) == row['source_sha256']
                             and str(saved['subject']) == row['subject']
                             and saved['observed'].shape == (2, 240, 22, 3)
                             and saved['target'].shape == (240, 22, 3)
                             and np.array_equal(saved['source_frame_ids'], frames)
                             and np.array_equal(saved['rare'], mask))
                    if not valid:
                        raise ValueError(f'Cached clip {index} has no matching provenance')
                json_save(receipt, dict(index=index, provenance=provenance, output_sha256=file_hash(destination)))
            continue
        pose, shape = load_amass_clip(source, frames)
        if decoder is None:
            decoder = SMPLDecoder(args.smpl, device=args.device)
        target = decoder(pose, shape)
        observed = []
        for replicate in range(2):
            noisy, _ = corrupt_rotations(pose, row['subject'], row['source_file'],
                                         int(row['source_start']), replicate)
            observed.append(decoder(noisy, shape))
        observed = np.stack(observed)
        if (target.shape != (240, 22, 3) or observed.shape != (2, 240, 22, 3)
                or not np.isfinite(target).all() or not np.isfinite(observed).all()):
            raise ValueError(f'Decoder returned invalid coordinates for clip {index}')
        npz_save(destination, observed=observed, target=target,
                 rare=mask, frame_ids=np.arange(240), source_frame_ids=frames,
                 subject=row['subject'], units='meters', source_sha256=row['source_sha256'],
                 preparation_provenance=provenance)
        json_save(receipt, dict(index=index, provenance=provenance, output_sha256=file_hash(destination)))
        print(f'Prepared {index:04d}', flush=True)


if __name__ == '__main__':
    main()
