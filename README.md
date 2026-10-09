# CMAT: Recovery and Generation of Rare 3D Human Poses

Code for **Local Manifold Attention for Recovery and Generation of Rare 3D Human Poses**.

CMAT (cylindrical manifold attention transformer) uses local geometric structure to recover noisy human motions and generate rare body configurations. 
The repository provides two methods:

- **Self-Action CMAT** corrects a sequence of 3D joint positions using comparable motion windows from that same sequence.
- **Rarity-preserving CMAT** generates poses near rare reference configurations and checks their rarity, separation from stored poses, and local geometric support.

It also includes independently calibrated rarity classification, correction and generation comparators, and scripts for the paper's experiments.


## Installation

Use **Python 3.10 or later**. Run the scripts from the repository root.

```bash
python -m pip install "numpy>=2.0" "scipy>=1.12"
```

These dependencies are sufficient for correction and numerical evaluation. Install additional packages for the tasks you need:

| Task | Additional requirements |
| --- | --- |
| Large-reference neighbor retrieval | `faiss-cpu` |
| SMPL decoding and generation with joint-level screening | `torch`, `smplx`, and licensed SMPL model files |
| SmoothNet comparison | `torch`, the official SmoothNet source checkout, and the required pretrained checkpoint |
| Result plots | `matplotlib` |

Motion datasets, body models, and third-party checkpoints must be obtained separately under their respective terms. 

## Code layout

| File | Purpose |
| --- | --- |
| `cmat_correction.py` | Self-Action CMAT, YY-based and temporal comparators, candidate selection, and trajectory solver |
| `cmat_rotations.py` | Quaternion operations, rotation distances, and local rotation coordinates |
| `cmat_geometry.py` | Weighted local fitting, bounded cylinders, and generation attention |
| `cmat_rarity.py` | Distinct-sequence retrieval and independent rarity calibration |
| `cmat_generation.py` | Pose proposals, refinement, generation comparators, and acceptance checks |
| `cmat_data.py` | Rare-first clip selection, perturbations, SMPL decoding, and EFT import |
| `cmat_evaluation.py` | Frame metrics, motion descriptors, aggregation, and statistical inference |
| `cmat_smoothnet.py` | Inference adapter for the external SmoothNet model |
| `run_cmat.py` | Single-clip correction, evaluation, classification, generation, and data-import commands |
| `run_experiments.py` | Batch correction and component experiments |
| `analyze_results.py` | Correction tables and generation acceptance summaries |
| `reference_stability.py` | Repeated reference-size experiments with recalibration for each draw |
| `prepare_amass.py` | AMASS clip reconstruction from a fixed manifest and rarity labels |

Run `python run_cmat.py --help` or the individual script's `--help` for available options.
