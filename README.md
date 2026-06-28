# IRIS: A Real-World Benchmark for Inverse Recovery and Identification of Physical Dynamic Systems from Monocular Video

> **ECCV 2026**

[**📄 Paper (arXiv)**](https://arxiv.org/abs/2603.16432) · [**🌐 Project Page**](https://kurbanintelligencelab.github.io/iris-benchmark.github.io/) · [**🤗 Dataset**](https://huggingface.co/datasets/rasulkhanbayov/IRIS)

This repository contains the **IRIS benchmark** and the code for a two-stage pipeline that **automatically identifies both the governing equation and its physical parameters from video**, without relying on folder structure or manual labelling.

---

## TL;DR

Most prior work assumes you already know which ODE governs a system (e.g. pendulum, free fall). We remove that assumption:

- **Stage 1 — Equation-family selection:** A Vision-Language Model (VLM) or a fine-tuned video classifier watches the video and selects which ODE from a library applies.
- **Stage 2 — Parameter estimation:** An encoder maps video frames into a latent space where a physics block fits the selected ODE and estimates physical parameters (e.g. gravity *g*, pendulum length, friction, decay rate).

We also introduce **IRIS**, a 4K real-world benchmark of **240 videos** across **8 dynamics classes** (single- and multi-body) with independently measured ground-truth parameters.

---

## The IRIS Dataset

IRIS is hosted on Hugging Face: **https://huggingface.co/datasets/rasulkhanbayov/IRIS**

- **240 videos** = 8 classes × 3 settings × 10 takes, at 3840×2160 / 60 fps
- Ground-truth physical parameters per `(class, setting)` in `parameters.json`

| Type | Class | Description |
|---|---|---|
| Single | `dropping_ball` | Ball released from rest under gravity |
| Single | `falling_ball` | Free-falling balls of different sizes |
| Single | `sliding_cone` | Cone sliding on an inclined surface |
| Single | `pendulum` | Single pendulum oscillation |
| Single | `rotation` | Rotating cone, fixed camera |
| Multi | `hitting_cones` | Ball colliding with a pyramid of cones |
| Multi | `two_moving_pendulums` | Two pendulums released together, colliding |
| Multi | `two_moving_pendulum_one_static` | Moving pendulum strikes a static one |

```bash
pip install huggingface_hub
huggingface-cli download rasulkhanbayov/IRIS --repo-type dataset --local-dir ./IRIS
```

We also evaluate on **Delfys75** (75 real videos, 5 systems), available on [Kaggle](https://www.kaggle.com/datasets/jaswar/physical-parameter-prediction).

---

## Installation

```bash
conda env create -f environment.yml
conda activate physextraction
# or: pip install -r requirements.txt
```

For VLM-based equation selection, set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY=your_key_here     # copy .env.example to .env
```

**Tested with:** Python 3.12, PyTorch (CUDA 12.1), single NVIDIA GPU. See `requirements.txt` / `environment.yml` for pinned versions and `scripts/check_cuda.py` to verify your GPU setup.

---

## Reproducing the Paper

All commands assume the IRIS videos are downloaded to `./IRIS` and Delfys75 to `./data`.

### 1. Convert videos to tensors

```bash
python src/utils/video2npy.py --input_dir ./IRIS --output_dir ./IRIS_npy
```

Input/output tensors have shape `(N, 10, 1, H, W)` — N clips, 10 frames, 1 channel, default 56×100.

### 2. Equation-family selection (Table: routing accuracy)

```bash
# VLM and CNN routing on IRIS (240 videos, 8 classes)
bash scripts/run_iris_equation_selection_evals.sh

# Train / evaluate the fine-tuned video classifier on IRIS
python scripts/train_video_classifier.py    --path ./IRIS --out Results/video_classifier_iris --iris --epochs 25 --batch 8
python scripts/evaluate_video_classifier.py --path ./IRIS --checkpoint Results/video_classifier_iris/best.pt --out Results/video_classifier_eval_iris --iris
```

### 3. Parameter estimation — baseline, unified, multi-step (main results)

```bash
# IRIS: baseline + unified model
bash scripts/run_iris_baseline_and_unified.sh

# IRIS: multi-step physics loss
bash scripts/run_iris_multistep_only.sh

# Delfys75: baseline vs multi-step
bash scripts/run_delfys75_baseline_vs_multistep.sh

# Full baseline-vs-improved comparison
bash scripts/run_baseline_vs_improved.sh ./data 0.05
```

### 4. Compare against ground truth

```bash
python scripts/compare_baseline_unified.py     # maps fitted params to physical units, reports error vs GT
```

Outputs (CSVs, summaries, confusion matrices) are written under `Results/`. A **negative `Diff_err`** in `parameter_errors_comparison.txt` means the improved run beats the baseline. Reference outputs for every table in the paper are committed under [`Results/`](Results/) so you can diff your runs against ours.

### 5. Rebuttal experiments

Two additional studies requested during review. Reference outputs are committed; re-run to reproduce.

```bash
# Multi-clip vs. per-clip generalization (80/20 split per phenomenon).
# Pass --iris_root to use real IRIS .npy data, or --synthetic for known-physics data.
python scripts/run_multi_clip_iris.py --iris_root ./IRIS_npy --outfolder iris_multi_clip
#   -> Results/iris_multi_clip/{iris_multi_clip.csv, per_clip_on_test_split.csv, table_iris_multi_clip.tex}
#   Multi-clip beats per-clip on linear phenomena (e.g. dropping ball g: 0.207 -> 0.046)
#   but fails on the nonlinear pendulum — the open problem IRIS surfaces.

# Coupling-coefficient ablation for hitting cones (kappa=0 vs kappa=learned).
# Self-contained: generates known-physics data, no flags needed.
python scripts/run_hitting_cones_kappa_experiment.py
#   -> Results/hitting_cones_reconstruction/{reconstruction_comparison.csv, kappa_consistency.csv, table_reconstruction.tex}
#   Learned kappa reduces reconstruction MSE by 6-19%, confirming a real physical signal.
```

Both scripts use a fixed seed and resolve paths relative to the repo, so the committed numbers are reproducible.

---

## Repository Structure

```
.
├── main.py                     # Entry point: two-stage pipeline
├── config.yaml                 # Default training config
├── config_unified.yaml         # Config for the unified model
├── src/
│   ├── models/                 # Encoders, physics blocks, unified & graph models
│   ├── integrators/            # Euler, Störmer-Verlet, RK4, Yoshida4
│   ├── losses/                 # 1-step and multi-step physics loss
│   ├── analysis/               # Identifiability & energy tracking
│   └── utils/
│       ├── video2npy.py        # Video → tensor conversion
│       ├── vlm_dynamics.py     # VLM Stage 1 (basic)
│       ├── vlm_improved/       # VLM Stage 1 (temporal, 5 frames)
│       ├── vlm_finetune/       # Fine-tuned VLM classifier
│       └── video_classifier/   # ResNet-18 video classifier
├── scripts/                    # Training / evaluation / reproduction scripts
│   ├── run_multi_clip_iris.py              # Rebuttal: multi-clip vs per-clip
│   └── run_hitting_cones_kappa_experiment.py  # Rebuttal: kappa ablation
├── Results/                    # Reference outputs matching the paper
├── tests/                      # Unit tests (integrators, coupling, unified N=1)
├── docs/                       # Project page (GitHub Pages source)
├── environment.yml
└── requirements.txt
```

---

## Key Contributions

1. **Automatic equation-family selection from video** — temporal-reasoning VLM and a fine-tuned ResNet-18 classifier (100% on the IRIS evaluation set).
2. **Multi-step physics loss** — rollout supervision over horizons 1–5 improves long-horizon consistency and parameter identifiability.
3. **Unified physics model** — a single graph-structured architecture for both single- and multi-body systems, with a corrected gradient-passing Euler update.
4. **IRIS benchmark** — 240 real-world 4K videos, 8 dynamics, with measured ground truth and a standardized evaluation protocol.

---

## Tests

```bash
pytest tests/
```

---

## License

- **Code:** MIT — see [`LICENSE.txt`](LICENSE.txt).
- **IRIS dataset:** CC-BY-NC-4.0 (non-commercial), per the [Hugging Face dataset card](https://huggingface.co/datasets/rasulkhanbayov/IRIS).

---

## Citation

```bibtex
@inproceedings{khanbayov2026iris,
  title     = {{IRIS}: A Real-World Benchmark for Inverse Recovery and Identification
               of Physical Dynamic Systems from Monocular Video},
  author    = {Khanbayov, Rasul and Barhdadi, Mohamed Rayan and
               Serpedin, Erchin and Kurban, Hasan},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

<!-- repository metadata refresh -->
