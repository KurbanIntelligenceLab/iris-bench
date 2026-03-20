# PhysExtraction: Identifying Equations and Physical Parameters from Video

> ECCV 2026 Submission

This repository contains the implementation of a two-stage pipeline that **automatically identifies both the governing equation and its physical parameters** from video, without relying on folder structure or manual labelling.

---

## Overview

Most prior work assumes you already know which ODE governs the system (e.g. pendulum, free fall). This project removes that assumption:

**Stage 1 — Equation-family selection:** A Vision-Language Model (VLM) or a lightweight video classifier watches the video and selects which ODE from a library applies (pendulum, dropped ball, LED decay, free fall, sliding block, Torricelli flow).

**Stage 2 — Parameter estimation:** An MLP encoder maps video frames into a latent space where a physics block fits the selected ODE and estimates physical parameters (e.g. gravity *g*, pendulum length, friction coefficient, decay rate).

---

## Key Contributions

1. **Automatic equation-family selection from video** — VLM with temporal reasoning across 5 frames achieves ~73% accuracy on Delfys75; a fine-tuned video classifier achieves 100% on the evaluation set.

2. **Multi-step physics loss** — Instead of a 1-step MSE, we supervise rollouts at horizons 1..5 with weighted loss. Improves long-horizon consistency and parameter identifiability, especially for gravity and free-fall systems.

3. **Unified physics model** — A single graph-structured architecture handles both single-body and multi-body systems (linear coupling, contact, double pendulum). Includes a critical bug fix: the original Euler step did not pass gradients to physics parameters; the corrected update (`z_next = z + dt·ż + dt²·z̈`) makes training meaningful.

4. **Reproducible evaluation vs ground truth** — Standardized CSV output and a comparison script that maps alpha/beta to physical parameters and reports errors against Delfys75 ground truth.

---

## Supported Dynamics

| Dynamics | Physical parameters |
|---|---|
| `pendulum` | length, damping |
| `dropped_ball` / `bouncing_ball` | gravity *g* |
| `free_fall` | gravity *g* |
| `sliding_block` | friction coefficient |
| `led` | decay rate |
| `torricelli` | flow coefficient *k* |
| `two_moving_pendulums` | length, coupling |
| `rotation`, `sliding_cone`, `hitting_cones` | rotation / friction params |

---

## Installation

```bash
conda env create -f environment.yml
conda activate physextraction
```

For VLM-based equation selection, set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY=your_key_here
```

---

## Data Format

Input data should be NumPy arrays of shape `(N, 10, 1, H, W)` — N clips, 10 frames each, 1 channel, height × width (default 56×100).

To convert raw `.mp4` videos:

```bash
python src/utils/video2npy.py --input_dir videos/ --output_dir data/
```

### Delfys75

75 real-world videos across 5 physical systems (pendulum, Torricelli flow, sliding block, LED decay, free fall) with frame-wise object masks and parameter ground truth. Available on [Kaggle](https://www.kaggle.com/datasets/jaswar/physical-parameter-prediction).

### IRIS

A new dataset introduced in this work for multi-class physics recognition and parameter estimation. Available on [Hugging Face](https://huggingface.co/datasets/rasulkhanbayov/IRIS).

IRIS covers **8 dynamics classes**:

| Class | Description |
|---|---|
| `dropping_ball` | Ball released from rest under gravity |
| `falling_ball` | Projectile / free-falling ball |
| `sliding_cone` | Cone sliding on an inclined surface |
| `pendulum` | Single pendulum oscillation |
| `rotation` | Rotating object (camera fixed) |
| `hitting_cones` | Collision between cones |
| `two_moving_pendulums` | Two independently swinging pendulums |
| `two_moving_pendulum_one_static` | Two-pendulum system with one at rest |

To download IRIS and run evaluation:

```bash
# Download via Hugging Face CLI
pip install huggingface_hub
huggingface-cli download rasulkhanbayov/IRIS --repo-type dataset --local-dir ./IRIS

# Convert to .npy tensors
python src/utils/video2npy.py --input_dir ./IRIS --output_dir ./IRIS_npy

# Run parameter estimation on IRIS
python main.py --path ./IRIS_npy --outfolder results_iris --dt 0.0167
```

To train or evaluate the video dynamics classifier on IRIS:

```bash
# Train (25 epochs, 8 classes)
python scripts/train_video_classifier.py --path ./IRIS --out Results/video_classifier_iris --iris --epochs 25 --batch 8

# Evaluate
python scripts/evaluate_video_classifier.py --path ./IRIS --checkpoint Results/video_classifier_iris/best.pt --out Results/video_classifier_eval_iris --iris
```

---

## Quick Start

### Baseline (path-based dynamics, 1-step loss)

```bash
python main.py --path ./data --outfolder results_baseline --dt 0.05
```

### With VLM equation selection (improved)

```bash
python main.py --path ./data --outfolder results_vlm --dt 0.05 --vlm_improved
```

### With multi-step loss

```bash
python main.py --path ./data --outfolder results_multistep --dt 0.05 --loss latent_loss_multistep
```

### Unified model (single/multi-body, corrected Euler)

```bash
python scripts/run_unified_delfys75.py --path ./data --outfolder results_unified --dt 0.05
```

### Full baseline vs improved comparison

```bash
bash scripts/run_baseline_vs_improved.sh ./data 0.05
```

Results are written to `Results/` as CSVs. The comparison script produces `parameter_errors_comparison.txt` where a **negative Diff_err** means the improved run has lower error than the baseline.

---

## Project Structure

```
.
├── main.py                        # Entry point: two-stage pipeline
├── config.yaml                    # Default training config
├── config_unified.yaml            # Config for unified model
├── src/
│   ├── models/
│   │   ├── model.py               # EndPhys: encoder + physics block
│   │   ├── PhysModels.py          # ODE physics blocks (alpha, beta)
│   │   ├── unified_model.py       # Unified single/multi-body model
│   │   ├── encoder_unified.py     # Encoder for unified model
│   │   └── interaction_graph.py   # Graph-structured physics block
│   ├── integrators/
│   │   └── integrators.py         # Euler, Störmer-Verlet, RK4, Yoshida4
│   ├── losses/
│   │   └── loss.py                # 1-step and multi-step physics loss
│   ├── utils/
│   │   ├── video2npy.py           # Video-to-tensor conversion
│   │   ├── vlm_dynamics.py        # VLM Stage 1 (basic)
│   │   ├── vlm_improved/          # VLM Stage 1 (enhanced, 5 frames)
│   │   ├── vlm_finetune/          # Fine-tuned VLM classifier
│   │   └── video_classifier/      # ResNet-18 video classifier
│   ├── analysis/
│   │   ├── identifiability.py     # Parameter identifiability analysis
│   │   └── energy_tracker.py      # Energy tracking utilities
│   ├── loader.py                  # DataLoader utilities
│   ├── train.py                   # Training loop
│   └── loss_func.py               # Loss function factory
├── scripts/
│   ├── run_unified_delfys75.py    # Run unified model on Delfys75
│   ├── run_baseline_vs_improved.sh  # Full comparison pipeline
│   ├── compare_baseline_unified.py  # Compare runs vs GT
│   ├── evaluate_vlm_dynamics.py   # Evaluate VLM accuracy
│   └── train_video_classifier.py  # Train the video classifier
├── Results/                       # Output CSVs, plots, model checkpoints
├── environment.yml
└── requirements.txt
```

---

## Integrators

Beyond the default Euler step, the unified model supports:

- **Störmer-Verlet** — symplectic, conserves energy
- **RK4** — 4th-order Runge-Kutta
- **Yoshida4** — 4th-order symplectic

Set via `config_unified.yaml`: `integrator: euler | verlet | rk4 | yoshida4`

---

## Logging

Optional Weights & Biases logging. Set `log_wandb: True` in `config.yaml` and run `wandb login` before training.

---

## License

See LICENSE file for details.
