<div align="left">

<h1>IRIS: A Real-World Benchmark for Inverse Recovery and Identification of Physical Dynamic Systems from Monocular Video</h1>

<p>
  <a href="https://www.linkedin.com/in/rasulkhanbayov/">Rasul Khanbayov</a><sup>*</sup>,
  <a href="https://bmrayan.com/">Mohamed Rayan Barhdadi</a><sup>*</sup>,
  <a href="https://scholar.google.com/citations?user=THi_ypMAAAAJ&amp;hl=en&amp;authuser=3&amp;oi=ao">Erchin Serpedin</a>,
  <a href="https://www.hasankurban.com/">Hasan Kurban</a>
</p>

<p><strong><sup>*</sup>Equal contribution</strong></p>
<p><strong>Accepted to ECCV 2026 🎉</strong></p>

<p>
  <a href="https://arxiv.org/pdf/2603.16432"><img src="assets/arxiv-link-logo.svg" alt="arXiv" width="18" height="18" align="absmiddle"> <strong>Paper (arXiv)</strong></a> ·
  <a href="https://kurbanintelligencelab.github.io/iris-bench/"><img src="assets/project-page-logo.svg" alt="Project page" width="18" height="18" align="absmiddle"> <strong>Project Page</strong></a> ·
  <a href="https://huggingface.co/datasets/rasulkhanbayov/IRIS"><img src="gifs/huggingface-logo.svg" alt="Hugging Face" width="19" height="18" align="absmiddle"> <strong>Dataset</strong></a>
</p>

<img src="assets/Benchmark_Preview.png" alt="IRIS benchmark preview" width="100%">

</div>

This repository contains the **IRIS benchmark** and the code for a two-stage pipeline that **automatically identifies both the governing equation and its physical parameters from video**, without relying on folder structure or manual labelling.

## Overview

IRIS uses a two-stage pipeline to identify the governing equation and estimate its physical parameters directly from video:

- **Stage 1 — Equation-family selection:** A Vision-Language Model (VLM) or a fine-tuned video classifier analyzes the video and selects the applicable ODE from the library.
- **Stage 2 — Parameter estimation:** An encoder maps video frames into a latent space where a physics block fits the selected ODE and estimates physical parameters (e.g. gravity *g*, pendulum length, friction, decay rate).

We also introduce **IRIS**, a 4K real-world benchmark of **240 videos** across **8 dynamics classes** (single- and multi-body) with independently measured ground-truth parameters.

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

## Quick Start

After installing the dependencies and downloading IRIS to `./IRIS`, convert the videos to tensors:

```bash
python src/utils/video2npy.py --input_dir ./IRIS --output_dir ./IRIS_npy
```

Run equation-family selection:

```bash
bash scripts/run_iris_equation_selection_evals.sh
```

Run the parameter-estimation baselines:

```bash
bash scripts/run_iris_baseline_and_unified.sh
bash scripts/run_iris_multistep_only.sh
```

Compare the estimated parameters with the ground truth:

```bash
python scripts/compare_baseline_unified.py
```

Generated reports and evaluation results are saved under [`Results/`](Results/).

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
├── scripts/                    # Training and evaluation scripts
│   ├── run_multi_clip_iris.py              # Multi-clip evaluation
│   └── run_hitting_cones_kappa_experiment.py  # Coupling-coefficient ablation
├── Results/                    # Reference outputs matching the paper
├── tests/                      # Unit tests (integrators, coupling, unified N=1)
├── assets/                     # README preview image
├── docs/                       # Project page (GitHub Pages source)
├── environment.yml
└── requirements.txt
```

## Key Contributions

1. **Automatic equation-family selection from video** — temporal-reasoning VLM and a fine-tuned ResNet-18 classifier (100% on the IRIS evaluation set).
2. **Multi-step physics loss** — rollout supervision over horizons 1–5 improves long-horizon consistency and parameter identifiability.
3. **Unified physics model** — a single graph-structured architecture for both single- and multi-body systems, with a corrected gradient-passing Euler update.
4. **IRIS benchmark** — 240 real-world 4K videos, 8 dynamics, with measured ground truth and a standardized evaluation protocol.

## Tests

```bash
pytest tests/
```

## License

- **Code:** MIT — see [`LICENSE.txt`](LICENSE.txt).
- **IRIS dataset:** CC-BY-NC-4.0 (non-commercial), per the [Hugging Face dataset card](https://huggingface.co/datasets/rasulkhanbayov/IRIS).

## Citation

```bibtex
@misc{khanbayov2026iris,
  title={IRIS: A Real-World Benchmark for Inverse Recovery and Identification of Physical Dynamic Systems from Monocular Video},
  author={Rasul Khanbayov and Mohamed Rayan Barhdadi and Erchin Serpedin and Hasan Kurban},
  year={2026},
  eprint={2603.16432},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.16432},
}
```

<!-- repository metadata refresh -->
