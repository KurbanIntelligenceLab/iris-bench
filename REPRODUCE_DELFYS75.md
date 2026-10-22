# Reproducing the Delfys75 results (paper Tables 4 and 5)

This guide gives the exact data, code version, settings and commands behind the Delfys75
results of the IRIS paper, together with a re-run from the raw videos.

## Summary

| Result | How to reproduce | Outcome |
|---|---|---|
| Table 5 (all values) | `bash scripts/reproduce_delfys75.sh table5` (CPU, seconds) | regenerated exactly from the released CSVs ([`Results/repro_d75/table5/checks.txt`](Results/repro_d75/table5/checks.txt)) |
| Table 4, "Base." column | `data` → `preprocess` → `table4` (GPU) | re-run from the raw videos, scored with the released script, reproduces the published numbers (table below) |
| Table 4, "Corr." column | same `table4` step | the original runs of `scripts/run_unified_delfys75.py` were not seeded, so this column varies between runs; the script now takes `--seed` (default 42) |
| Table 4, Dropped Ball / Free Fall / Sliding Block | — | no valid result in the published setting: these clips are too short for stride 6 (see below) |

## Setting used for Table 4

| Item | Value |
|---|---|
| Data | Delfys75 on Kaggle, [`jaswar/physical-parameter-prediction`](https://www.kaggle.com/datasets/jaswar/physical-parameter-prediction), version 11 (zip SHA-256 `1d2320351105ceb0b90c8a926d815acb0fe4eb629ed859318b90e269853c7137`) |
| Code | the code version that produced Table 4 (2026-03-03), included unchanged in [`repro_d75/table4_code/`](repro_d75/table4_code/README.md) |
| Preprocessing | 100×56 grayscale, 10 frames per sample, fixed stride 6 at 60 fps, i.e. Δt = 0.1 s between frames of a sample |
| Training | "Base.": `main.py`; "Corr.": `scripts/run_unified_delfys75.py`; both with `--dt 0.05` (the default of `scripts/run_baseline_vs_improved.sh`), one-step loss, 500 epochs. Seeds: `main.py` sets `torch.manual_seed(42)` per clip; the original `run_unified_delfys75.py` runs were unseeded (the re-run uses seed 42) |
| Released outputs | `Results/delfys75_baseline/delfys75_baseline.csv`, `Results/delfys75_improved/delfys75_improved.csv` |
| Scoring | `scripts/compare_baseline_unified.py` → `Results/delfys75_comparison/parameter_errors_comparison.txt` |
| Failed runs | written as all-zero rows by the pipeline; Table 4 counts them as an estimate of 0 over the 5 clips of a setting |

Clips of 60 frames or fewer give no samples at stride 6 (samples = frames − 60). This is the case for every
dropped-ball (27–33 frames), free-fall (16–23) and sliding-block (39–57) clip
([`preprocess_s6_report.csv`](Results/repro_d75/preprocessing/preprocess_s6_report.csv)).
These 45 clips are the 45 all-zero rows of both released CSVs; the baseline CSV has three
more (`led/led_5s`, `torricelli/large`, `torricelli/medium`). The original run logs were not
retained; in our re-run the same kind of failure (a NaN loss, which this code version turns into
an exception) occurred for `torricelli/small/01`.
The current `src/utils/video2npy.py` uses a smaller stride for short videos, so it produces
samples for these clips; use `repro_d75/video2npy_fixed.py --stride 6` to obtain the Table 4 data.

The sample counts printed in the archived log of that period (`Results/delfys75_1/delfys75.log`,
available in this repository's history at commit `d41564e`) match stride 6 for all 45 clips that
loaded ([`stride_evidence.txt`](Results/repro_d75/preprocessing/stride_evidence.txt)).

## Re-run of Table 4 from the raw videos

Released scoring script, published setting
([`Results/repro_d75/table4_published_setting/`](Results/repro_d75/table4_published_setting/parameter_errors_comparison.txt)):

| Row | Paper "Base." | Re-run "Base." | Paper "Corr." | Re-run "Corr." (seed 42) |
|---|---|---|---|---|
| Pendulum 45, L | 95.07 | 95.54 | 3.80 | 18.57 |
| Pendulum 90, L | 50.15 | 50.87 | 2.93 | 9.60 |
| Pendulum 150, L | 29.83 | 29.65 | 0.71 | 1.74 |
| LED 2 s, γ | 2.30 | 2.30 | 0.40 | 0.56 |
| Torricelli large, k | 6.07 | 9.78 | 2.91 | 7.46 |
| Torricelli small, k | 7.67 | 7.53 | 5.69 | 1.31 |

Notes on the comparison:

- Run-to-run differences remain on GPU, for example where single Torricelli clips converge to different values.
- The "Corr." values depend on the seed.
- The published "Base." Torricelli large value (6.07) includes one failed clip counted as 0 (`large/04`); without it, the 4-clip MAE is 7.59.
- In the re-run, `torricelli/small/01` failed and is excluded (N = 4 in `parameter_errors_baseline.csv`), while the published setting counted failed clips as 0.

## Table 5

Table 5 is `Results/delfys75/parameter_errors_comparison.txt`:

- **"1-step"** is `Results/delfys75_original/delfys75.csv`.
- **"Multi"** is `Results/delfys75/delfys75.csv`. This run also used VLM-based equation-family selection; see `Results/delfys75/vlm_results.csv`.

Both Table 5 columns are runs of the `main.py` pipeline (the Table 4 "Base." model). They are separate runs from those of Table 4, and are evaluated with a separate script. Failed runs are excluded, and the number of clips used is reported per row; for example, N = 4 for `free_fall/mousepad`.

The original Table 5 evaluation script was not retained. `scripts/evaluate_delfys75.py` is a re-implementation that regenerates every value from the two CSVs:

```bash
bash scripts/reproduce_delfys75.sh table5
```

## Commands

```bash
git clone https://github.com/KurbanIntelligenceLab/iris-bench && cd iris-bench
pip install -r repro_d75/requirements_repro.txt      # pinned versions used for the re-run
bash scripts/reproduce_delfys75.sh table5             # Table 5 (CPU)
export KAGGLE_API_TOKEN=...                           # kaggle.com -> Settings -> API
bash scripts/reproduce_delfys75.sh data               # Delfys75 v11 -> ./delfys75_raw
bash scripts/reproduce_delfys75.sh preprocess         # stride 6 -> ./delfys75_s6 + SHA-256 manifest
bash scripts/reproduce_delfys75.sh table4             # both Table 4 columns (GPU, a few hours)
```

Outputs go to `Results/repro_d75_local/`. Existing folders are never overwritten.

To check your `.npy` files, compare `delfys75_s6.sha256.tsv` with
[`Results/repro_d75/preprocessing/delfys75_s6.sha256.tsv`](Results/repro_d75/preprocessing/delfys75_s6.sha256.tsv).
Identical hashes require the same video decoder build (`opencv-python-headless==4.10.0.84`, as pinned).
The `data` step checks the zip against the SHA-256 above; Kaggle may re-package a dataset, in
which case set `D75_SKIP_ZIP_CHECK=1` and rely on the `.npy` manifest instead.

`repro_d75/run_baseline.py` and `repro_d75/run_unified.py` call the released `main.py` and
`run_unified_delfys75.py` functions clip by clip, unchanged. Around them they:

- record a failed clip with its error message in `clips.csv`, instead of a row of zeros. Clips where training stopped on a NaN loss, or returned before the first epoch, get their own status and are excluded from the MAE;
- seed each clip (`main.py` already seeds 42; the unified runner seeds explicitly);
- delete `best_model.pt` before each clip, because `main.py` would otherwise load the previous clip's checkpoint when a clip saves none;
- process clips in sorted order.

As in the released script, the unified model is fit to the first 10-frame window of each clip.

## Environment of the reference re-run

Python 3.12, PyTorch 2.5.1 (CUDA 12.1), single NVIDIA A100 80 GB / H200 GPU per run. Exact
package versions are in each run folder's `pip_freeze.txt`, and the commits in `provenance.txt`.
Two things in these records don't match the committed files exactly:

- **Tool hashes.** The hashes in `provenance.txt` are of the tool versions at run time. Since then, `run_baseline.py` gained the `--seed` option (default unchanged), `run_unified.py` a fallback for older code, and `evaluate_delfys75.py` only docstring edits.
- **Preprocessing logs.** These also cover two further preprocessing variants of the same job that are not used here.
- **Paths.** Absolute paths in the records are shortened to placeholders such as `<repo>/`, `<delfys75_raw>/` and `<delfys75_s6>/`.

## Files

| Path | Content |
|---|---|
| `scripts/reproduce_delfys75.sh` | the commands above |
| `scripts/evaluate_delfys75.py` | Table 5 evaluation (failed runs excluded, N reported) |
| `repro_d75/video2npy_fixed.py`, `verify_npy.py` | preprocessing at a fixed stride, shape and checksum verification |
| `repro_d75/run_baseline.py`, `run_unified.py` | clip-by-clip runners with failure recording and seeding |
| `repro_d75/stride_evidence.py` | matches logged sample counts to video frame counts |
| `repro_d75/table4_code/` | the code version that produced Table 4 (unchanged copy) |
| `Results/repro_d75/` | reference outputs of the re-run (reports, manifests, per-clip CSVs, logs) |
