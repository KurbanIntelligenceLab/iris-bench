# All-in-one run: how it works and how the comparison was produced

## What “all-in-one” does (from your training logs)

Command run:
```bash
python scripts/run_unified_delfys75.py --path ./delfys75 --outfolder delfys75_all --dt 0.05 --vlm_improved --multistep
```

For **each video** (each `.npy` file), the pipeline does:

1. **Stage 1 – VLM equation-family selection**  
   The script calls the improved VLM on that video and gets a dynamics label, e.g. `dynamics=dropped_ball`, `dynamics=free_fall`, `dynamics=pendulum`.  
   You see this in the logs as: `Processing .../path/to/video.npy (dynamics=...)`.

2. **Unified model + multi-step loss**  
   The same unified physics model is trained on that video with **multi-step physics loss** (horizons 1..5, weighted). The model does **not** change with the VLM label; the label is only used for logging and for deciding whether to skip the file if no dynamics was found.

3. **CSV row**  
   After training, one row is written to `delfys75_all.csv`. The row’s **identity** (dynamics, setting, video_id) comes from the **folder path**, not from the VLM.  
   Example: file `delfys75/free_fall/mousepad/01/01.npy` → row with `run = free_fall, mousepad, 01` and the (alpha, beta) from the unified model for that video.

So “all-in-one” = **VLM (equation-family) + unified model + multi-step loss** in a single script; the CSV is still keyed by **path** so evaluation matches the same videos as baseline.

## Why the logs show different dynamics than the path

In the logs you see cases where the VLM label differs from the folder name, e.g.:

- `free_fall/mousepad/01/01.npy` → `(dynamics=pendulum)`  
- `dropped_ball/large/04/04.npy` → `(dynamics=free_fall)`  
- `led/led_2s/02/02.npy` → `(dynamics=free_fall)`

That only affects what is printed and (if we ever used it) skipping. It does **not** change:

- The CSV row: it is still `(dynamics, setting, video_id)` from the **path** (e.g. free_fall, mousepad, 01).
- The model: same unified model and loss for every video.
- Evaluation: ground truth and parameter mapping use **path** (dynamics/setting), so each row is compared to the correct GT (e.g. free_fall mousepad → g = 9.81).

So the comparison is path-consistent: same (dynamics, setting) across baseline and all-in-one.

## How `parameter_errors_comparison.txt` was produced

That file is **not** produced by the training script. It is produced by the comparison script when you run something like:

```bash
python scripts/compare_baseline_unified.py --baseline <path/to/baseline.csv> --unified Results/delfys75_all/delfys75_all.csv --out Results/delfys75_all
```

- **Baseline CSV**: e.g. main.py run (path-based or equation-family), so “Base_est” / “Base_|Err|” come from that run.
- **Unified CSV**: `delfys75_all.csv` from the all-in-one run above, so “Unif_est” / “Unif_|Err|” are the **all-in-one** estimates.

The script:

1. Loads both CSVs (main.py format: run = dynamics, setting, video_id; then alpha, beta, …).
2. Groups rows by **(dynamics, setting)** and averages estimates per (dynamics, setting).
3. Maps (alpha, beta) → physical params (g, decay, length_m, etc.) using the **path** (dynamics, setting).
4. Computes absolute error vs ground truth (from `Delfys75/parameters.json` and fixed g for dropped_ball/free_fall).
5. Writes the comparison table: for each (dynamics, setting, param), GT, Unif_est, Unif_|Err|, Base_est, Base_|Err|, and **Diff_err** = Unif_|Err| − Base_|Err| (negative ⇒ all-in-one is better).

So the table you see is: **baseline (main.py) vs all-in-one (unified + VLM + multistep)**. The training logs you shared are the run that produced the **Unif_est** column in that comparison.

## Summary

| Item | Meaning |
|------|--------|
| **All-in-one run** | `run_unified_delfys75.py --vlm_improved --multistep` → one pipeline: VLM per video + unified model + multi-step loss. |
| **CSV row identity** | Always from **folder path** (dynamics, setting, video_id). VLM label is for logging only. |
| **parameter_errors_comparison.txt** | Output of `compare_baseline_unified.py`: baseline CSV vs `delfys75_all.csv`. “Unif_*” = all-in-one. |
| **Diff_err** | Unif_|Err| − Base_|Err|. **Negative** ⇒ all-in-one has lower error (better). |

So the comparison file is exactly the “all-in-one mode” (unified + VLM + multistep) vs whatever baseline CSV you passed to `compare_baseline_unified.py`.
