# Improved VLM dynamics detection

This folder contains the **improved** VLM-based dynamics classifier used to show that better prompts and more frames yield higher accuracy.

## Changes vs baseline (`src/utils/vlm_dynamics.py`)

1. **Enhanced prompt**: Per-class one-line descriptions so the model can tell similar cases apart:
   - **free_fall**: ball filmed from above; apparent size shrinks (scale change).
   - **dropped_ball**: ball falling in frame (e.g. side view).
   - **pendulum**: object swinging back and forth (oscillation).
   - (Same for sliding_block, led, torricelli.)

2. **More frames**: 5 frames (start, 25%, 50%, 75%, end) instead of 3 for better temporal signal.

## How to compare

**Evaluation only (no training):**
```bash
# Baseline (3 frames, short prompt)
python scripts/evaluate_vlm_dynamics.py --path ./delfys75 --out ./Results/vlm_eval

# Improved (5 frames, enhanced prompt)
python scripts/evaluate_vlm_dynamics_improved.py --path ./delfys75 --out ./Results/vlm_eval_improved
```
Compare `Results/vlm_eval/vlm_eval_summary.txt` vs `Results/vlm_eval_improved/vlm_eval_summary.txt` (and confusion matrices).

**Full pipeline with improved VLM:**
```bash
python main.py --path ./delfys75 --outfolder delfys75 --dt 0.05 --vlm_improved
```
Metrics are written to `Results/delfys75/vlm_accuracy.txt` (and vlm_results.csv, vlm_confusion_matrix.csv).
