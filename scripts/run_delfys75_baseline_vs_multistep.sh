#!/usr/bin/env bash
#
# Run baseline (1-step) loss and multi-step loss on Delfys75, then compare.
# Both use main.py (path-based dynamics); only --loss differs.
#
# Usage (from project root: learning_from_physics):
#   bash scripts/run_delfys75_baseline_vs_multistep.sh
#   bash scripts/run_delfys75_baseline_vs_multistep.sh ./delfys75 0.05
#
# Output:
#   Results/delfys75_baseline/delfys75_baseline.csv
#   Results/delfys75_multistep/delfys75_multistep.csv
#   Results/delfys75_baseline_vs_multistep/parameter_errors_*.csv
#   Results/delfys75_baseline_vs_multistep/parameter_errors_comparison.txt
#   (Negative Diff_err = multi-step has lower error than baseline.)
#

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_PATH="${1:-./delfys75}"
DT="${2:-0.05}"
BASELINE_FOLDER="delfys75_baseline"
MULTISTEP_FOLDER="delfys75_multistep"
COMPARE_OUT="delfys75_baseline_vs_multistep"

PARAMS_JSON=""
for p in "$ROOT/Delfys75/parameters.json" "$ROOT/delfys75/parameters.json"; do
  if [[ -f "$p" ]]; then
    PARAMS_JSON="$p"
    break
  fi
done
if [[ -z "$PARAMS_JSON" ]]; then
  echo "Warning: parameters.json not found; comparison may miss GT. Use --params if needed."
fi

echo "=== 1) Baseline (1-step loss) ==="
python main.py --path "$DATA_PATH" --outfolder "$BASELINE_FOLDER" --dt "$DT" --loss latent_loss

echo ""
echo "=== 2) Multi-step loss (weighted 1..5 step) ==="
python main.py --path "$DATA_PATH" --outfolder "$MULTISTEP_FOLDER" --dt "$DT" --loss latent_loss_multistep

echo ""
echo "=== 3) Compare baseline vs multi-step ==="
COMPARE_ARGS=(
  --baseline "Results/$BASELINE_FOLDER/$BASELINE_FOLDER.csv"
  --unified "Results/$MULTISTEP_FOLDER/$MULTISTEP_FOLDER.csv"
  --out "Results/$COMPARE_OUT"
)
if [[ -n "$PARAMS_JSON" ]]; then
  COMPARE_ARGS+=(--params "$PARAMS_JSON")
fi
python scripts/compare_baseline_unified.py "${COMPARE_ARGS[@]}"

echo ""
echo "Done. Results in Results/$COMPARE_OUT/parameter_errors_comparison.txt (negative Diff_err = multi-step better)."
