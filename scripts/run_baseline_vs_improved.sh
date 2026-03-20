#!/usr/bin/env bash
#
# One pipeline: run baseline and improved (unified) on Delfys75, then compare.
# Usage (from project root: learning_from_physics):
#   bash scripts/run_baseline_vs_improved.sh
#   bash scripts/run_baseline_vs_improved.sh ./delfys75 0.05
#
# Output:
#   Results/delfys75_baseline/delfys75_baseline.csv
#   Results/delfys75_improved/delfys75_improved.csv
#   Results/delfys75_comparison/parameter_errors_baseline.csv
#   Results/delfys75_comparison/parameter_errors_improved.csv
#   Results/delfys75_comparison/parameter_errors_comparison.txt  (negative Diff_err = improved better)
#

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_PATH="${1:-./delfys75}"
DT="${2:-0.05}"
BASELINE_FOLDER="delfys75_baseline"
IMPROVED_FOLDER="delfys75_improved"
COMPARE_OUT="delfys75_comparison"

# Ground truth: try Delfys75 then delfys75 (lowercase)
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

echo "=== 1) Baseline (path-based dynamics, 1-step loss) ==="
python main.py --path "$DATA_PATH" --outfolder "$BASELINE_FOLDER" --dt "$DT"

echo ""
echo "=== 2) Improved (unified model) ==="
python scripts/run_unified_delfys75.py --path "$DATA_PATH" --outfolder "$IMPROVED_FOLDER" --dt "$DT"

echo ""
echo "=== 3) Compare baseline vs improved ==="
COMPARE_ARGS=(
  --baseline "Results/$BASELINE_FOLDER/$BASELINE_FOLDER.csv"
  --unified "Results/$IMPROVED_FOLDER/$IMPROVED_FOLDER.csv"
  --out "Results/$COMPARE_OUT"
)
if [[ -n "$PARAMS_JSON" ]]; then
  COMPARE_ARGS+=(--params "$PARAMS_JSON")
fi
python scripts/compare_baseline_unified.py "${COMPARE_ARGS[@]}"

echo ""
echo "Done. Check Results/$COMPARE_OUT/parameter_errors_comparison.txt (negative Diff_err = improved better)."
