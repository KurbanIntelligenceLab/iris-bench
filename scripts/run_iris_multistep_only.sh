#!/usr/bin/env bash
#
# Run ONLY the unified model with multi-step loss on IRIS, then compare with
# existing baseline and 1-step unified results.
# Prerequisite: baseline and 1-step unified already run (run_iris_baseline_and_unified.sh).
# Run from project root: bash scripts/run_iris_multistep_only.sh
#
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== 1) Unified model on IRIS with multi-step loss ==="
python scripts/run_unified_delfys75.py --path ./IRIS --outfolder iris_unified_multistep --dt 0.05 --multistep

echo ""
echo "=== 2) Compare baseline vs 1-step unified vs multi-step unified ==="
python scripts/compare_baseline_unified.py \
  --baseline Results/iris_baseline/iris_baseline.csv \
  --unified Results/iris_unified/iris_unified.csv \
  --unified_multistep Results/iris_unified_multistep/iris_unified_multistep.csv \
  --params IRIS/parameters.json \
  --out Results/iris_comparison

echo ""
echo "Done. Multi-step results:"
echo "  CSV:   Results/iris_unified_multistep/iris_unified_multistep.csv"
echo "  3-way: Results/iris_comparison/parameter_errors_comparison_3way.txt"
