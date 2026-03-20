#!/usr/bin/env bash
#
# Train baseline and unified models on IRIS, then compare.
# Run from project root: bash scripts/run_iris_baseline_and_unified.sh
#
# Optional: set SKIP_VIDEO2NPY=1 if .npy files already exist.
#
SKIP_VIDEO2NPY=1
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${SKIP_VIDEO2NPY}" != "1" ]]; then
  echo "=== 0) Converting IRIS videos to .npy ==="
  python -c "
from src.utils.video2npy import iterate_and_process_videos
iterate_and_process_videos('IRIS')
"
else
  echo "=== 0) Skipping video2npy (SKIP_VIDEO2NPY=1) ==="
fi

echo ""
echo "=== 1) Baseline (main.py) on IRIS ==="
python main.py --path ./IRIS --outfolder iris_baseline --dt 0.05

echo ""
echo "=== 2) Unified model on IRIS (multi-object for hitting_cones & two pendulums) ==="
python scripts/run_unified_delfys75.py --path ./IRIS --outfolder iris_unified --dt 0.05

echo ""
echo "=== 3) Compare baseline vs unified vs GT ==="
python scripts/compare_baseline_unified.py \
  --baseline Results/iris_baseline/iris_baseline.csv \
  --unified Results/iris_unified/iris_unified.csv \
  --params IRIS/parameters.json \
  --out Results/iris_comparison

echo ""
echo "Done. Results:"
echo "  Baseline CSV:  Results/iris_baseline/iris_baseline.csv"
echo "  Unified CSV:   Results/iris_unified/iris_unified.csv"
echo "  Comparison:    Results/iris_comparison/parameter_errors_comparison.txt"
