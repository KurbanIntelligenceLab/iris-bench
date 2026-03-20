#!/usr/bin/env bash
# Run VLM (temporal), VLM (describe-then-classify), and optionally CNN on IRIS.
# Requires: OPENROUTER_API_KEY for VLM. For CNN: either 8-class checkpoint (train with --iris) or 6-class checkpoint for --iris_6class.
#
# Usage (from project root):
#   bash scripts/run_iris_equation_selection_evals.sh
#   bash scripts/run_iris_equation_selection_evals.sh /path/to/cnn_checkpoint.pt  # 6-class for 4-class subset, or 8-class with --iris

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
IRIS="${1:-./IRIS}"

echo "=== 1) VLM temporal reasoning on IRIS (8 classes) ==="
python3 scripts/evaluate_vlm_on_iris.py --path "$IRIS" --out Results/vlm_eval_iris

echo ""
echo "=== 2) VLM describe-then-classify on IRIS (8 classes) ==="
python3 scripts/evaluate_vlm_on_iris.py --path "$IRIS" --out Results/vlm_eval_iris_describe_then --describe_then_classify

echo ""
echo "=== 3) CNN on IRIS ==="
if [[ -n "$2" ]]; then
  # Optional: 8-class checkpoint
  python3 scripts/evaluate_video_classifier.py --path "$IRIS" --checkpoint "$2" --out Results/video_classifier_eval_iris --iris
elif [[ -f "Results/video_classifier_iris/best.pt" ]]; then
  python3 scripts/evaluate_video_classifier.py --path "$IRIS" --checkpoint Results/video_classifier_iris/best.pt --out Results/video_classifier_eval_iris --iris
elif [[ -f "Results/video_classifier/best.pt" ]]; then
  echo "Using 6-class CNN on 4-class IRIS subset (120 videos)"
  python3 scripts/evaluate_video_classifier.py --path "$IRIS" --checkpoint Results/video_classifier/best.pt --out Results/video_classifier_eval_iris --iris_6class
else
  echo "No CNN checkpoint found. Train with: python3 scripts/train_video_classifier.py --path ./IRIS --out Results/video_classifier_iris --iris"
fi

echo ""
echo "Done. Update paper_drafts/table_iris_equation_selection.tex with accuracies from:"
echo "  Results/vlm_eval_iris/vlm_iris_summary.txt"
echo "  Results/vlm_eval_iris_describe_then/vlm_iris_summary.txt"
echo "  Results/video_classifier_eval_iris/video_classifier_eval_summary.txt (if CNN was run)"
