"""
Evaluate trained video dynamics classifier on Delfys75 (same format as VLM evaluation).

Usage:
  python scripts/evaluate_video_classifier.py --path ./delfys75 --checkpoint Results/video_classifier/best.pt --out Results/video_classifier_eval
  python scripts/evaluate_video_classifier.py --path ./IRIS --checkpoint Results/video_classifier/best.pt --out Results/video_classifier_eval_iris --iris_6class
"""

import argparse
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.video_classifier import (
    get_gt_dynamics_from_path,
    get_gt_dynamics_from_path_iris,
    load_classifier,
    predict_dynamics_from_npy,
    DYNAMICS_CLASSES,
    IRIS_DYNAMICS_CLASSES,
)

# IRIS class -> 6-class label (for evaluating Delfys75-trained CNN on IRIS subset)
IRIS_TO_6CLASS = {
    "dropping_ball": "dropped_ball",
    "falling_ball": "free_fall",
    "sliding_cone": "sliding_block",
    "pendulum": "pendulum",
}
IRIS_6CLASS_SUBSET = set(IRIS_TO_6CLASS.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default="./delfys75")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    ap.add_argument("--out", type=str, default="Results/video_classifier_eval")
    ap.add_argument("--iris", action="store_true", help="Evaluate on IRIS 8-class (checkpoint must be 8-class trained on IRIS)")
    ap.add_argument("--iris_6class", action="store_true", help="Evaluate 6-class CNN on IRIS: only videos with dropping_ball, falling_ball, sliding_cone, pendulum (4 classes, 120 videos)")
    args = ap.parse_args()

    if args.iris_6class:
        # 6-class checkpoint, IRIS path: only include 4 overlapping classes, map GT to 6-class for comparison
        num_classes = 6
        model = load_classifier(args.checkpoint, num_classes=6)
        rows = []
        for root, _, files in os.walk(args.path):
            for f in files:
                if not f.endswith(".npy"):
                    continue
                file_path = os.path.join(root, f)
                gt_iris = get_gt_dynamics_from_path_iris(file_path)
                if gt_iris is None or gt_iris not in IRIS_6CLASS_SUBSET:
                    continue
                gt_mapped = IRIS_TO_6CLASS[gt_iris]
                pred = predict_dynamics_from_npy(file_path, model=model, num_classes=6, classes=None)
                if pred is None:
                    pred = ""
                correct = pred == gt_mapped
                rows.append({"path": file_path, "gt": gt_iris, "gt_6class": gt_mapped, "pred": pred, "correct": correct})
    else:
        num_classes = 8 if args.iris else 6
        model = load_classifier(args.checkpoint, num_classes=num_classes)
        get_gt = get_gt_dynamics_from_path_iris if args.iris else get_gt_dynamics_from_path
        classes = IRIS_DYNAMICS_CLASSES if args.iris else None
        rows = []
        for root, _, files in os.walk(args.path):
            for f in files:
                if not f.endswith(".npy"):
                    continue
                file_path = os.path.join(root, f)
                gt = get_gt(file_path)
                if gt is None:
                    continue
                pred = predict_dynamics_from_npy(file_path, model=model, num_classes=num_classes, classes=classes)
                if pred is None:
                    pred = ""
                correct = pred == gt
                rows.append({"path": file_path, "gt": gt, "pred": pred, "correct": correct})

    if not rows:
        print("No .npy files with known dynamics found.")
        return

    correct_count = sum(r["correct"] for r in rows)
    accuracy = correct_count / len(rows)
    gt_all = [r["gt"] for r in rows]
    pred_all = [r["pred"] for r in rows]
    classes = sorted(set(gt_all))

    print("\n--- Video classifier dynamics evaluation ---")
    if args.iris_6class:
        print("Mode: 6-class CNN on IRIS (4 overlapping classes: dropping_ball, falling_ball, sliding_cone, pendulum)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data path: {args.path}")
    print(f"Total videos: {len(rows)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print("Per-class accuracy (GT class -> accuracy):")
    per_class = {}
    for c in classes:
        indices = [i for i, g in enumerate(gt_all) if g == c]
        if args.iris_6class:
            correct_c = sum(1 for i in indices if rows[i]["correct"])
        else:
            correct_c = sum(1 for i in indices if pred_all[i] == c)
        acc_c = correct_c / len(indices) if indices else 0
        per_class[c] = acc_c
        print(f"  {c}: {correct_c}/{len(indices)} = {acc_c:.2%}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "video_classifier_eval_summary.txt"), "w") as f:
        f.write("Video classifier dynamics evaluation\n")
        if args.iris_6class:
            f.write("Mode: 6-class CNN on IRIS (4 overlapping classes)\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Data path: {args.path}\n")
        f.write(f"Total videos: {len(rows)}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Per-class accuracy: {per_class}\n")
    csv_fields = ["path", "gt", "gt_6class", "pred", "correct"] if args.iris_6class else ["path", "gt", "pred", "correct"]
    with open(os.path.join(args.out, "video_classifier_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}/video_classifier_eval_summary.txt and video_classifier_results.csv")


if __name__ == "__main__":
    main()
