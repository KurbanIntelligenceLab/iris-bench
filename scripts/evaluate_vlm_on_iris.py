"""
Evaluate VLM (improved) dynamics classification on the IRIS dataset.

With --iris_classes (default): uses IRIS-extended prompt (rotation, hitting_cones,
two_moving_pendulums, two_moving_pendulum_one_static) so all 8 IRIS dynamics can be
classified and we report accuracy over all classes.

Without --iris_classes: uses the original 6-class VLM; we report accuracy on the
4 comparable classes and prediction distribution for the rest.

Usage:
  python scripts/evaluate_vlm_on_iris.py --path ./IRIS [--out ./Results/vlm_eval_iris]
  python scripts/evaluate_vlm_on_iris.py --path ./IRIS --out ./Results/vlm_eval_iris --no_iris_classes  # original 6-class only
  Set OPENROUTER_API_KEY for the VLM API.
"""

import os
import sys
import argparse
import csv
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# IRIS dynamics from path; order matters (longer / more specific first)
IRIS_DYNAMICS_KEYWORDS = [
    "two_moving_pendulum_one_static",
    "two_moving_pendulums",
    "dropping_ball",
    "falling_ball",
    "sliding_cone",
    "hitting_cones",
    "rotation",
    "pendulum",
]


def get_gt_dynamics_from_path(path: str) -> str | None:
    """Ground-truth dynamics from IRIS folder path."""
    normalized = path.replace(" ", "").lower()
    for kw in IRIS_DYNAMICS_KEYWORDS:
        if kw in normalized:
            return kw
    return None


# Map VLM/IRIS-extended output back to IRIS folder names for comparison
VLM_TO_IRIS = {
    "dropped_ball": "dropping_ball",
    "free_fall": "falling_ball",
    "sliding_block": "sliding_cone",
    "pendulum": "pendulum",
    "rotation": "rotation",
    "hitting_cones": "hitting_cones",
    "two_moving_pendulums": "two_moving_pendulums",
    "two_moving_pendulum_one_static": "two_moving_pendulum_one_static",
    "led": "led",
    "torricelli": "torricelli",
}


def run_evaluation(
    data_path: str,
    output_dir: str,
    use_describe_then_classify: bool = False,
    use_iris_classes: bool = True,
) -> None:
    from src.utils.vlm_improved import detect_dynamics_from_npy

    rows = []
    for root, _, files in os.walk(data_path):
        for f in files:
            if not f.endswith(".npy"):
                continue
            file_path = os.path.join(root, f)
            gt_iris = get_gt_dynamics_from_path(file_path)
            if gt_iris is None:
                continue
            pred = detect_dynamics_from_npy(
                file_path,
                use_temporal_reasoning=not use_describe_then_classify,
                use_describe_then_classify=use_describe_then_classify,
                use_iris_classes=use_iris_classes,
            )
            if pred is None:
                pred = ""
            pred_iris = VLM_TO_IRIS.get(pred, pred)  # map to IRIS name when possible
            correct = pred_iris == gt_iris
            comparable = use_iris_classes or gt_iris in ("dropping_ball", "falling_ball", "sliding_cone", "pendulum")
            rows.append({
                "path": file_path,
                "gt_iris": gt_iris,
                "pred_raw": pred,
                "pred_iris": pred_iris,
                "comparable": comparable,
                "correct": correct,
            })
            rel = os.path.relpath(file_path, data_path)
            print(f"  {rel}: gt={gt_iris} pred={pred_iris} correct={correct}")

    if not rows:
        print("No .npy files with known IRIS dynamics found.")
        return

    comparable_rows = [r for r in rows if r["comparable"]]
    if comparable_rows:
        correct_count = sum(r["correct"] for r in comparable_rows)
        accuracy = correct_count / len(comparable_rows)
        mode = "IRIS-extended (all 8 classes)" if use_iris_classes else "comparable only (4 classes)"
        print(f"\n--- VLM on IRIS [{mode}] ---")
        print(f"Videos: {len(comparable_rows)}")
        print(f"Correct: {correct_count}")
        print(f"Accuracy: {accuracy:.2%}")
        gt_all = [r["gt_iris"] for r in comparable_rows]
        pred_all = [r["pred_iris"] for r in comparable_rows]
        classes = sorted(set(gt_all))
        print("\nPer-class accuracy (GT class -> accuracy):")
        for c in classes:
            indices = [i for i, r in enumerate(comparable_rows) if r["gt_iris"] == c]
            n = len(indices)
            correct_c = sum(1 for i in indices if comparable_rows[i]["pred_iris"] == c)
            print(f"  {c}: {correct_c}/{n} = {correct_c/n:.2%}")
    else:
        accuracy = 0.0
        comparable_rows = []
        gt_all = []
        pred_all = []
        classes = []

    # When not using IRIS classes: show prediction distribution for IRIS-only GT classes
    iris_only = [r for r in rows if not r["comparable"]]
    by_gt = defaultdict(list)
    for r in iris_only:
        by_gt[r["gt_iris"]].append(r["pred_raw"] or "<empty>")
    if iris_only and not use_iris_classes:
        print("\n--- IRIS-only classes (no VLM label; prediction distribution) ---")
        for gt_name in sorted(by_gt.keys()):
            preds = by_gt[gt_name]
            counts = defaultdict(int)
            for p in preds:
                counts[p] += 1
            total = len(preds)
            dist = ", ".join(f"{p}: {c}" for p, c in sorted(counts.items(), key=lambda x: -x[1]))
            print(f"  {gt_name} ({total} videos): {dist}")

    # Confusion matrix (comparable only)
    try:
        from sklearn.metrics import confusion_matrix, classification_report
    except ImportError:
        sklearn_available = False
    else:
        sklearn_available = True

    if comparable_rows and sklearn_available:
        y_true = [r["gt_iris"] for r in comparable_rows]
        y_pred = [r["pred_iris"] if r["pred_iris"] else "<empty>" for r in comparable_rows]
        labels = sorted(set(y_true) | set(y_pred))
        if "<empty>" in labels:
            labels.remove("<empty>")
            labels.append("<empty>")
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print("\nConfusion matrix (rows=GT, cols=pred) [comparable only]:")
        print("Labels:", labels)
        print(cm)
        try:
            print(classification_report(y_true, y_pred, labels=[l for l in labels if l != "<empty>"], zero_division=0))
        except TypeError:
            print(classification_report(y_true, y_pred, labels=[l for l in labels if l != "<empty>"]))

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "vlm_iris_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "gt_iris", "pred_raw", "pred_iris", "comparable", "correct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-video results saved to {csv_path}")

    if comparable_rows and sklearn_available:
        cm_path = os.path.join(output_dir, "confusion_matrix.csv")
        with open(cm_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + labels)
            for i, label in enumerate(labels):
                w.writerow([label] + list(cm[i]))
        print(f"Confusion matrix saved to {cm_path}")

    summary_path = os.path.join(output_dir, "vlm_iris_summary.txt")
    with open(summary_path, "w") as f:
        f.write("VLM (improved) on IRIS dataset\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"use_iris_classes: {use_iris_classes}\n")
        f.write(f"Total videos: {len(rows)}\n")
        f.write(f"Evaluated: {len(comparable_rows)}\n")
        if comparable_rows:
            f.write(f"Accuracy: {accuracy:.2%}\n")
        if iris_only and not use_iris_classes:
            f.write("\nIRIS-only classes - VLM prediction distribution:\n")
            for gt_name in sorted(by_gt.keys()):
                counts = defaultdict(int)
                for p in by_gt[gt_name]:
                    counts[p] += 1
                f.write(f"  {gt_name}: {dict(counts)}\n")
    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM dynamics classification on IRIS.")
    parser.add_argument("--path", type=str, default="./IRIS", help="Root folder with IRIS .npy files")
    parser.add_argument("--out", type=str, default="./Results/vlm_eval_iris", help="Output directory")
    parser.add_argument("--describe_then_classify", action="store_true", help="Use describe-then-classify (2 API calls) instead of temporal reasoning")
    parser.add_argument("--no_iris_classes", action="store_true", help="Use original 6-class VLM only (report accuracy on 4 comparable + distribution on rest)")
    args = parser.parse_args()
    run_evaluation(
        args.path,
        args.out,
        use_describe_then_classify=args.describe_then_classify,
        use_iris_classes=not args.no_iris_classes,
    )


if __name__ == "__main__":
    main()
