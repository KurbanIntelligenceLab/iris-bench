"""
Evaluate IMPROVED VLM dynamics detection (enhanced prompt + 5 frames).
Outputs to a separate folder (default ./Results/vlm_eval_improved) so you can compare
with baseline: python scripts/evaluate_vlm_dynamics.py --path ./delfys75 --out ./Results/vlm_eval
             vs python scripts/evaluate_vlm_dynamics_improved.py --path ./delfys75 --out ./Results/vlm_eval_improved
Usage:
  python scripts/evaluate_vlm_dynamics_improved.py --path ./delfys75 [--out ./Results/vlm_eval_improved]
"""

import os
import sys
import argparse
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_gt_dynamics_from_path(path: str) -> str | None:
    keywords = [
        "pendulum", "sliding_block", "bouncing_ball", "dropped_ball",
        "led", "free_fall", "torricelli",
    ]
    normalized = path.replace(" ", "").lower()
    for kw in keywords:
        if kw in normalized:
            return kw
    return None


def normalize_gt_for_vlm(gt: str) -> str:
    if gt == "bouncing_ball":
        return "dropped_ball"
    return gt


def run_evaluation(data_path: str, output_dir: str, use_describe_then_classify: bool = False) -> None:
    from src.utils.vlm_improved import detect_dynamics_from_npy

    rows = []
    for root, _, files in os.walk(data_path):
        for f in files:
            if not f.endswith(".npy"):
                continue
            file_path = os.path.join(root, f)
            gt = get_gt_dynamics_from_path(file_path)
            if gt is None:
                continue
            gt_norm = normalize_gt_for_vlm(gt)
            pred = detect_dynamics_from_npy(
                file_path,
                use_temporal_reasoning=not use_describe_then_classify,
                use_describe_then_classify=use_describe_then_classify,
            )
            if pred is None:
                pred = ""
            correct = (pred == gt_norm)
            rows.append({"path": file_path, "gt": gt_norm, "pred": pred, "correct": correct})
            print(f"  {os.path.basename(root)}: gt={gt_norm} pred={pred} correct={correct}")

    if not rows:
        print("No .npy files with known dynamics found.")
        return

    correct_count = sum(r["correct"] for r in rows)
    accuracy = correct_count / len(rows)
    gt_all = [r["gt"] for r in rows]
    pred_all = [r["pred"] for r in rows]
    classes = sorted(set(gt_all))

    mode = "describe-then-classify (2 calls)" if use_describe_then_classify else "temporal reasoning (1 call)"
    print(f"\n--- VLM IMPROVED dynamics evaluation [{mode}] ---")
    print(f"Total videos: {len(rows)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nPer-class accuracy (GT class -> accuracy):")
    for c in classes:
        indices = [i for i, g in enumerate(gt_all) if g == c]
        if not indices:
            continue
        correct_c = sum(1 for i in indices if pred_all[i] == c)
        print(f"  {c}: {correct_c}/{len(indices)} = {correct_c/len(indices):.2%}")

    try:
        from sklearn.metrics import confusion_matrix, classification_report
    except ImportError:
        pass
    else:
        y_true = gt_all
        y_pred = [p if p else "<empty>" for p in pred_all]
        labels = sorted(set(y_true) | set(y_pred))
        if "<empty>" in labels:
            labels.remove("<empty>")
            labels.append("<empty>")
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print("\nConfusion matrix (rows=GT, cols=pred):")
        print("Labels:", labels)
        print(cm)
        try:
            print(classification_report(y_true, y_pred, labels=[l for l in labels if l != "<empty>"], zero_division=0))
        except TypeError:
            print(classification_report(y_true, y_pred, labels=[l for l in labels if l != "<empty>"]))

        os.makedirs(output_dir, exist_ok=True)
        cm_path = os.path.join(output_dir, "confusion_matrix.csv")
        with open(cm_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + labels)
            for i, label in enumerate(labels):
                w.writerow([label] + list(cm[i]))
        print(f"Confusion matrix saved to {cm_path}")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "vlm_dynamics_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "gt", "pred", "correct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-video results saved to {csv_path}")

    summary_path = os.path.join(output_dir, "vlm_eval_summary.txt")
    per_class = {}
    for c in classes:
        n = sum(1 for g in gt_all if g == c)
        correct_c = sum(1 for i, g in enumerate(gt_all) if g == c and pred_all[i] == c)
        per_class[c] = correct_c / n if n else 0.0
    with open(summary_path, "w") as f:
        f.write("VLM IMPROVED dynamics evaluation (enhanced prompt + 5 frames)\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Total videos: {len(rows)}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Per-class accuracy: {per_class}\n")
    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate IMPROVED VLM dynamics (prompt + 5 frames).")
    parser.add_argument("--path", type=str, required=True, help="Root folder with .npy files (e.g. ./delfys75)")
    parser.add_argument("--out", type=str, default="./Results/vlm_eval_improved", help="Output directory (default: vlm_eval_improved for comparison)")
    parser.add_argument("--describe_then_classify", action="store_true", help="Use two-step reasoning: describe motion then classify from text (2 API calls per video)")
    args = parser.parse_args()
    run_evaluation(args.path, args.out, use_describe_then_classify=args.describe_then_classify)


if __name__ == "__main__":
    main()
