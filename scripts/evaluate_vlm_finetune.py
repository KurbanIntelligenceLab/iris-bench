"""
Evaluate fine-tuned Qwen2-VL on Delfys75.
Two modes: (1) LoRA adapter: --adapter path  (2) Frozen VLM + classifier head: --checkpoint path

  python scripts/evaluate_vlm_finetune.py --adapter Results/vlm_finetune/adapter --path ./delfys75 --out Results/vlm_finetune_eval
  python scripts/evaluate_vlm_finetune.py --checkpoint Results/vlm_finetune_classifier/classifier.pt --path ./delfys75 --out Results/vlm_classifier_eval
"""

import argparse
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.vlm_finetune import (
    get_gt_dynamics_from_path,
    load_finetuned_vlm,
    predict_dynamics_from_npy,
    DYNAMICS_CLASSES,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter dir (train_vlm_dynamics.py)")
    ap.add_argument("--checkpoint", type=str, default=None, help="Path to classifier.pt (train_vlm_dynamics_classifier.py); uses frozen VLM + head")
    ap.add_argument("--path", type=str, default="./delfys75")
    ap.add_argument("--out", type=str, default="Results/vlm_finetune_eval")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--num_frames", type=int, default=1)
    ap.add_argument("--no_scoring", action="store_true", help="[LoRA only] Use free generation instead of 6-way scoring")
    ap.add_argument("--verbose", type=int, default=0, metavar="N", help="Print per-class scores for the first N samples (0=off)")
    args = ap.parse_args()

    if args.checkpoint:
        from src.utils.vlm_finetune.vlm_classifier import load_vlm_classifier, predict_dynamics_with_classifier
        print("Loading frozen VLM + classifier head...")
        model, processor, classifier, dev, pool_last_k = load_vlm_classifier(args.checkpoint, model_name=args.model_name)
        use_classifier = True
    elif args.adapter:
        print("Loading model and LoRA adapter...")
        model, processor = load_finetuned_vlm(args.adapter, model_name=args.model_name)
        use_classifier = False
    else:
        print("Provide either --adapter or --checkpoint")
        return

    os.makedirs(args.out, exist_ok=True)
    rows = []
    verbose_count = 0
    for root, _, files in os.walk(args.path):
        for f in files:
            if not f.endswith(".npy"):
                continue
            file_path = os.path.join(root, f)
            gt = get_gt_dynamics_from_path(file_path)
            if gt is None:
                continue
            if use_classifier:
                pred = predict_dynamics_with_classifier(
                    file_path, model, processor, classifier, dev,
                    num_frames=args.num_frames, pool_last_k=pool_last_k,
                )
            else:
                do_verbose = args.verbose > 0 and verbose_count < args.verbose
                if do_verbose and not args.no_scoring:
                    pred, all_scores = predict_dynamics_from_npy(
                        file_path,
                        model=model,
                        processor=processor,
                        num_frames=args.num_frames,
                        use_scoring=True,
                        return_all_scores=True,
                    )
                    print(f"\n[verbose] {file_path}")
                    print(f"  GT: {gt}  pred: {pred}")
                    for c, s in sorted(all_scores.items(), key=lambda x: x[1]):
                        print(f"    {c}: {s:.4f}")
                    verbose_count += 1
                else:
                    pred = predict_dynamics_from_npy(
                        file_path,
                        model=model,
                        processor=processor,
                        num_frames=args.num_frames,
                        use_scoring=not args.no_scoring,
                    )
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

    print("\n--- VLM dynamics evaluation ---")
    print(f"Mode: {'classifier (frozen VLM + head)' if use_classifier else 'LoRA adapter'}")
    print(f"Checkpoint/Adapter: {args.checkpoint or args.adapter}")
    print(f"Total videos: {len(rows)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nPer-class (GT) counts and correct:")
    for c in classes:
        n = sum(1 for r in rows if r["gt"] == c)
        ok = sum(1 for r in rows if r["gt"] == c and r["correct"])
        print(f"  {c}: {ok}/{n}")

    out_csv = os.path.join(args.out, "vlm_finetune_eval.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "gt", "pred", "correct"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv}")

    with open(os.path.join(args.out, "vlm_eval_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Checkpoint/Adapter: {args.checkpoint or args.adapter}\n")
        f.write(f"Total: {len(rows)}\n")
        f.write(f"Correct: {correct_count}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
    print(f"Summary: {args.out}/vlm_eval_summary.txt")


if __name__ == "__main__":
    main()
