"""
Train a classifier head on top of frozen Qwen2-VL features (last hidden state after image+prompt).
The VLM is never updated; only the head is trained. Features depend on the image.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, random_split
from collections import Counter
from torch.utils.data import WeightedRandomSampler

from src.utils.vlm_finetune import VLMDynamicsDataset, DYNAMICS_CLASSES
from src.utils.vlm_finetune.vlm_classifier import (
    load_frozen_vlm_and_processor,
    extract_features,
    DynamicsClassifierHead,
    NUM_CLASSES,
)


def main():
    ap = argparse.ArgumentParser(description="Train classifier head on frozen VLM features")
    ap.add_argument("--path", type=str, default="./delfys75")
    ap.add_argument("--out", type=str, default="Results/vlm_finetune_classifier")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--num_frames", type=int, default=1)
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--linear", action="store_true", help="Use linear head only (default: MLP)")
    ap.add_argument("--balanced", action="store_true", default=True, help="Class-balanced sampling")
    ap.add_argument("--class_weights", action="store_true", default=True, help="Weight CE loss by 1/count(class) to avoid collapse")
    ap.add_argument("--label_smoothing", type=float, default=0.1, help="CE label smoothing (0=off)")
    ap.add_argument("--pool_last_k", type=int, default=4, help="Mean-pool last K hidden states (1=last only)")
    ap.add_argument("--save_best", action="store_true", default=True, help="Save checkpoint with best val accuracy (not last)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out, exist_ok=True)

    dataset = VLMDynamicsDataset(args.path, num_frames=args.num_frames)
    n = len(dataset)
    if n == 0:
        print(f"No .npy with known dynamics under {args.path}")
        return
    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    class_to_idx = {c: i for i, c in enumerate(DYNAMICS_CLASSES)}
    train_classes = [train_ds.dataset.samples[train_ds.indices[i]][1] for i in range(len(train_ds))]
    train_class_counts = Counter(train_classes)
    if args.balanced:
        weights = [1.0 / train_class_counts[c] for c in train_classes]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds))
    else:
        sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=lambda x: x,
    )

    print("Loading frozen VLM (no LoRA)...")
    model, processor = load_frozen_vlm_and_processor(args.model_name)
    dev = next(model.parameters()).device
    # Qwen2-VL: hidden_size is under text_config (e.g. 1536 for 2B)
    if hasattr(model.config, "text_config") and model.config.text_config is not None:
        hidden_size = model.config.text_config.hidden_size
    else:
        hidden_size = getattr(model.config, "hidden_size", 1536)
    use_mlp = not args.linear
    classifier = DynamicsClassifierHead(hidden_size, num_classes=NUM_CLASSES, use_mlp=use_mlp).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr)
    # Class weights: inverse frequency so head can't collapse to one class
    class_weights = None
    if args.class_weights:
        counts = [train_class_counts.get(c, 1) for c in DYNAMICS_CLASSES]
        class_weights = torch.tensor([1.0 / max(1, k) for k in counts], dtype=torch.float32, device=device)
        print(f"Class weights (1/count): {[f'{w:.3f}' for w in class_weights.tolist()]}")

    best_val_acc = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(args.epochs):
        classifier.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for batch in train_loader:
            features_list = []
            labels_list = []
            for content, answer in batch:
                images = [c["image"] for c in content if c.get("type") == "image"]
                feats = extract_features(model, processor, images, device=dev, pool_last_k=args.pool_last_k)
                features_list.append(feats.cpu())
                labels_list.append(class_to_idx[answer])
            features = torch.cat(features_list, dim=0).to(device)
            labels = torch.tensor(labels_list, dtype=torch.long, device=device)
            logits = classifier(features)
            loss = torch.nn.functional.cross_entropy(
                logits, labels, weight=class_weights, label_smoothing=args.label_smoothing
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n += labels.size(0)
        train_acc = total_correct / total_n if total_n else 0

        # Validation
        classifier.eval()
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for i in range(len(val_ds)):
                content, answer = val_ds[i]
                images = [c["image"] for c in content if c.get("type") == "image"]
                feats = extract_features(model, processor, images, device=dev, pool_last_k=args.pool_last_k).to(device)
                logits = classifier(feats)
                pred = logits.argmax(1).item()
                if pred == class_to_idx[answer]:
                    val_correct += 1
                val_n += 1
        val_acc = val_correct / val_n if val_n else 0
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
            best_epoch = epoch + 1

        print(f"Epoch {epoch+1}/{args.epochs}  loss={total_loss/total_n:.4f}  train_acc={train_acc:.2%}  val_acc={val_acc:.2%}  best_val={best_val_acc:.2%} (ep{best_epoch})")

    print("Saving classifier...")
    state_to_save = best_state if (args.save_best and best_state is not None) else classifier.state_dict()
    ckpt = {
        "classifier_state_dict": state_to_save,
        "hidden_size": hidden_size,
        "use_mlp": use_mlp,
        "pool_last_k": args.pool_last_k,
    }
    torch.save(ckpt, os.path.join(args.out, "classifier.pt"))
    if args.save_best and best_epoch >= 0:
        print(f"Saved best checkpoint (val_acc={best_val_acc:.2%} at epoch {best_epoch}) to {args.out}/classifier.pt")
    else:
        print(f"Saved to {args.out}/classifier.pt")


if __name__ == "__main__":
    main()
