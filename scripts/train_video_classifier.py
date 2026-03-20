"""
Train the small video dynamics classifier (ResNet18 + temporal mean) on Delfys75 or IRIS.
Labels from folder path. Saves best checkpoint and prints train/val accuracy. No API key required.

Usage:
  # Delfys75 (6 classes)
  python scripts/train_video_classifier.py --path ./delfys75 --out Results/video_classifier
  # IRIS (8 classes)
  python scripts/train_video_classifier.py --path ./IRIS --out Results/video_classifier_iris --iris --epochs 25 --batch 8
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.utils.video_classifier import DynamicsDataset, VideoDynamicsClassifier, DYNAMICS_CLASSES, IRIS_DYNAMICS_CLASSES


def main():
    ap = argparse.ArgumentParser(description="Train video dynamics classifier on .npy videos")
    ap.add_argument("--path", type=str, default="./delfys75", help="Root folder containing dynamics subfolders with .npy")
    ap.add_argument("--out", type=str, default="Results/video_classifier", help="Output dir for checkpoint and log")
    ap.add_argument("--iris", action="store_true", help="Use IRIS 8-class dataset (path should be ./IRIS)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of data for validation")
    ap.add_argument("--num_frames", type=int, default=5)
    ap.add_argument("--freeze_epochs", type=int, default=2, help="Epochs with backbone frozen")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    dataset = DynamicsDataset(args.path, num_frames=args.num_frames, use_iris=args.iris)
    n = len(dataset)
    if n == 0:
        print(f"No .npy files with known dynamics under {args.path}. Check path.")
        return
    n_classes = len(IRIS_DYNAMICS_CLASSES) if args.iris else len(DYNAMICS_CLASSES)
    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = VideoDynamicsClassifier(num_classes=n_classes, pretrained=True, freeze_backbone_epochs=args.freeze_epochs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs:
            model.unfreeze_backbone()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_total += y.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_total += y.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()

        train_acc = train_correct / train_total if train_total else 0
        val_acc = val_correct / val_total if val_total else 0
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss/len(train_loader):.4f}  train_acc={train_acc:.2%}  val_acc={val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, os.path.join(args.out, "best.pt"))

    print(f"Best val accuracy: {best_val_acc:.2%}. Checkpoint: {args.out}/best.pt")
    print("To use: from src.utils.video_classifier import load_classifier, predict_dynamics_from_npy; model = load_classifier('Results/video_classifier/best.pt'); predict_dynamics_from_npy('path/to/video.npy', model=model)")


if __name__ == "__main__":
    main()
