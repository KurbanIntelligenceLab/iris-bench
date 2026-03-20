# CNN classifier trained on IRIS (8 classes)

Train the 8-class video dynamics classifier on IRIS:

```bash
cd /workspace/learning_from_physics
python3 scripts/train_video_classifier.py --path ./IRIS --out Results/video_classifier_iris --iris --epochs 25 --batch 8 --val_ratio 0.2
```

After training, `best.pt` will appear here. Then evaluate on full IRIS:

```bash
python3 scripts/evaluate_video_classifier.py --path ./IRIS --checkpoint Results/video_classifier_iris/best.pt --out Results/video_classifier_eval_iris --iris
```

Requires: `torch`, `torchvision`. Resize uses `cv2` if available, else `torch.nn.functional.interpolate`.
