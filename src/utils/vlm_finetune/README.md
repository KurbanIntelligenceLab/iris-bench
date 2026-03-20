# VLM fine-tuning for dynamics classification

Two approaches:

1. **LoRA** – Fine-tune Qwen2-VL with LoRA (see below). Often ignores the image (scores identical for every video).
2. **Frozen VLM + classifier head** – Use the VLM as a **frozen feature extractor**: take the last hidden state after image+prompt, train a small MLP/linear head to 6 classes. **Uses the image** and avoids collapse.

## Requirements

```bash
pip install transformers peft accelerate
```

Use a recent `transformers` (≥4.45 or install from main if needed for Qwen2-VL). First run will download the base model (~4GB).

## Train

**Option A – Frozen VLM + classifier head (recommended)**  
VLM stays frozen; only a small head is trained on the last-hidden-state features.

```bash
python scripts/train_vlm_dynamics_classifier.py --path ./delfys75 --out Results/vlm_finetune_classifier
```

Options: `--epochs`, `--batch_size`, `--num_frames`, `--lr`, `--linear` (linear head only), `--balanced` (default on).

**Option B – LoRA** (often image-independent)

```bash
python scripts/train_vlm_dynamics.py --path ./delfys75 --out Results/vlm_finetune --balanced --class_weights
```

Options: `--epochs`, `--batch_size` (keep 1), `--num_frames`, `--lr`, `--lora_r`, `--lora_alpha`.

## Evaluate

**Classifier (frozen VLM + head):**
```bash
python scripts/evaluate_vlm_finetune.py --checkpoint Results/vlm_finetune_classifier/classifier.pt --path ./delfys75 --out Results/vlm_classifier_eval
```

**LoRA adapter:**  
Uses 6-way scoring by default (multiple-choice over class names).
```bash
python scripts/evaluate_vlm_finetune.py --adapter Results/vlm_finetune/adapter --path ./delfys75 --out Results/vlm_finetune_eval
```
Use `--no_scoring` for free-text generation (can collapse to one class).

## Use in pipeline

From code: `load_finetuned_vlm(adapter_path)` then `predict_dynamics_from_npy(npy_path, model=model, processor=processor)`.

## Known limitation (LoRA only): image-independent scores

With **LoRA-only** training, the model often **does not condition on the image** (scores identical for every video; run with `--verbose 3`). The **frozen VLM + classifier head** (Option A) avoids this by using the VLM only as a feature extractor and training a head on those features, so predictions depend on the image.
