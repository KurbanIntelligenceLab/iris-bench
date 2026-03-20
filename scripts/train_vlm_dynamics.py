"""
Fine-tune Qwen2-VL-2B-Instruct with LoRA for dynamics classification on Delfys75.
Requires: pip install transformers peft accelerate (and recent transformers for Qwen2-VL).
First run downloads the model from Hugging Face (~4GB).

Usage:
  python scripts/train_vlm_dynamics.py --path ./delfys75 --out Results/vlm_finetune
  python scripts/train_vlm_dynamics.py --path ./delfys75 --out Results/vlm_finetune --epochs 5 --batch_size 2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from collections import Counter

from src.utils.vlm_finetune import VLMDynamicsDataset

# Qwen2-VL prompt: one <|vision_start|><|image_pad|><|vision_end|> per image
def _build_prompt_text(content: list, prompt_sentence: str) -> tuple[str, list]:
    images = []
    for c in content:
        if c.get("type") == "image":
            images.append(c["image"])
        elif c.get("type") == "text":
            prompt_sentence = c.get("text", prompt_sentence)
    num_images = len(images)
    image_block = "<|vision_start|><|image_pad|><|vision_end|>\n"
    user_content = image_block * num_images + prompt_sentence
    text = "<|im_start|>user\n" + user_content + "<|im_end|>\n<|im_start|>assistant\n"
    return text, images


def main():
    ap = argparse.ArgumentParser(description="Fine-tune Qwen2-VL with LoRA for dynamics classification")
    ap.add_argument("--path", type=str, default="./delfys75")
    ap.add_argument("--out", type=str, default="Results/vlm_finetune")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1, help="Keep 1 for GPU memory (per-sample forward)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--num_frames", type=int, default=1, help="1 = middle frame only (saves memory), 5 = all five")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balanced", action="store_true", help="Use class-balanced sampling to avoid collapse to one class")
    ap.add_argument("--class_weights", action="store_true", help="Weight loss by 1/count(class) per sample")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA not available; training will be slow.")

    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    except ImportError as e:
        print("Install: pip install transformers peft accelerate")
        raise e
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        print("Install: pip install peft")
        raise

    os.makedirs(args.out, exist_ok=True)

    dataset = VLMDynamicsDataset(args.path, num_frames=args.num_frames)
    n = len(dataset)
    if n == 0:
        print(f"No .npy files with known dynamics under {args.path}")
        return
    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    # Class-balanced sampling: weight each sample by 1/count(class) so all classes are seen equally
    train_class_counts = Counter(
        train_ds.dataset.samples[train_ds.indices[i]][1] for i in range(len(train_ds))
    )
    train_sample_weights = [
        1.0 / train_class_counts[train_ds.dataset.samples[train_ds.indices[i]][1]]
        for i in range(len(train_ds))
    ]
    sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_ds)) if args.balanced else None
    class_weight = {c: 1.0 / train_class_counts[c] for c in train_class_counts} if args.class_weights else None
    if args.balanced:
        print(f"Using class-balanced sampling (counts: {dict(train_class_counts)})")
    if class_weight:
        print(f"Using per-sample class weights (inverse count)")

    print(f"Loading model and processor: {args.model_name} (first run may download ~4GB)")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type == "cpu":
        model = model.to(device)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    eos_id = processor.tokenizer.eos_token_id

    def collate_fn(batch):
        return batch

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        num_steps = 0
        optimizer.zero_grad()
        for batch in train_loader:
            batch_loss = 0.0
            for content, answer in batch:
                text, images = _build_prompt_text(content, VLMDynamicsDataset.PROMPT)
                # Processor returns input_ids that include image tokens
                inputs = processor(
                    text=[text],
                    images=images if images else None,
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                prompt_input_ids = inputs["input_ids"].squeeze(0)
                answer_ids = processor.tokenizer(
                    answer,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids.squeeze(0)
                if eos_id is not None:
                    answer_ids = torch.cat([answer_ids, torch.tensor([eos_id], dtype=answer_ids.dtype)])
                full_input_ids = torch.cat([prompt_input_ids, answer_ids], dim=0)
                labels = torch.full_like(full_input_ids, -100)
                labels[-len(answer_ids) :] = answer_ids

                to_device = lambda x: x.to(device, non_blocking=True) if hasattr(x, "to") else x
                model_inputs = {
                    "input_ids": to_device(full_input_ids.unsqueeze(0)),
                    "labels": to_device(labels.unsqueeze(0)),
                }
                if "pixel_values" in inputs and inputs["pixel_values"] is not None:
                    model_inputs["pixel_values"] = to_device(inputs["pixel_values"])
                if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
                    model_inputs["image_grid_thw"] = to_device(inputs["image_grid_thw"])
                if "attention_mask" in inputs:
                    attn = inputs["attention_mask"].squeeze(0)
                    attn = torch.cat([attn, torch.ones(len(answer_ids), dtype=attn.dtype)], dim=0)
                    model_inputs["attention_mask"] = to_device(attn.unsqueeze(0))

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    out = model(**model_inputs)
                loss = out.loss / len(batch)
                if class_weight is not None:
                    loss = loss * class_weight.get(answer, 1.0)
                batch_loss += loss.item()
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss
            num_steps += 1

        print(f"Epoch {epoch+1}/{args.epochs}  loss={total_loss/num_steps:.4f}")

    model.save_pretrained(os.path.join(args.out, "adapter"))
    processor.save_pretrained(os.path.join(args.out, "adapter"))
    print(f"Saved adapter and processor to {args.out}/adapter")


if __name__ == "__main__":
    main()
