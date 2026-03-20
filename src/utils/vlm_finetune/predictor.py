"""
Inference with a fine-tuned Qwen2-VL adapter for dynamics classification.
Use from run_unified_delfys75 by passing adapter path to get_dynamics(..., vlm_finetune_path=...).
"""

import os
import re

from .dataset import VLMDynamicsDataset, _load_frames_as_pil, DYNAMICS_CLASSES


def _build_prompt_text(num_images: int, prompt_sentence: str) -> str:
    image_block = "<|vision_start|><|image_pad|><|vision_end|>\n"
    user_content = image_block * num_images + prompt_sentence
    return "<|im_start|>user\n" + user_content + "<|im_end|>\n<|im_start|>assistant\n"


def _parse_dynamics_from_output(text: str) -> str | None:
    """Pick first occurrence of a known dynamics keyword (case-insensitive)."""
    text = (text or "").strip().lower()
    for d in DYNAMICS_CLASSES:
        if d in text:
            return d
    if "bouncing_ball" in text:
        return "dropped_ball"
    return None


def load_finetuned_vlm(adapter_path: str, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
    """Load base model + PEFT adapter and processor. Returns (model, processor)."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    import torch

    processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, processor


def predict_dynamics_from_npy(
    npy_path: str,
    model,
    processor,
    num_frames: int = 1,
    max_new_tokens: int = 32,
    use_scoring: bool = True,
    return_all_scores: bool = False,
) -> str | None | tuple[str | None, dict[str, float]]:
    """
    Run VLM on frames from .npy and return dynamics string (e.g. 'pendulum') or None.
    model, processor = load_finetuned_vlm(adapter_path) before calling.
    If use_scoring=True (default), score each of the 6 class names (mean NLL) and pick the best.
    If return_all_scores=True and use_scoring=True, returns (best_class, {class: score}).
    """
    import random
    import torch
    pil_list = _load_frames_as_pil(npy_path, num_frames=num_frames)
    if not pil_list:
        return (None, {}) if return_all_scores else None
    text = _build_prompt_text(len(pil_list), VLMDynamicsDataset.PROMPT)
    eos_id = getattr(processor.tokenizer, "eos_token_id", None)

    if use_scoring:
        # Multiple-choice: for each class, compute mean NLL (model loss is already mean over label tokens)
        device = next(model.parameters()).device
        prompt_inputs = processor(
            text=[text],
            images=pil_list,
            padding=True,
            return_tensors="pt",
        )
        prompt_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in prompt_inputs.items()}
        prompt_ids = prompt_inputs["input_ids"].squeeze(0)
        scores = {}
        with torch.no_grad():
            for cls in DYNAMICS_CLASSES:
                answer_ids = processor.tokenizer(cls, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0).to(device)
                if eos_id is not None:
                    answer_ids = torch.cat([answer_ids, torch.tensor([eos_id], dtype=answer_ids.dtype, device=device)])
                num_answer_tokens = answer_ids.size(0)
                full_ids = torch.cat([prompt_ids, answer_ids], dim=0).unsqueeze(0)
                labels = torch.full_like(full_ids, -100)
                labels[0, -num_answer_tokens:] = answer_ids
                inputs = {"input_ids": full_ids, "labels": labels}
                if "pixel_values" in prompt_inputs and prompt_inputs["pixel_values"] is not None:
                    inputs["pixel_values"] = prompt_inputs["pixel_values"]
                if "image_grid_thw" in prompt_inputs and prompt_inputs["image_grid_thw"] is not None:
                    inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"]
                out = model(**inputs)
                scores[cls] = out.loss.item()
        best_score = min(scores.values())
        # Tie-break: if multiple classes within 1e-5 of best, pick randomly
        candidates = [c for c, s in scores.items() if abs(s - best_score) < 1e-5]
        best_class = random.choice(candidates) if len(candidates) > 1 else candidates[0]
        if return_all_scores:
            return best_class, scores
        return best_class

    # Free generation (can collapse to one class)
    inputs = processor(text=[text], images=pil_list, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        )
    generated_ids = out[:, prompt_len:]
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    answer = decoded[0] if decoded else ""
    return _parse_dynamics_from_output(answer)
