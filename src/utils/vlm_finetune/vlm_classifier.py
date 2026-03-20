"""
VLM-as-feature-extractor + classifier head: freeze Qwen2-VL, take last hidden state
after image+prompt, train a small head to 6 dynamics classes. Uses the image.
"""

import os
import torch
import torch.nn as nn

from .dataset import VLMDynamicsDataset, _load_frames_as_pil, DYNAMICS_CLASSES

NUM_CLASSES = len(DYNAMICS_CLASSES)


def _build_prompt_text(num_images: int, prompt_sentence: str) -> str:
    image_block = "<|vision_start|><|image_pad|><|vision_end|>\n"
    user_content = image_block * num_images + prompt_sentence
    return "<|im_start|>user\n" + user_content + "<|im_end|>\n<|im_start|>assistant\n"


class DynamicsClassifierHead(nn.Module):
    """Maps VLM hidden state to 6 dynamics logits. Optional 1 hidden layer."""
    def __init__(self, hidden_size: int, num_classes: int = NUM_CLASSES, use_mlp: bool = True):
        super().__init__()
        if use_mlp:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, num_classes),
            )
        else:
            self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def load_frozen_vlm_and_processor(model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
    """Load VLM and processor, set model to eval and freeze."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, processor


def extract_features(
    model,
    processor,
    images: list,
    prompt_text: str = VLMDynamicsDataset.PROMPT,
    device: torch.device | None = None,
    pool_last_k: int = 1,
) -> torch.Tensor:
    """
    Run frozen VLM on image(s) + prompt (no answer). Return feature from last position(s).
    If pool_last_k > 1, mean-pool over the last K hidden states (more image context).
    images: list of PIL; prompt_text: str. Returns (1, hidden_size) on device.
    """
    if device is None:
        device = next(model.parameters()).device
    text = _build_prompt_text(len(images), prompt_text)
    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    last_hidden = getattr(out, "last_hidden_state", None) or (out.hidden_states[-1] if out.hidden_states else None)
    if last_hidden is None:
        raise RuntimeError("Model did not return last_hidden_state or hidden_states")
    # (batch, seq_len, hidden_size): take last K positions and mean-pool
    seq_len = last_hidden.size(1)
    k = min(pool_last_k, seq_len)
    features = last_hidden[:, -k:, :].mean(dim=1)  # (batch, hidden_size)
    return features.float()  # classifier in float32


def load_vlm_classifier(
    checkpoint_path: str,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
) -> tuple:
    """Load frozen VLM + trained classifier head. Returns (model, processor, classifier, device, pool_last_k)."""
    model, processor = load_frozen_vlm_and_processor(model_name)
    device = next(model.parameters()).device
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    hidden_size = ckpt.get("hidden_size")
    if hidden_size is None:
        if hasattr(model.config, "text_config") and model.config.text_config is not None:
            hidden_size = model.config.text_config.hidden_size
        else:
            hidden_size = getattr(model.config, "hidden_size", 1536)
    use_mlp = ckpt.get("use_mlp", True)
    classifier = DynamicsClassifierHead(hidden_size, num_classes=NUM_CLASSES, use_mlp=use_mlp)
    classifier.load_state_dict(ckpt["classifier_state_dict"])
    classifier = classifier.to(device).eval()
    pool_last_k = ckpt.get("pool_last_k", 1)
    return model, processor, classifier, device, pool_last_k


def predict_dynamics_with_classifier(
    npy_path: str,
    model,
    processor,
    classifier,
    device: torch.device,
    num_frames: int = 1,
    pool_last_k: int = 1,
) -> str:
    """Extract features from VLM for video frames, run classifier, return class name."""
    pil_list = _load_frames_as_pil(npy_path, num_frames=num_frames)
    if not pil_list:
        return DYNAMICS_CLASSES[0]  # fallback
    features = extract_features(model, processor, pil_list, device=device, pool_last_k=pool_last_k)
    with torch.no_grad():
        logits = classifier(features)
    pred_idx = logits[0].argmax().item()
    return DYNAMICS_CLASSES[pred_idx]
