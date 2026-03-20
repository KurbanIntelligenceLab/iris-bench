"""
VLM fine-tuning for dynamics classification (Qwen2-VL-2B + LoRA).
Train with scripts/train_vlm_dynamics.py. Infer with predictor.load_finetuned_vlm + predict_dynamics_from_npy.
Requires: transformers, peft, accelerate.
"""

from .dataset import VLMDynamicsDataset, get_gt_dynamics_from_path, DYNAMICS_CLASSES
from .predictor import load_finetuned_vlm, predict_dynamics_from_npy

__all__ = [
    "VLMDynamicsDataset",
    "get_gt_dynamics_from_path",
    "DYNAMICS_CLASSES",
    "load_finetuned_vlm",
    "predict_dynamics_from_npy",
]
