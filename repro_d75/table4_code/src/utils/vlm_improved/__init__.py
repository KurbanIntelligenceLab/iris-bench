"""
Improved VLM dynamics detection: enhanced prompt + more frames for better accuracy.
Use this module to compare against the baseline (src.utils.vlm_dynamics).
"""
from .detector import detect_dynamics_from_npy, DYNAMICS_CHOICES

__all__ = ["detect_dynamics_from_npy", "DYNAMICS_CHOICES"]
