"""
Load trained video classifier and predict dynamics from .npy path (same interface as VLM detect_dynamics_from_npy).
"""

import os
import torch
from .dataset import DYNAMICS_CLASSES, _load_frames_from_npy
from .model import VideoDynamicsClassifier


def load_classifier(
    checkpoint_path: str,
    device: str | torch.device | None = None,
    num_classes: int = 6,
) -> VideoDynamicsClassifier:
    """Load VideoDynamicsClassifier from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    model = VideoDynamicsClassifier(num_classes=num_classes, pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_dynamics_from_npy(
    npy_path: str,
    model: VideoDynamicsClassifier | None = None,
    checkpoint_path: str | None = None,
    device: str | torch.device | None = None,
    num_frames: int = 5,
) -> str | None:
    """
    Predict dynamics class for one video. Returns one of DYNAMICS_CLASSES or None if load fails.
    Either pass a loaded model or checkpoint_path to load from file.
    """
    if model is None and checkpoint_path is None:
        return None
    if model is None:
        model = load_classifier(checkpoint_path, device=device)
    if device is None:
        device = next(model.parameters()).device
    try:
        frames = _load_frames_from_npy(npy_path, num_frames=num_frames, resize_hw=(224, 224))
    except Exception:
        return None
    # (T, 1, H, W) -> (1, T, 3, H, W)
    import numpy as np
    frames = np.repeat(frames, 3, axis=1)
    x = torch.from_numpy(frames).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    pred_idx = logits.argmax(dim=1).item()
    return DYNAMICS_CLASSES[pred_idx]
