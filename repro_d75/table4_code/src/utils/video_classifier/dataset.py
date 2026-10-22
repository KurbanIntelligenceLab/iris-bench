"""
Dataset for video dynamics classification: load .npy files, sample 5 frames, return (frames, label_idx).
Labels are derived from folder path (same as VLM evaluation).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Same order as vlm_improved.DYNAMICS_CHOICES for compatibility
DYNAMICS_CLASSES = [
    "dropped_ball",
    "free_fall",
    "led",
    "pendulum",
    "sliding_block",
    "torricelli",
]

CLASS_TO_IDX = {c: i for i, c in enumerate(DYNAMICS_CLASSES)}


def get_gt_dynamics_from_path(path: str) -> str | None:
    """Extract dynamics keyword from file path (folder structure)."""
    keywords = [
        "pendulum", "sliding_block", "bouncing_ball", "dropped_ball",
        "led", "free_fall", "torricelli",
    ]
    normalized = path.replace(" ", "").lower().replace("\\", "/")
    for kw in keywords:
        if kw in normalized:
            return "dropped_ball" if kw == "bouncing_ball" else kw
    return None


def _load_frames_from_npy(npy_path: str, num_frames: int = 5, resize_hw: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load .npy, sample num_frames (start, 25%, 50%, 75%, end), return (num_frames, 1, H, W) float [0,1]."""
    data = np.load(npy_path, allow_pickle=True)
    if getattr(data, "ndim", 0) == 0 and getattr(data.dtype, "name", "") == "object":
        data = data.item()
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.ndim == 5:
        n_samples, nf, _, h, w = data.shape
        indices = np.linspace(0, nf - 1, num=min(num_frames, nf), dtype=int)
        frames = np.stack([data[0, i, 0, :, :] for i in indices], axis=0)  # (T, H, W)
    elif data.ndim == 4:
        nf, _, h, w = data.shape
        indices = np.linspace(0, nf - 1, num=min(num_frames, nf), dtype=int)
        frames = np.stack([data[i, 0, :, :] for i in indices], axis=0)  # (T, H, W)
    else:
        raise ValueError(f"Unexpected .npy shape {data.shape}")

    frames = np.clip(frames.astype(np.float32), 0.0, 255.0) / 255.0 if frames.max() > 1.0 else frames.astype(np.float32)
    # (T, H, W) -> (T, 1, H, W)
    frames = np.expand_dims(frames, axis=1)
    # Resize to (T, 1, resize_hw[0], resize_hw[1])
    if (frames.shape[2], frames.shape[3]) != resize_hw:
        import cv2
        out = []
        for t in range(frames.shape[0]):
            img = (frames[t, 0] * 255).astype(np.uint8)
            img = cv2.resize(img, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_LINEAR)
            out.append(img.astype(np.float32) / 255.0)
        frames = np.stack(out, axis=0)[:, np.newaxis, :, :]
    return frames


class DynamicsDataset(Dataset):
    """Dataset of (video_path, dynamics_label_idx). Loads 5 frames on the fly."""

    def __init__(
        self,
        root_dir: str,
        num_frames: int = 5,
        resize_hw: tuple[int, int] = (224, 224),
        transform=None,
    ):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.resize_hw = resize_hw
        self.transform = transform
        self.samples = []  # list of (npy_path, label_idx)
        for root, _, files in os.walk(root_dir):
            for f in files:
                if not f.endswith(".npy"):
                    continue
                path = os.path.join(root, f)
                gt = get_gt_dynamics_from_path(path)
                if gt is None:
                    continue
                idx = CLASS_TO_IDX.get(gt)
                if idx is None:
                    continue
                self.samples.append((path, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[i]
        frames = _load_frames_from_npy(path, num_frames=self.num_frames, resize_hw=self.resize_hw)
        # (T, 1, H, W) -> (T, 3, H, W) by repeating channel for ResNet
        frames = np.repeat(frames, 3, axis=1)
        x = torch.from_numpy(frames).float()  # (T, 3, H, W)
        if self.transform:
            x = self.transform(x)
        return x, label
