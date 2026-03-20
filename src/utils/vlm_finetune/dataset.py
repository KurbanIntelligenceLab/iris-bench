"""
Dataset for VLM fine-tuning: load .npy, sample 1 or 5 frames as PIL Images, return (content for chat, answer string).
Uses single middle frame by default to save memory; optional num_frames=5 for multi-image.
"""

import os
import numpy as np
from PIL import Image

DYNAMICS_CLASSES = [
    "dropped_ball",
    "free_fall",
    "led",
    "pendulum",
    "sliding_block",
    "torricelli",
]


def get_gt_dynamics_from_path(path: str) -> str | None:
    """Extract dynamics keyword from file path."""
    keywords = [
        "pendulum", "sliding_block", "bouncing_ball", "dropped_ball",
        "led", "free_fall", "torricelli",
    ]
    normalized = path.replace(" ", "").lower().replace("\\", "/")
    for kw in keywords:
        if kw in normalized:
            return "dropped_ball" if kw == "bouncing_ball" else kw
    return None


def _load_frames_as_pil(npy_path: str, num_frames: int = 1, resize_hw: tuple[int, int] = (224, 224)) -> list[Image.Image]:
    """Load .npy, sample num_frames (default 1 = middle), return list of PIL Images (RGB)."""
    data = np.load(npy_path, allow_pickle=True)
    if getattr(data, "ndim", 0) == 0 and getattr(data.dtype, "name", "") == "object":
        data = data.item()
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.ndim == 5:
        n_samples, nf, _, h, w = data.shape
        indices = np.linspace(0, nf - 1, num=min(num_frames, nf), dtype=int) if num_frames > 1 else [nf // 2]
        frames = [data[0, i, 0, :, :] for i in indices]
    elif data.ndim == 4:
        nf, _, h, w = data.shape
        indices = np.linspace(0, nf - 1, num=min(num_frames, nf), dtype=int) if num_frames > 1 else [nf // 2]
        frames = [data[i, 0, :, :] for i in indices]
    else:
        raise ValueError(f"Unexpected .npy shape {data.shape}")

    import cv2
    out = []
    for frame in frames:
        frame = np.clip(frame.astype(np.float32), 0.0, 255.0) if frame.max() <= 1.0 else frame.astype(np.float32) * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        if (frame.shape[0], frame.shape[1]) != resize_hw:
            frame = cv2.resize(frame, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_LINEAR)
        pil = Image.fromarray(frame).convert("RGB")
        out.append(pil)
    return out


class VLMDynamicsDataset:
    """
    Returns (messages_content, answer_string).
    messages_content: list for one sample, e.g. [{"type": "image", "image": PIL}, {"type": "text", "text": "..."}]
    So the trainer can call processor.apply_chat_template(messages, ...) and tokenize the answer for labels.
    """

    PROMPT = (
        "These frames are from a physics experiment video. "
        "What is the dynamics type? Answer with exactly one word: dropped_ball, free_fall, led, pendulum, sliding_block, or torricelli."
    )

    def __init__(self, root_dir: str, num_frames: int = 1, resize_hw: tuple[int, int] = (224, 224)):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.resize_hw = resize_hw
        self.samples = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if not f.endswith(".npy"):
                    continue
                path = os.path.join(root, f)
                gt = get_gt_dynamics_from_path(path)
                if gt is None:
                    continue
                self.samples.append((path, gt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[list, str]:
        path, answer = self.samples[i]
        pil_list = _load_frames_as_pil(path, num_frames=self.num_frames, resize_hw=self.resize_hw)
        content = []
        for pil in pil_list:
            content.append({"type": "image", "image": pil})
        content.append({"type": "text", "text": self.PROMPT})
        return content, answer
