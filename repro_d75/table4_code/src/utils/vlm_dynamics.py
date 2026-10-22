"""
VLM-based dynamics detection using OpenRouter (https://openrouter.ai/).
Uses a vision-language model to classify physical dynamics from sample video frames.
Set OPENROUTER_API_KEY in the environment; no key is stored in code.
"""

import os
import base64
import re
import numpy as np

# Supported dynamics (must match main.py physics models)
DYNAMICS_CHOICES = [
    "pendulum",
    "sliding_block",
    "dropped_ball",
    "free_fall",
    "led",
    "torricelli",
]

# OpenRouter vision model: strong at image understanding
DEFAULT_MODEL = "openai/gpt-4o"


def _frames_from_npy_to_base64(npy_path: str, num_frames: int = 3) -> list[str]:
    """Load .npy, take a few sample frames, return base64 data URLs (PNG)."""
    data = np.load(npy_path, allow_pickle=True)
    if getattr(data, "ndim", 0) == 0 and getattr(data.dtype, "name", "") == "object":
        data = data.item()
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    # Expect shape (N, nf, 1, H, W) e.g. (samples, 10, 1, 56, 100)
    if data.ndim == 5:
        n_samples, nf, _, h, w = data.shape
        # Use first sample, frames at start, mid, end
        indices = np.linspace(0, nf - 1, num=min(num_frames, nf), dtype=int)
        frames = [data[0, i, 0, :, :] for i in indices]
    elif data.ndim == 4:
        nf, _, h, w = data.shape
        indices = np.linspace(0, nf - 1, num=min(num_frames, nf), dtype=int)
        frames = [data[i, 0, :, :] for i in indices]
    else:
        raise ValueError(f"Unexpected .npy shape {data.shape}")

    out = []
    for i, frame in enumerate(frames):
        # Normalize to [0, 255] if in [0, 1]
        if frame.max() <= 1.0:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        # PNG bytes (cv2 is used elsewhere in the project)
        import cv2
        _, buf = cv2.imencode(".png", frame)
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        out.append(f"data:image/png;base64,{b64}")
    return out


def detect_dynamics_from_npy(
    npy_path: str,
    *,
    model: str = DEFAULT_MODEL,
    num_frames: int = 3,
    api_key: str | None = None,
) -> str | None:
    """
    Use a VLM to classify the physical dynamics shown in the video stored as .npy.

    Parameters
    ----------
    npy_path : str
        Path to the .npy file (shape like N, 10, 1, H, W).
    model : str
        OpenRouter model id (vision-capable). Default: openai/gpt-4o.
    num_frames : int
        Number of sample frames to send (start/mid/end).
    api_key : str, optional
        OpenRouter API key. If None, uses env OPENROUTER_API_KEY.

    Returns
    -------
    str or None
        One of pendulum, sliding_block, dropped_ball, free_fall, led, torricelli,
        or None if the VLM could not be queried or response was invalid.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("VLM dynamics: OPENROUTER_API_KEY not set; skipping VLM.")
        return None

    try:
        image_urls = _frames_from_npy_to_base64(npy_path, num_frames=num_frames)
    except Exception as e:
        print(f"VLM dynamics: failed to load frames from {npy_path}: {e}")
        return None

    prompt = (
        "These images are sample frames from a single physics experiment video. "
        "Classify the type of physical dynamics into exactly one of: "
        + ", ".join(DYNAMICS_CHOICES)
        + ". "
        "Reply with only one word: the dynamics type (e.g. pendulum or free_fall). "
        "No other text."
    )

    content = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    try:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=32,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"VLM dynamics: API error: {e}")
        return None

    # Parse: take first word, lowercase, allow underscores
    raw_lower = raw.lower().strip()
    # Remove trailing period or extra words
    first_word = raw_lower.split()[0] if raw_lower else ""
    first_word = re.sub(r"[^a-z_]", "", first_word)
    for choice in DYNAMICS_CHOICES:
        if choice == first_word or choice.replace("_", "") == first_word.replace("_", ""):
            return choice
    # Fuzzy: allow "dropped ball" -> dropped_ball
    if "dropped" in raw_lower and "ball" in raw_lower:
        return "dropped_ball"
    if "free" in raw_lower and "fall" in raw_lower:
        return "free_fall"
    if "sliding" in raw_lower and "block" in raw_lower:
        return "sliding_block"
    if "pendulum" in raw_lower:
        return "pendulum"
    if "led" in raw_lower or "light" in raw_lower:
        return "led"
    if "torricelli" in raw_lower or "water" in raw_lower and "drain" in raw_lower:
        return "torricelli"
    print(f"VLM dynamics: could not parse dynamics from: {raw!r}")
    return None
