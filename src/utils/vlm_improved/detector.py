"""
Improved VLM dynamics detector: (1) enhanced prompt with per-class descriptions,
(2) 5 frames (start, 25%, 50%, 75%, end) for better temporal signal.
Set OPENROUTER_API_KEY in the environment.
"""

import os
import base64
import re
import numpy as np

DYNAMICS_CHOICES = [
    "pendulum",
    "sliding_block",
    "dropped_ball",
    "free_fall",
    "led",
    "torricelli",
]

# IRIS-extended set: adds rotation and multi-body so the VLM can distinguish them from single pendulum/sliding
IRIS_DYNAMICS_CHOICES = [
    "pendulum",
    "sliding_block",
    "dropped_ball",
    "free_fall",
    "led",
    "torricelli",
    "rotation",
    "hitting_cones",
    "two_moving_pendulums",
    "two_moving_pendulum_one_static",
]

# Per-class one-line descriptions to reduce confusion (especially free_fall vs dropped_ball vs pendulum)
DYNAMICS_DESCRIPTIONS = """
- pendulum: A SINGLE object or ball swinging back and forth on a string or rod (oscillating arc). Camera usually side-on.
- sliding_block: A block or cone sliding down a ramp or inclined plane in one direction (no swing).
- dropped_ball: Camera is to the SIDE of the ball; ball falls, size in frame stays roughly constant; horizon or wall visible to the side.
- free_fall: Camera is ABOVE the ball (top-down or diagonal from above); you look down at the ball; it may shrink as it falls.
- led: The brightness or intensity of a light source changing over time (no object sliding or swinging).
- torricelli: Water draining from a container; water level or a floating object going down.
"""

# IRIS-extended descriptions (add rotation and multi-body)
IRIS_DYNAMICS_DESCRIPTIONS = DYNAMICS_DESCRIPTIONS + """
- rotation: An object rotating or spinning in place (e.g. on a turntable); motion is in-place spin, NOT a pendulum arc.
- hitting_cones: A ball or object colliding with and knocking over other objects (e.g. cones or pins).
- two_moving_pendulums: TWO pendulums; in the first frame BOTH are already moving or both at an angle (released together).
- two_moving_pendulum_one_static: In the first frame ONE bob is perfectly vertical and motionless; the OTHER swings and hits it (one at rest, one moving).
"""

# Temporal reasoning: compare first vs last frame; sharper cues for free_fall vs dropped_ball vs pendulum
TEMPORAL_REASONING_INSTRUCTIONS = """
Step 1 - Temporal change: Look at the FIRST and LAST frame. What changes?
- Ball's SIZE in the frame gets SMALLER over time → camera above ball = free_fall.
- Ball moves DOWN but keeps similar size (side view) → dropped_ball.
- Object swings BACK AND FORTH in an arc (left-right or diagonal) = pendulum.
- Block or cone slides down a ramp in one direction (no swing) = sliding_block.
- Only brightness/glow changes = led. Water level or container draining = torricelli.

Step 2 - Reply with ONLY one word: pendulum, sliding_block, dropped_ball, free_fall, led, or torricelli.
"""

# IRIS: decision order matters. Use camera viewpoint for free_fall vs dropped_ball; strict "one at rest" for one_static.
IRIS_TEMPORAL_REASONING_INSTRUCTIONS = """
Step 1 - BALL FALLING (camera viewpoint is the key):
  - free_fall: You are looking DOWN at the ball (camera above it, like looking at the ground from above). The ball may appear to shrink as it falls, or you see the top of the ball / a downward view.
  - dropped_ball: You are looking from the SIDE at the ball (camera beside it). The ball falls and keeps similar size in the frame. Horizon or wall is visible to the side.
  If the viewpoint is from above / top-down → free_fall. If the viewpoint is from the side → dropped_ball.
Step 2 - SOLID object (cone, block) sliding down a RAMP in one direction → sliding_block. (Sliding = one-way motion along a surface. NOT pendulum; pendulum swings back and forth. torricelli = liquid only.)
Step 3 - LIQUID or water level going down in a container → torricelli.
Step 4 - ONE object swinging back and forth in an arc (single pendulum) → pendulum.
Step 5 - Object SPINNING in place (turntable, no arc) → rotation.
Step 6 - Ball HITTING and scattering other objects (e.g. cones) → hitting_cones.
Step 7 - TWO pendulums (look at the VERY FIRST frame):
  - two_moving_pendulum_one_static: In frame 1, ONE bob is perfectly VERTICAL and STILL (not moving at all); the OTHER bob is at an angle and then swings. One starts at rest, the other hits it.
  - two_moving_pendulums: In frame 1, BOTH bobs are already moving or both are at an angle (released together). No single motionless vertical bob at the start.
Step 8 - Only brightness/glow changing → led.

Reply with ONLY one word: pendulum, sliding_block, dropped_ball, free_fall, led, torricelli, rotation, hitting_cones, two_moving_pendulums, or two_moving_pendulum_one_static.
"""

DEFAULT_MODEL = "openai/gpt-4o"
DEFAULT_NUM_FRAMES = 5
# More frames for IRIS to better capture size change (free_fall) and pendulum type
IRIS_NUM_FRAMES = 7


def _frames_from_npy_to_base64(npy_path: str, num_frames: int = 5) -> list[str]:
    """Load .npy, take num_frames spread across time (start, 25%, 50%, 75%, end), return base64 PNG URLs."""
    data = np.load(npy_path, allow_pickle=True)
    if getattr(data, "ndim", 0) == 0 and getattr(data.dtype, "name", "") == "object":
        data = data.item()
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.ndim == 5:
        n_samples, nf, _, h, w = data.shape
        indices = np.linspace(0, nf - 1, num=min(num_frames, nf), dtype=int)
        frames = [data[0, i, 0, :, :] for i in indices]
    elif data.ndim == 4:
        nf, _, h, w = data.shape
        indices = np.linspace(0, nf - 1, num=min(num_frames, nf), dtype=int)
        frames = [data[i, 0, :, :] for i in indices]
    else:
        raise ValueError(f"Unexpected .npy shape {data.shape}")

    import cv2
    out = []
    for frame in frames:
        if frame.max() <= 1.0:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        _, buf = cv2.imencode(".png", frame)
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        out.append(f"data:image/png;base64,{b64}")
    return out


def _call_vlm(client, model: str, content: list, max_tokens: int = 128) -> str:
    """Single VLM call; content = list of text + image_url items."""
    from openai import OpenAI
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_dynamics_from_response(raw: str, choices: list[str] | None = None) -> str | None:
    """Extract dynamics choice from VLM reply. If choices is None, use DYNAMICS_CHOICES. Checks longer names first."""
    choices = choices or DYNAMICS_CHOICES
    raw_lower = raw.lower().strip()
    # Check longer names first so e.g. two_moving_pendulums matches before pendulum
    choices_sorted = sorted(choices, key=len, reverse=True)
    for choice in choices_sorted:
        if choice in raw_lower or choice.replace("_", " ") in raw_lower or choice.replace("_", "") in raw_lower.replace(" ", ""):
            return choice
    first_word = raw_lower.split()[0] if raw_lower else ""
    first_word = re.sub(r"[^a-z_]", "", first_word)
    for choice in choices:
        if choice == first_word or choice.replace("_", "") == first_word.replace("_", ""):
            return choice
    if "dropped" in raw_lower and "ball" in raw_lower:
        return "dropped_ball"
    if "free" in raw_lower and "fall" in raw_lower:
        return "free_fall"
    if "sliding" in raw_lower and ("block" in raw_lower or "cone" in raw_lower):
        return "sliding_block"
    if "two" in raw_lower and "pendulum" in raw_lower and "static" in raw_lower:
        return "two_moving_pendulum_one_static"
    if "two" in raw_lower and "pendulum" in raw_lower:
        return "two_moving_pendulums"
    if "pendulum" in raw_lower:
        return "pendulum"
    if ("hit" in raw_lower and "cone" in raw_lower) or ("collision" in raw_lower and "cone" in raw_lower):
        return "hitting_cones"
    if "rotat" in raw_lower or ("spin" in raw_lower and "place" in raw_lower):
        return "rotation"
    if "led" in raw_lower or "light" in raw_lower:
        return "led"
    if "torricelli" in raw_lower or ("water" in raw_lower and "drain" in raw_lower):
        return "torricelli"
    return None


def detect_dynamics_from_npy(
    npy_path: str,
    *,
    model: str = DEFAULT_MODEL,
    num_frames: int = DEFAULT_NUM_FRAMES,
    api_key: str | None = None,
    use_temporal_reasoning: bool = True,
    use_describe_then_classify: bool = False,
    use_iris_classes: bool = False,
) -> str | None:
    """
    Improved VLM dynamics classification: enhanced prompt + 5 frames.
    - use_temporal_reasoning=True (default): temporal reasoning (first vs last frame).
    - use_describe_then_classify=True: two-step reasoning (2 API calls).
    - use_iris_classes=True: use IRIS-extended set (rotation, hitting_cones, two_moving_pendulums, two_moving_pendulum_one_static) for better accuracy on IRIS.
    Returns one of DYNAMICS_CHOICES or, if use_iris_classes, one of IRIS_DYNAMICS_CHOICES.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("VLM improved: OPENROUTER_API_KEY not set; skipping.")
        return None

    n_frames = IRIS_NUM_FRAMES if use_iris_classes else num_frames
    try:
        image_urls = _frames_from_npy_to_base64(npy_path, num_frames=n_frames)
    except Exception as e:
        print(f"VLM improved: failed to load frames from {npy_path}: {e}")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    except Exception as e:
        print(f"VLM improved: OpenAI client: {e}")
        return None

    choices = IRIS_DYNAMICS_CHOICES if use_iris_classes else DYNAMICS_CHOICES
    descriptions = IRIS_DYNAMICS_DESCRIPTIONS if use_iris_classes else DYNAMICS_DESCRIPTIONS
    temporal_instructions = IRIS_TEMPORAL_REASONING_INSTRUCTIONS if use_iris_classes else TEMPORAL_REASONING_INSTRUCTIONS
    choices_str = ", ".join(choices)

    if use_describe_then_classify:
        prompt1 = (
            "These images are consecutive frames from a single physics video (first to last in time order). "
            "In 1–2 short sentences, describe only the motion or change you see (e.g. what moves, in which direction, or what changes). "
            "Do not name a dynamics type yet."
        )
        content1 = [{"type": "text", "text": prompt1}]
        for url in image_urls:
            content1.append({"type": "image_url", "image_url": {"url": url}})
        try:
            description = _call_vlm(client, model, content1, max_tokens=80)
        except Exception as e:
            print(f"VLM improved: describe step failed: {e}")
            return None
        if not description:
            return None
        prompt2 = (
            "Given this description of motion from a physics video:\n\"\"\"\n" + description + "\n\"\"\"\n\n"
            "Choose exactly one dynamics type:\n" + descriptions + "\n\n"
            f"Reply with ONLY one word: {choices_str}."
        )
        content2 = [{"type": "text", "text": prompt2}]
        try:
            raw = _call_vlm(client, model, content2, max_tokens=64)
        except Exception as e:
            print(f"VLM improved: classify step failed: {e}")
            return None
        out = _parse_dynamics_from_response(raw, choices=choices)
        if out is None:
            print(f"VLM improved: could not parse dynamics from: {raw!r}")
        return out
    # Single-call path
    if use_temporal_reasoning:
        prompt = (
            "These images are consecutive frames from a single physics video (first to last in time order). "
            "Use TEMPORAL REASONING to classify the physical dynamics.\n\n"
            + temporal_instructions
            + f"\nReply with ONLY one word: {choices_str}. No other text."
        )
    else:
        prompt = (
            "These images are sample frames from a single physics experiment video (in time order). "
            "Classify the type of physical dynamics into exactly one of the following. "
            "Use the descriptions to tell similar cases apart:\n"
            + descriptions
            + "\nReply with only one word: the dynamics type. No other text."
        )

    content = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    try:
        raw = _call_vlm(client, model, content, max_tokens=64)
    except Exception as e:
        print(f"VLM improved: API error: {e}")
        return None

    out = _parse_dynamics_from_response(raw, choices=choices)
    if out is None:
        print(f"VLM improved: could not parse dynamics from: {raw!r}")
    return out
