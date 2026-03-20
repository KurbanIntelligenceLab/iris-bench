# VLM reasoning alternatives (no fine-tuning)

Fine-tuning VLMs for this 6-way dynamics task has not worked well (collapse to one class, ~16.67% accuracy). Below are **reasoning-based** options that use an API VLM with better prompts and no training.

## What you already have

- **Temporal reasoning** (`use_temporal_reasoning=True` in `detector.py`): compare first vs last frame, then choose. **~73%** on Delfys75.
- **Describe-then-classify** (`use_describe_then_classify=True`): first ask for a short motion description (with images), then a second call classifies from that text only. Two API calls per video. Eval: `python scripts/evaluate_vlm_dynamics_improved.py --path ./delfys75 --out Results/vlm_describe_then_classify --describe_then_classify`

## Options you can try

| Approach | Idea | Pros / cons |
|----------|------|------------------|
| **Temporal reasoning** | Compare first vs last frame; use rules (size change → free_fall, back-and-forth → pendulum, etc.). | Already implemented; 73% on Delfys75. |
| **Describe then classify** | Call 1: "Describe the motion in 1–2 sentences." Call 2: "Given: «description». Which dynamics: …?" | Forces verbalization; two API calls per video. |
| **Chain-of-thought (single call)** | One prompt: "Step 1: What motion do you see? Step 2: So which dynamics: …? Reply with one word at the end." | One call; model may still skip reasoning. |
| **Hierarchical** | First: "Is the motion periodic, one-directional, or other?" Then narrow (e.g. periodic → pendulum vs LED). | Smaller decisions; more calls or longer prompt. |
| **Few-shot in prompt** | Add 1–2 short example descriptions per class in the system message. | Clearer anchors; uses more tokens. |
| **Self-consistency** | Run the same prompt 3–5 times with temperature > 0; majority vote. | More robust; 3–5× API cost. |
| **Stronger model** | Switch model (e.g. GPT-4o → Claude or Gemini) or use a dedicated vision model. | May improve without changing prompt. |

## Recommendation

For your pipeline, the best **VLM** option without training is:

1. **Keep temporal reasoning** as the default (already 73%).
2. **Try describe-then-classify** for comparison (see `use_describe_then_classify` in `detector.py`).
3. For highest accuracy on Delfys75, use the **video classifier** (trained on the same data; 100% on the eval set, but watch overfitting on a held-out test set).
