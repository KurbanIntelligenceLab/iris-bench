"""
Train the unified ECCV model (UnifiedPhysicsModel).
Usage:
  python scripts/train_unified.py --config config_unified.yaml --data path/to/video.npy
  python scripts/train_unified.py --config config_unified.yaml --data Data/data.npy --out Results/unified_run
"""

import argparse
import os
import sys
import numpy as np
import torch
import yaml

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.unified_model import UnifiedPhysicsModel
from src.analysis.identifiability import IdentifiabilityAnalyzer


def load_video_tensor(path: str) -> torch.Tensor:
    """Load .npy and return [1, T, C, H, W] float tensor."""
    data = np.load(path, allow_pickle=True)
    if data.ndim == 0 and data.dtype == object:
        data = data.item()
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.ndim == 1:
        nf, h, w = 10, 56, 100
        data = data.reshape(-1, nf, 1, h, w).astype(np.float32)
    if data.shape[0] == 0:
        raise ValueError("No samples in file")
    # Use first sample as single video
    x = data[0:1]
    return torch.from_numpy(x).float()


def configure_optimizer(model, config: dict):
    """Parameter-adaptive LR for physics params (original paper style)."""
    physics_params = []
    encoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "physics" in name or "gamma" in name or "kappa" in name or "coupling" in name:
            init_mag = abs(param.item()) if param.numel() == 1 else 0.01
            init_mag = max(init_mag, 1e-6)
            lr = 10 ** int(np.floor(np.log10(init_mag)))
            physics_params.append({"params": param, "lr": lr})
        else:
            encoder_params.append(param)
    param_groups = [{"params": encoder_params, "lr": config.get("encoder_lr", 0.01)}]
    param_groups.extend(physics_params)
    return torch.optim.Adam(param_groups)


def train(config_path: str, data_path: str, out_dir: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["input_dim"] = config.get("input_dim", 5600)
    config["dt"] = config.get("dt", 0.0167)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    video = load_video_tensor(data_path).to(device)
    # video: [1, T, C, H, W]
    print("Video shape:", video.shape)
    print("Tip: On random/synthetic data, physics params (gamma) may not move much; use real video .npy for meaningful learning.")

    model = UnifiedPhysicsModel(config).to(device)
    optimizer = configure_optimizer(model, config)

    os.makedirs(out_dir, exist_ok=True)
    epochs = config.get("epochs", 500)
    log_interval = config.get("log_interval", 50)

    for epoch in range(epochs):
        model.train()
        result = model(video)
        loss = result["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.6f}")
            # Log current physics params (after this epoch's step)
            grad_norms = []
            for name, p in model.physics.named_parameters():
                if p.numel() == 1:
                    print(f"  {name}: {p.item():.6f}")
                    if p.grad is not None:
                        grad_norms.append(p.grad.item() ** 2)
            if grad_norms:
                print(f"  |grad(gamma)|: {(sum(grad_norms)) ** 0.5:.6f}")

    # Save
    torch.save(model.state_dict(), os.path.join(out_dir, "unified_model.pt"))
    with open(os.path.join(out_dir, "config_unified.yaml"), "w") as f:
        yaml.dump(config, f)

    # Identifiability analysis
    model.eval()
    analyzer = IdentifiabilityAnalyzer(model.encoder, model.physics, model.dt)
    frames = video.squeeze(0)
    report = analyzer.full_identifiability_analysis(frames)
    print("Identifiability injectivity:", report["injectivity"])
    print("ODE mean residual:", report["ode_residual"]["mean_residual"])
    return model, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_unified.yaml")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="Results/unified_run")
    args = ap.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(ROOT, config_path)

    train(config_path, args.data, args.out)


if __name__ == "__main__":
    main()
