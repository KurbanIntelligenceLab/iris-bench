"""
Run the unified ECCV model over all Delfys75 .npy files and write results
in the same CSV format as main.py (run, alpha, beta, max_z, min_z, z0, z1)
so you can compare with baseline and run parameter-error comparison.

Usage:
  # Unified model only (path-based dynamics)
  python scripts/run_unified_delfys75.py --path ./delfys75 --outfolder delfys75_unified --dt 0.05

  # All improvements together: VLM equation-family selection + unified model + multi-step loss
  python scripts/run_unified_delfys75.py --path ./delfys75 --outfolder delfys75_all --dt 0.05 --vlm_improved --multistep

  # Options: --use_vlm / --vlm_improved (equation family from video), --multistep (multi-step physics loss)
  python scripts/run_unified_delfys75.py --path ./delfys75 --outfolder delfys75_unified --dt 0.05 --config config_unified.yaml --epochs 300 --vlm_improved --multistep
"""

import argparse
import csv
import os
import sys
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.unified_model import UnifiedPhysicsModel


def get_dynamics_from_path(path: str) -> str:
    """Extract dynamics keyword from path (same logic as main.py)."""
    dynamics_keywords = [
        "two_moving_pendulum_one_static", "two_moving_pendulums",
        "dropping_ball", "falling_ball", "sliding_cone", "hitting_cones", "rotation",
        "pendulum", "sliding_block", "bouncing_ball", "dropped_ball",
        "led", "free_fall", "torricelli",
    ]
    normalized = path.replace(" ", "").lower()
    for keyword in dynamics_keywords:
        if keyword in normalized:
            return keyword
    return None


def get_dynamics(
    file_path: str,
    use_vlm: bool = False,
    vlm_improved: bool = False,
) -> str:
    """
    Get dynamics for this video: VLM (if enabled) or path-based.
    Returns dynamics keyword or None if unknown.
    """
    if use_vlm:
        try:
            if vlm_improved:
                from src.utils.vlm_improved import detect_dynamics_from_npy
            else:
                from src.utils.vlm_dynamics import detect_dynamics_from_npy
            pred = detect_dynamics_from_npy(file_path)
            if pred is not None:
                return pred
        except Exception as e:
            print(f"  VLM failed ({e}); falling back to path.")
        return get_dynamics_from_path(file_path)
    return get_dynamics_from_path(file_path)


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
    return torch.from_numpy(data[0:1].astype(np.float32))


def configure_optimizer(model, config: dict):
    """Parameter-adaptive LR for physics params."""
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


# Multi-object dynamics: use num_objects=2 and coupling (IRIS)
MULTI_OBJECT_DYNAMICS = ("hitting_cones", "two_moving_pendulums", "two_moving_pendulum_one_static")


def config_for_dynamics(base_config: dict, dynamics: str) -> dict:
    """Override config for multi-object dynamics (N=2, coupling)."""
    config = dict(base_config)
    if dynamics in MULTI_OBJECT_DYNAMICS:
        config["num_objects"] = 2
        config["coupling_type"] = "contact" if dynamics == "hitting_cones" else "double_pendulum"
    return config


def train_one_video(config: dict, video: torch.Tensor, device: torch.device, epochs: int, log_interval: int):
    """Train UnifiedPhysicsModel on one video; return model and [alpha, beta, max_z, min_z, z0, z1]."""
    model = UnifiedPhysicsModel(config).to(device)
    optimizer = configure_optimizer(model, config)

    for epoch in range(epochs):
        model.train()
        result = model(video)
        loss = result["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if log_interval and (epoch + 1) % log_interval == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f}")

    # Extract alpha, beta (object 0: gamma_0_0, gamma_0_1; for CSV compatibility)
    alpha = model.physics.gamma["gamma_0_0"].detach().cpu().item()
    beta = model.physics.gamma["gamma_0_1"].detach().cpu().item()

    # Encode full sequence to get z stats (match baseline evaluate_model semantics)
    model.eval()
    with torch.no_grad():
        Z = model.encode_sequence(video)  # [1, T, N, d]
    z_flat = Z[0, :, 0, 0].cpu().numpy()  # [T]
    max_z = float(np.max(z_flat))
    min_z = float(np.min(z_flat))
    z0 = float(z_flat[0])
    z1 = float(z_flat[-1])

    return model, [alpha, beta, max_z, min_z, z0, z1]


def run_pipeline(
    root_folder: str,
    output_folder: str,
    dt: float,
    config_path: str,
    epochs: int = 500,
    log_interval: int = 50,
    use_vlm: bool = False,
    vlm_improved: bool = False,
    use_multistep_loss: bool = False,
):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["input_dim"] = config.get("input_dim", 5600)
    config["dt"] = dt
    config["epochs"] = epochs
    config["use_multistep_loss"] = use_multistep_loss
    if use_multistep_loss:
        config.setdefault("multistep_num_steps", 5)
        config.setdefault("multistep_step_weights", [1.0, 1.0, 0.5, 0.5, 0.25])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    out_dir = os.path.join("./Results", output_folder)
    os.makedirs(out_dir, exist_ok=True)

    for folder_name, _, files in os.walk(root_folder):
        for file in files:
            if not file.endswith(".npy"):
                continue
            file_path = os.path.join(folder_name, file)
            relative = os.path.relpath(folder_name, root_folder)
            path_components = relative.split(os.sep)

            dynamics = get_dynamics(file_path, use_vlm=use_vlm, vlm_improved=vlm_improved)
            if dynamics is None:
                print(f"Skipping (no dynamics in path): {file_path}")
                continue

            if use_vlm:
                print(f"Processing {file_path} (dynamics={dynamics}) ...")
            else:
                print(f"Processing {file_path} ...")
            try:
                video = load_video_tensor(file_path).to(device)
                if video.shape[1] < 3:
                    print(f"  Skip: too few frames ({video.shape[1]})")
                    results.append(path_components + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    continue
                # Multi-object dynamics: use N=2 and coupling (hitting_cones, two pendulums)
                run_config = config_for_dynamics(config, dynamics)
                _, [a, b, max_z, min_z, z0, z1] = train_one_video(
                    run_config, video, device, epochs=epochs, log_interval=log_interval
                )
                results.append(path_components + [a, b, max_z, min_z, z0, z1])
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results.append(path_components + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Write CSV same format as main.py (see main.py iterate_folders_and_process)
    max_depth = max(len(row) - 2 for row in results)
    headers = ["run", "alpha", "beta", "max_z", "min_z", "z0", "z1"]
    with open(os.path.join(out_dir, f"{output_folder}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in results:
            padded_row = row[:-2] + [""] * (max_depth - len(row[:-2])) + row[-2:]
            writer.writerow(padded_row)

    print(f"Unified pipeline done. Results: ./Results/{output_folder}/{output_folder}.csv")
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Run unified model on Delfys75 and write main.py-style CSV. "
        "Use --use_vlm/--vlm_improved for equation-family selection and --multistep for multi-step loss to combine all improvements."
    )
    ap.add_argument("--path", type=str, required=True, help="Root folder containing .npy files (e.g. ./delfys75)")
    ap.add_argument("--outfolder", type=str, default="delfys75_unified", help="Output folder under ./Results/")
    ap.add_argument("--dt", type=float, default=0.05, help="Time step (e.g. 0.05 for Delfys75)")
    ap.add_argument("--config", type=str, default="config_unified.yaml", help="Unified model config")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--use_vlm", action="store_true", help="Stage 1: Use VLM to select equation family from video (else path-based).")
    ap.add_argument("--vlm_improved", action="store_true", help="Use improved VLM (enhanced prompt + 5 frames). Implies --use_vlm.")
    ap.add_argument("--multistep", action="store_true", help="Use multi-step physics loss (horizons 1..5, weighted).")
    args = ap.parse_args()

    use_vlm = args.use_vlm or args.vlm_improved
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(ROOT, config_path)
    run_pipeline(
        args.path,
        args.outfolder,
        args.dt,
        config_path,
        epochs=args.epochs,
        log_interval=args.log_interval,
        use_vlm=use_vlm,
        vlm_improved=args.vlm_improved,
        use_multistep_loss=args.multistep,
    )


if __name__ == "__main__":
    main()
