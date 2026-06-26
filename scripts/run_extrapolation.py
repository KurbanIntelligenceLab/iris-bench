"""
Extrapolation-beyond-training-horizon analysis (main paper Table 7).

For the pendulum (the long, 150 s clips) the corrected one-step baseline is
trained on the first frames, then the learned ODE is unrolled past the training
window. The extrapolation error at horizon k is

    E_k = || zhat_k - zbar_k ||^2 ,   k > T_train          (paper Eq. 2)

where zhat_k is obtained by unrolling the learned physics block from the last
training frame and zbar_k is the encoder output for the held-out frame I_k.
Mean +/- std are reported over the test clips at k in {10, 25, 50}.

Usage
-----
  python scripts/run_extrapolation.py --iris_root ./IRIS_npy --outfolder extrapolation
  python scripts/run_extrapolation.py --synthetic --outfolder extrapolation

Outputs (under Results/<outfolder>/)
  extrapolation.csv        — setting, E@10_mean, E@10_std, E@25_..., E@50_...
  table_extrapolation.tex  — booktabs LaTeX matching Table 7

Determinism: fixed seed (42); same EndPhys model and dt as main.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.models import model as mainmodel

SEED = 42
DT = 1 / 60
EPOCHS = 200
LR_ENC = 1e-3
LR_PHYS = 0.01
IMG_H, IMG_W = 56, 100
TRAIN_FRAMES = 100          # train on first ~1.67 s (matches paper)
HORIZONS = (10, 25, 50)     # frames beyond the training window
N_CLIPS = 6                 # test clips per setting
PMODEL = "pendulum"

# pendulum settings: initial release angle in degrees -> larger angle, stronger nonlinearity
SETTINGS = {"pend_20": 20.0, "pend_45": 45.0, "pend_90": 90.0}

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


def _scalar_to_frame(val, H=IMG_H, W=IMG_W, lo=-1.5, hi=1.5):
    img = np.zeros((1, H, W), dtype=np.float32)
    col = int(np.clip((val - lo) / (hi - lo) * W, 0, W - 1))
    img[0, :, col] = 1.0
    return img


def _pendulum_traj(theta0_deg, n_frames, seed, sigma=0.003):
    """Nonlinear pendulum theta'' = -(g/L) sin(theta), encoded as a 1-D signal."""
    rng = np.random.default_rng(seed)
    g_over_L = 9.81 / 0.5
    theta = np.deg2rad(theta0_deg)
    omega = 0.0
    out = np.zeros(n_frames)
    for t in range(n_frames):
        out[t] = np.sin(theta)               # observable ~ horizontal projection
        omega += -g_over_L * np.sin(theta) * DT
        theta += omega * DT
    out += rng.standard_normal(n_frames) * sigma
    return out.astype(np.float32)


def synthetic_clips(theta0, n_clips, total_frames, base_seed):
    clips = []
    for k in range(n_clips):
        traj = _pendulum_traj(theta0, total_frames, seed=base_seed + k)
        clips.append(np.stack([_scalar_to_frame(z) for z in traj], axis=0))
    return np.stack(clips, axis=0).astype(np.float32)   # (N, T, 1, H, W)


def load_iris_clips(iris_root, setting):
    key = setting.replace("pend_", "pendulum_")
    for folder, _, files in os.walk(iris_root):
        low = folder.lower()
        if "pendulum" in low and key in low and "two" not in low and "static" not in low:
            for f in files:
                if f.endswith(".npy"):
                    arr = np.load(os.path.join(folder, f), allow_pickle=True)
                    if getattr(arr, "ndim", 0) >= 4:
                        return arr.astype(np.float32)
    return None


def train_on_window(data):
    """Train EndPhys on the first TRAIN_FRAMES frames of every clip."""
    torch.manual_seed(SEED)
    model = mainmodel.EndPhys(dt=DT, pmodel=PMODEL, init_phys=10.0, initw=True).to(device)
    opt = torch.optim.Adam(
        [{"params": model.encoder.parameters(), "lr": LR_ENC},
         {"params": model.pModel.parameters(), "lr": LR_PHYS}]
    )
    x = torch.from_numpy(data[:, :TRAIN_FRAMES]).float().to(device)
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        z_enc, z_phys, _ = model(x)
        loss = nn.MSELoss()(z_enc[:, model.order:, :], z_phys[:, model.order:, :])
        loss.backward()
        opt.step()
    return model


def extrapolation_errors(model, data):
    """For each clip, unroll the learned ODE past TRAIN_FRAMES and compare to encoder."""
    order = model.order
    model.eval()
    per_horizon = {k: [] for k in HORIZONS}
    with torch.no_grad():
        x = torch.from_numpy(data).float().to(device)
        for n in range(x.shape[0]):
            clip = x[n:n + 1]
            # encode the full clip
            z_enc, _, _ = model(clip)              # [1, T, d]
            T = z_enc.shape[1]
            # unroll physics from the last two training frames
            zroll = z_enc[:, TRAIN_FRAMES - order:TRAIN_FRAMES, :].clone()
            max_k = max(HORIZONS)
            for _ in range(max_k):
                window = zroll[:, -order:, :]
                pred = model.pModel(window, model.dt)
                zroll = torch.cat([zroll, pred], dim=1)
            for k in HORIZONS:
                idx = TRAIN_FRAMES + k
                if idx < T:
                    zhat = zroll[:, order + k - 1, :]
                    zbar = z_enc[:, idx, :]
                    per_horizon[k].append(float((zhat - zbar).pow(2).sum().item()))
    return {k: (float(np.mean(v)) if v else float("nan"),
                float(np.std(v)) if v else float("nan"))
            for k, v in per_horizon.items()}


def main():
    ap = argparse.ArgumentParser(description="Extrapolation error beyond training horizon (Table 7).")
    ap.add_argument("--iris_root", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--outfolder", type=str, default="extrapolation")
    args = ap.parse_args()

    use_synth = args.synthetic or (args.iris_root is None)
    out_dir = os.path.join("Results", args.outfolder)
    os.makedirs(out_dir, exist_ok=True)
    total_frames = TRAIN_FRAMES + max(HORIZONS) + 5

    results = []
    for setting, theta0 in SETTINGS.items():
        if use_synth:
            data = synthetic_clips(theta0, N_CLIPS, total_frames, base_seed=SEED + int(theta0))
        else:
            data = load_iris_clips(args.iris_root, setting)
            if data is None:
                print(f"[skip] no .npy for pendulum/{setting} under {args.iris_root}")
                continue
            if data.shape[1] < TRAIN_FRAMES + max(HORIZONS):
                print(f"[skip] {setting}: clip too short ({data.shape[1]} frames)")
                continue
        print(f"{setting} (theta0={theta0})  data {data.shape}")
        model = train_on_window(data)
        errs = extrapolation_errors(model, data)
        row = {"setting": setting}
        for k in HORIZONS:
            row[f"E_{k}_mean"], row[f"E_{k}_std"] = errs[k]
        results.append(row)
        print("    " + "  ".join(f"E@{k}={errs[k][0]:.3f}±{errs[k][1]:.3f}" for k in HORIZONS))

    csv_path = os.path.join(out_dir, "extrapolation.csv")
    fields = ["setting"] + [f"E_{k}_{s}" for k in HORIZONS for s in ("mean", "std")]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {csv_path}")

    tex = (
        "\\begin{table}[t]\n  \\centering\n"
        "  \\caption{Extrapolation error $\\mathcal{E}_k$ (mean $\\pm$ std over test clips) "
        "at selected horizons $k$ beyond the training window.}\n"
        "  \\label{tab:extrapolation}\n  \\begin{tabular}{lccc}\n    \\toprule\n"
        "    Setting & $\\mathcal{E}_{k=10}$ & $\\mathcal{E}_{k=25}$ & $\\mathcal{E}_{k=50}$ \\\\\n"
        "    \\midrule\n"
    )
    for r in results:
        tex += (f"    {r['setting'].replace('_', '\\_')} & "
                f"{r['E_10_mean']:.3f} $\\pm$ {r['E_10_std']:.3f} & "
                f"{r['E_25_mean']:.3f} $\\pm$ {r['E_25_std']:.3f} & "
                f"{r['E_50_mean']:.3f} $\\pm$ {r['E_50_std']:.3f} \\\\\n")
    tex += "    \\bottomrule\n  \\end{tabular}\n\\end{table}\n"
    tex_path = os.path.join(out_dir, "table_extrapolation.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"Saved: {tex_path}")


if __name__ == "__main__":
    main()
