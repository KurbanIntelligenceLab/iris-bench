"""
Identifiability diagnostics (main paper Table 6).

For each phenomenon/setting this script trains the corrected one-step baseline
and logs the physics-parameter gradient norm

    G_gamma^(e) = || d L / d gamma ||_2

at selected epochs (1, 50, 200), together with the converged ODE residual

    R = (T-1)^{-1} * sum_t || z_{t+1} - P(z_t; gamma, dt) ||^2

These are exactly the quantities reported in Table 6. The uncorrected baseline
yields G_gamma ~ 0 at all epochs (the gradient-flow bug), so only the corrected
variant is logged here.

Usage
-----
  # Real IRIS .npy data (converted with video2npy):
  python scripts/run_identifiability.py --iris_root ./IRIS_npy --outfolder identifiability

  # Self-contained known-physics data (no real data needed):
  python scripts/run_identifiability.py --synthetic --outfolder identifiability

Outputs (under Results/<outfolder>/)
  identifiability.csv      — phenomenon, setting, G@1, G@50, G@200, R
  table_identifiability.tex — booktabs LaTeX table matching Table 6

Determinism: fixed seed (42); same EndPhys model, optimizer, and dt as main.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Repo root = parent of this script's folder. Portable: no hardcoded path.
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
N_CLIPS = 10
N_FRAMES = 10
IMG_H, IMG_W = 56, 100
LOG_EPOCHS = (1, 50, 200)

# Representative (phenomenon, setting, pmodel) rows mirroring Table 6.
ROWS = [
    ("Pendulum",       "pend_45",  "pendulum"),
    ("Rotation",       "mid",      "rotation"),
    ("Sliding cone",   "cone_60",  "sliding_cone"),
    ("Dropping ball",  "drop_100", "dropping_ball"),
    ("Two mov. pend.", "pend_45",  "two_moving_pendulums"),
    ("One stat. pend.","pend_45",  "two_moving_pendulum_one_static"),
]

# Known-physics generator parameters per pmodel (used in --synthetic mode).
SYNTH_ACCEL = {
    "pendulum": 9.0,
    "rotation": 4.0,
    "sliding_cone": 0.0,
    "dropping_ball": 9.81,
    "two_moving_pendulums": 9.0,
    "two_moving_pendulum_one_static": 9.0,
}

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


def _scalar_to_frame(val, H=IMG_H, W=IMG_W, lo=-0.5, hi=4.0):
    img = np.zeros((1, H, W), dtype=np.float32)
    col = int(np.clip((val - lo) / (hi - lo) * W, 0, W - 1))
    img[0, :, col] = 1.0
    return img


def _simulate(accel, n_frames, z0, v0, sigma=0.005):
    """Verlet trajectory of z'' = -accel."""
    z = np.zeros(n_frames)
    z[0] = z0
    z[1] = z0 + v0 * DT
    for t in range(1, n_frames - 1):
        z[t + 1] = 2 * z[t] - z[t - 1] - accel * DT * DT
    z += np.random.randn(n_frames) * sigma
    return z.astype(np.float32)


def synthetic_clips(pmodel, n_clips=N_CLIPS):
    accel = SYNTH_ACCEL[pmodel]
    clips = []
    for k in range(n_clips):
        # Deterministic per-clip RNG (no process-randomized hash()).
        rng = np.random.default_rng(SEED + k)
        z0 = rng.uniform(1.0, 2.0)
        v0 = rng.uniform(-0.5, 0.5)
        traj = _simulate(accel, N_FRAMES, z0, v0)
        clips.append(np.stack([_scalar_to_frame(z) for z in traj], axis=0))
    return np.stack(clips, axis=0).astype(np.float32)  # (N, T, 1, H, W)


def load_iris_clips(iris_root, phenomenon, setting):
    """Find a .npy for (phenomenon, setting) under iris_root; return (N,T,1,H,W) or None."""
    for folder, _, files in os.walk(iris_root):
        low = folder.lower()
        if phenomenon.lower().replace(" ", "_").rstrip("s") in low and setting.lower() in low:
            for f in files:
                if f.endswith(".npy"):
                    arr = np.load(os.path.join(folder, f), allow_pickle=True)
                    if getattr(arr, "ndim", 0) >= 4:
                        return arr.astype(np.float32)
    return None


def grad_norm(model):
    """L2 norm of gradient w.r.t. physics parameters (alpha/beta of pModel)."""
    total = 0.0
    for p in model.pModel.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return total ** 0.5


def ode_residual(model, data):
    """R = mean_t || z_{t+1} - P(z_t) ||^2 over the encoded trajectory."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(data).float().to(device)
        z_enc, z_phys, _ = model(x)
        res = (z_enc[:, model.order:, :] - z_phys[:, model.order:, :]).pow(2)
        return float(res.mean().item())


def run_row(pmodel, data):
    torch.manual_seed(SEED)
    model = mainmodel.EndPhys(dt=DT, pmodel=pmodel, init_phys=10.0, initw=True).to(device)
    opt = torch.optim.Adam(
        [{"params": model.encoder.parameters(), "lr": LR_ENC},
         {"params": model.pModel.parameters(), "lr": LR_PHYS}]
    )
    x = torch.from_numpy(data).float().to(device)
    g_at = {}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        z_enc, z_phys, _ = model(x)
        loss = nn.MSELoss()(z_enc[:, model.order:, :], z_phys[:, model.order:, :])
        loss.backward()
        if epoch in LOG_EPOCHS:
            g_at[epoch] = grad_norm(model)
        opt.step()
    R = ode_residual(model, data)
    return g_at, R


def main():
    ap = argparse.ArgumentParser(description="Identifiability diagnostics (Table 6).")
    ap.add_argument("--iris_root", type=str, default=None,
                    help="Root of IRIS .npy data. If omitted, --synthetic is used.")
    ap.add_argument("--synthetic", action="store_true",
                    help="Use known-physics synthetic data (no real data needed).")
    ap.add_argument("--outfolder", type=str, default="identifiability")
    args = ap.parse_args()

    use_synth = args.synthetic or (args.iris_root is None)
    out_dir = os.path.join("Results", args.outfolder)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for phen, setting, pmodel in ROWS:
        if use_synth:
            data = synthetic_clips(pmodel)
            src = "synthetic"
        else:
            data = load_iris_clips(args.iris_root, phen, setting)
            if data is None:
                print(f"[skip] no .npy for {phen}/{setting} under {args.iris_root}")
                continue
            src = "iris"
        print(f"[{src}] {phen} / {setting} ({pmodel})  data {data.shape}")
        g_at, R = run_row(pmodel, data)
        results.append({
            "phenomenon": phen, "setting": setting,
            "G_1": g_at.get(1, float('nan')),
            "G_50": g_at.get(50, float('nan')),
            "G_200": g_at.get(200, float('nan')),
            "R": R,
        })
        print(f"    G@1={results[-1]['G_1']:.3e}  G@50={results[-1]['G_50']:.3e}  "
              f"G@200={results[-1]['G_200']:.3e}  R={R:.3e}")

    csv_path = os.path.join(out_dir, "identifiability.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["phenomenon", "setting", "G_1", "G_50", "G_200", "R"])
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {csv_path}")

    tex = (
        "\\begin{table}[t]\n  \\centering\n"
        "  \\caption{Identifiability diagnostics: physics-parameter gradient norms "
        "$G_\\gamma^{(e)}=\\lVert\\nabla_\\gamma\\mathcal{L}^{(e)}\\rVert_2$ at epochs 1, 50, 200, "
        "and converged ODE residual $\\mathcal{R}$. The uncorrected baseline yields "
        "$G_\\gamma\\approx0$ at all epochs (omitted).}\n"
        "  \\label{tab:identifiability}\n  \\begin{tabular}{llcccc}\n    \\toprule\n"
        "    Phenomenon & Setting & $G_\\gamma^{(1)}$ & $G_\\gamma^{(50)}$ & "
        "$G_\\gamma^{(200)}$ & $\\mathcal{R}\\,(\\times10^{-3})$ \\\\\n    \\midrule\n"
    )
    for r in results:
        tex += (f"    {r['phenomenon']} & {r['setting']} & "
                f"{r['G_1']:.1e} & {r['G_50']:.1e} & {r['G_200']:.1e} & "
                f"{r['R']*1e3:.2f} \\\\\n")
    tex += "    \\bottomrule\n  \\end{tabular}\n\\end{table}\n"
    tex_path = os.path.join(out_dir, "table_identifiability.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"Saved: {tex_path}")


if __name__ == "__main__":
    main()
