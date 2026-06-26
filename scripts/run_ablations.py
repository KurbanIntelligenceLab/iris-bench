"""
Component ablations (main paper Sec. 6.7).

Two self-contained ablations are reproduced here:

(A) Integrator comparison ("Symplectic integrator"). On a conservative
    (undamped) oscillator we roll out each integrator and measure (i) energy
    drift |H_T - H_0| / H_0 and (ii) ODE residual vs. the analytic trajectory.
    Stormer-Verlet conserves energy on conservative systems; RK4/Yoshida4 are
    higher order; Forward-Euler drifts. This backs the claim that the symplectic
    integrator achieves lower ODE residual on conservative systems with
    comparable performance on non-conservative ones.

(B) Multi-step horizon (K in {1,2,3,5}). A pointer to run_iris_multistep_only.sh
    / the committed iris_comparison results, which already isolate K=1 vs K=5.

The VLM frame-count sweep (single/3/5-frame -> 54/67/73%) is NOT reproduced here
because it requires live VLM API calls (OPENROUTER_API_KEY); see
scripts/evaluate_vlm_dynamics_improved.py and Supplementary E.1 for the prompt /
frame-sampling configuration used to produce those numbers.

Usage
-----
  python scripts/run_ablations.py --outfolder ablations

Outputs (under Results/<outfolder>/)
  integrator_comparison.csv   — integrator, energy_drift, ode_residual
  table_integrators.tex       — booktabs LaTeX

Determinism: fixed seed (42).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.integrators.integrators import get_integrator

SEED = 42
DT = 1 / 60
N_STEPS = 600                 # ~10 s rollout
OMEGA2 = 4.0                  # conservative oscillator z'' = -omega^2 z
INTEGRATORS = ["euler", "stormer_verlet", "rk4", "yoshida4"]

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cpu"   # tiny rollout; CPU keeps it fully deterministic


def accel_fn(z, zd):
    """Undamped linear oscillator: z'' = -omega^2 z (energy-conserving)."""
    return -OMEGA2 * z


def total_energy(z, zd):
    """H = 0.5 z'^2 + 0.5 omega^2 z^2."""
    return 0.5 * (zd ** 2).sum().item() + 0.5 * OMEGA2 * (z ** 2).sum().item()


def analytic(t):
    """z(t) = cos(omega t) for z0=1, zd0=0."""
    return math.cos(math.sqrt(OMEGA2) * t)


def run_integrator(name):
    integ = get_integrator(name)
    z = torch.tensor([[1.0]], device=device)
    zd = torch.tensor([[0.0]], device=device)
    H0 = total_energy(z, zd)
    residuals = []
    for step in range(1, N_STEPS + 1):
        z, zd = integ(z, zd, DT, accel_fn)
        z_true = analytic(step * DT)
        residuals.append((z.item() - z_true) ** 2)
    HT = total_energy(z, zd)
    energy_drift = abs(HT - H0) / (abs(H0) + 1e-12)
    ode_residual = float(np.mean(residuals))
    return energy_drift, ode_residual


def main():
    ap = argparse.ArgumentParser(description="Component ablations (Sec. 6.7).")
    ap.add_argument("--outfolder", type=str, default="ablations")
    args = ap.parse_args()
    out_dir = os.path.join("Results", args.outfolder)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for name in INTEGRATORS:
        drift, res = run_integrator(name)
        rows.append({"integrator": name, "energy_drift": drift, "ode_residual": res})
        print(f"  {name:15s}  energy_drift={drift:.3e}  ode_residual={res:.3e}")

    csv_path = os.path.join(out_dir, "integrator_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["integrator", "energy_drift", "ode_residual"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {csv_path}")

    pretty = {"euler": "Forward-Euler", "stormer_verlet": "St\\\"ormer-Verlet",
              "rk4": "RK4", "yoshida4": "Yoshida4"}
    tex = (
        "\\begin{table}[t]\n  \\centering\n"
        "  \\caption{Integrator ablation on a conservative oscillator: relative energy "
        "drift $|H_T-H_0|/H_0$ and mean ODE residual vs.\\ the analytic trajectory over a "
        "10\\,s rollout. St\\\"ormer-Verlet (symplectic) bounds the energy drift; RK4/Yoshida4 "
        "are higher order; Forward-Euler drifts.}\n"
        "  \\label{tab:integrators}\n  \\begin{tabular}{lcc}\n    \\toprule\n"
        "    Integrator & Energy drift & ODE residual \\\\\n    \\midrule\n"
    )
    for r in rows:
        tex += f"    {pretty.get(r['integrator'], r['integrator'])} & {r['energy_drift']:.2e} & {r['ode_residual']:.2e} \\\\\n"
    tex += "    \\bottomrule\n  \\end{tabular}\n\\end{table}\n"
    tex_path = os.path.join(out_dir, "table_integrators.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"Saved: {tex_path}")

    print("\nNote: K-horizon ablation -> see scripts/run_iris_multistep_only.sh and "
          "Results/iris_comparison/ (K=1 vs K=5).")
    print("Note: VLM frame-count sweep needs OPENROUTER_API_KEY; see "
          "scripts/evaluate_vlm_dynamics_improved.py and Supplementary E.1.")


if __name__ == "__main__":
    main()
