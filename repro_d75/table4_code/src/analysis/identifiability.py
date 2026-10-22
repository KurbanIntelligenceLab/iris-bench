"""
Identifiability analysis: injectivity, ODE residual, affine mapping z_learned vs z_real.
"""

import torch
import numpy as np
from typing import Dict, Optional

try:
    from scipy.stats import pearsonr
except ImportError:
    pearsonr = None


class IdentifiabilityAnalyzer:
    def __init__(self, encoder, physics_block, dt: float):
        self.encoder = encoder
        self.physics = physics_block
        self.dt = dt

    def check_injectivity(
        self, frames: torch.Tensor, threshold: float = 1e-4
    ) -> Dict:
        """Empirically verify encoder injectivity. frames: [T, C, H, W] or [T, D]."""
        with torch.no_grad():
            if frames.dim() > 2:
                flat = frames.flatten(1)
            else:
                flat = frames
            z_all = self.encoder(flat)
            z_all = z_all.reshape(z_all.shape[0], -1)

        z_dists = torch.cdist(z_all.unsqueeze(0), z_all.unsqueeze(0)).squeeze(0)
        x_dists = torch.cdist(flat.unsqueeze(0), flat.unsqueeze(0)).squeeze(0)

        distinct_frames = x_dists > threshold
        close_z = z_dists < threshold
        collisions = ((distinct_frames & close_z).sum().item() - z_all.shape[0]) // 2
        collisions = max(0, collisions)

        z_dists_nz = z_dists[z_dists > 0]
        min_z_dist = z_dists_nz.min().item() if z_dists_nz.numel() > 0 else 0.0

        return {
            "is_injective": collisions == 0,
            "min_z_distance": min_z_dist,
            "num_collisions": collisions,
        }

    def compute_affine_mapping(
        self, z_learned: np.ndarray, z_real: np.ndarray
    ) -> Dict:
        """Fit z_learned = a * z_real + b. Return R², slope, intercept, Pearson."""
        if z_learned.ndim == 1:
            z_learned = z_learned.reshape(-1, 1)
            z_real = z_real.reshape(-1, 1)

        results = {}
        for dim in range(z_learned.shape[1]):
            zl = z_learned[:, dim]
            zr = z_real[:, dim]
            A = np.vstack([zr, np.ones(len(zr))]).T
            (slope, intercept), _, _, _ = np.linalg.lstsq(A, zl, rcond=None)
            ss_res = np.sum((zl - slope * zr - intercept) ** 2)
            ss_tot = np.sum((zl - np.mean(zl)) ** 2) + 1e-12
            R2 = 1 - ss_res / ss_tot
            r, p = (pearsonr(zl, zr) if pearsonr else (0.0, 1.0))
            results[f"dim_{dim}"] = {
                "R_squared": float(R2),
                "slope": float(slope),
                "intercept": float(intercept),
                "pearson_r": float(r),
                "p_value": float(p),
            }
        return results

    def compute_ode_residual(self, frames: torch.Tensor) -> Dict:
        """Residual = z_{t+1} - P(z_t, z_{t-1}). Frames: [T, C, H, W] or [T, D]."""
        with torch.no_grad():
            if frames.dim() > 2:
                flat = frames.flatten(1)
            else:
                flat = frames
            z_all = self.encoder(flat)
            if z_all.dim() == 3:
                z_all = z_all.squeeze(1)

        # z_all from encoder is [T, N, d]; physics expects [batch, ode_order+1, N, d]
        if z_all.dim() == 2:
            z_all = z_all.unsqueeze(-1)
        residuals = []
        for t in range(1, len(z_all) - 1):
            z_hist = torch.stack([z_all[t - 1], z_all[t]], dim=0).unsqueeze(0)
            z_pred = self.physics(z_hist, self.dt)
            res = (z_all[t + 1] - z_pred.squeeze(0)).norm().item()
            residuals.append(res)

        residuals = np.array(residuals) if residuals else np.array([0.0])
        return {
            "mean_residual": float(residuals.mean()),
            "max_residual": float(residuals.max()),
            "residual_time_series": residuals.tolist(),
        }

    def parameter_identifiability_report(
        self, gamma_learned: Dict, gamma_true: Dict
    ) -> Dict:
        """Compare learned vs true parameters."""
        report = {}
        for name in gamma_learned:
            gl = gamma_learned[name]
            gt = gamma_true.get(name, gl)
            gl = gl if isinstance(gl, (int, float)) else float(gl.item())
            gt = gt if isinstance(gt, (int, float)) else float(gt)
            abs_err = abs(gl - gt)
            rel_err = abs_err / (abs(gt) + 1e-10)
            report[name] = {
                "learned": gl,
                "true": gt,
                "absolute_error": abs_err,
                "relative_error": rel_err,
            }
        return report

    def full_identifiability_analysis(
        self,
        frames: torch.Tensor,
        z_real: Optional[np.ndarray] = None,
        gamma_true: Optional[Dict] = None,
    ) -> Dict:
        """Run full analysis."""
        results = {}
        results["injectivity"] = self.check_injectivity(frames)
        results["ode_residual"] = self.compute_ode_residual(frames)

        if z_real is not None:
            with torch.no_grad():
                if frames.dim() > 2:
                    flat = frames.flatten(1)
                else:
                    flat = frames
                z_learned = self.encoder(flat)
                z_learned = z_learned.reshape(z_learned.shape[0], -1).cpu().numpy()
            results["affine_mapping"] = self.compute_affine_mapping(z_learned, z_real)

        if gamma_true is not None:
            gamma_learned = {
                name: p.item() if p.numel() == 1 else p.mean().item()
                for name, p in self.physics.named_parameters()
                if "gamma" in name or "kappa" in name
            }
            results["parameter_comparison"] = self.parameter_identifiability_report(
                gamma_learned, gamma_true
            )

        return results
