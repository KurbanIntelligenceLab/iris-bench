"""
Energy conservation tracking for Hamiltonian/near-Hamiltonian systems.
Works for any N (single or multi-body).
"""

import torch
from typing import List


class EnergyTracker:
    """Track total energy H = 0.5 * z'^2 + 0.5 * gamma_0 * z^2 per object (no coupling)."""

    def compute_energy(
        self,
        Z: torch.Tensor,
        Zd: torch.Tensor,
        physics_block,
    ) -> torch.Tensor:
        """
        Z, Zd: [batch, N, d]. Returns H: [batch].
        """
        H = torch.zeros(Z.shape[0], device=Z.device, dtype=Z.dtype)
        for i in range(Z.shape[1]):
            gamma_0 = physics_block.gamma[f"gamma_{i}_0"]
            z_i = Z[:, i, :]
            zd_i = Zd[:, i, :]
            H = H + 0.5 * (zd_i ** 2).sum(dim=-1) + 0.5 * gamma_0 * (z_i ** 2).sum(dim=-1)
        return H

    def track_rollout(
        self,
        physics_block,
        Z0: torch.Tensor,
        Zd0: torch.Tensor,
        dt: float,
        num_steps: int,
    ) -> tuple:
        """
        Z0, Zd0: [1, N, d]. Returns (energies list, Z_trajectory list).
        """
        energies = []
        Z_traj = [Z0]
        Z, Zd = Z0, Zd0

        for _ in range(num_steps):
            energies.append(self.compute_energy(Z, Zd, physics_block).item())
            Z, Zd = physics_block.integrator(
                Z, Zd, dt, physics_block.compute_acceleration
            )
            Z_traj.append(Z)

        return energies, Z_traj
