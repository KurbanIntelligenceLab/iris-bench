"""
Multi-body interaction graph for coupled dynamical systems.
ONE physics block for N=1 (zero edges) and N>1 (coupling edges).
All tensors [batch, N, d] throughout.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

# Lazy import to avoid circular dependency; use get_integrator at runtime
def _get_integrator(name: str):
    from src.integrators.integrators import get_integrator
    return get_integrator(name)


class CouplingFunction(nn.Module):
    """
    Learnable coupling C_{ij}(z_i, z_j; κ).
    Supports: linear, contact, double_pendulum.
    """

    def __init__(self, coupling_type: str, d: int, **kwargs):
        super().__init__()
        self.coupling_type = coupling_type
        self.d = d

        if coupling_type == "linear":
            kappa_init = kwargs.get("kappa_init", 0.1)
            self.kappa = nn.Parameter(torch.tensor(kappa_init, dtype=torch.float32))

        elif coupling_type == "contact":
            self.kappa = nn.Parameter(torch.tensor(kwargs.get("kappa_init", 1.0), dtype=torch.float32))
            self.contact_radius = nn.Parameter(torch.tensor(kwargs.get("R_init", 0.1), dtype=torch.float32))
            self.epsilon = kwargs.get("epsilon", 1e-3)

        elif coupling_type == "double_pendulum":
            self.mass_ratio = nn.Parameter(torch.tensor(kwargs.get("mass_ratio_init", 1.0), dtype=torch.float32))
            self.length_ratio = nn.Parameter(torch.tensor(kwargs.get("length_ratio_init", 1.0), dtype=torch.float32))
            self.g_over_L1 = nn.Parameter(torch.tensor(kwargs.get("g_over_L1_init", 9.81), dtype=torch.float32))
            self.kappa = nn.Parameter(torch.tensor(kwargs.get("kappa_init", 0.1), dtype=torch.float32))

        else:
            self.kappa = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        zd_i: torch.Tensor,
        zd_j: torch.Tensor,
    ) -> torch.Tensor:
        """Coupling acceleration on object i due to j. [batch, d]."""
        if self.coupling_type == "linear":
            # κ * (z_i - z_j) as spring-like force → acceleration
            diff = z_i - z_j
            return self.kappa * diff

        elif self.coupling_type == "contact":
            # Soft repulsion: φ(r) ∝ max(0, R - r)^2 / ε
            diff = z_i - z_j
            r = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)
            direction = diff / r
            overlap = (self.contact_radius - r).clamp(min=0.0)
            force = self.kappa * (overlap ** 2) / (self.epsilon + 1e-8)
            return force * direction

        elif self.coupling_type == "double_pendulum":
            # Linearized coupling for small angles: kappa * (z_j - z_i)
            return self.kappa * (z_j - z_i)

        else:
            diff = z_i - z_j
            return self.kappa * diff


class InteractionPhysicsBlock(nn.Module):
    """
    Graph-structured physics block. Same class for N=1 and N>1.
    N=1 → adjacency empty → only self-dynamics (original paper).
    All tensors [batch, N, d].
    """

    def __init__(
        self,
        num_objects: int = 1,
        d_per_object: int = 1,
        ode_order: int = 2,
        coupling_type: str = "linear",
        adjacency: Optional[List[Tuple[int, int]]] = None,
        integrator_type: str = "stormer_verlet",
    ):
        super().__init__()
        self.N = num_objects
        self.d = d_per_object
        self.ode_order = ode_order

        # Per-object ODE parameters: gamma_{i}_{k} for z'' + gamma_1*z' + gamma_0*z = 0
        self.gamma = nn.ParameterDict()
        for i in range(num_objects):
            for k in range(ode_order):
                self.gamma[f"gamma_{i}_{k}"] = nn.Parameter(torch.tensor(0.5 * (0.1 ** k), dtype=torch.float32))

        if adjacency is None:
            adjacency = [
                (i, j) for i in range(num_objects) for j in range(num_objects) if i != j
            ]
        self.adjacency = adjacency

        self.couplings = nn.ModuleDict()
        for (i, j) in adjacency:
            self.couplings[f"coupling_{i}_{j}"] = CouplingFunction(
                coupling_type=coupling_type, d=d_per_object
            )

        self.integrator = _get_integrator(integrator_type)

    def compute_acceleration(self, Z: torch.Tensor, Zd: torch.Tensor) -> torch.Tensor:
        """Z, Zd: [batch, N, d] -> Zdd: [batch, N, d]."""
        Zdd = torch.zeros_like(Z)

        for i in range(self.N):
            z_i = Z[:, i, :]
            zd_i = Zd[:, i, :]

            gamma_0 = self.gamma[f"gamma_{i}_0"]
            gamma_1 = self.gamma[f"gamma_{i}_1"]
            self_accel = -gamma_1 * zd_i - gamma_0 * z_i

            coupling_accel = torch.zeros_like(z_i)
            for (a, b) in self.adjacency:
                if a == i:
                    j = b
                    z_j = Z[:, j, :]
                    zd_j = Zd[:, j, :]
                    coupling_accel = coupling_accel + self.couplings[
                        f"coupling_{a}_{b}"
                    ](z_i, z_j, zd_i, zd_j)

            Zdd[:, i, :] = self_accel + coupling_accel

        return Zdd

    def forward(self, Z_history: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Z_history: [batch, ode_order+1, N, d] (e.g. [batch, 3, N, d] for order 2)
        Returns: Z_next [batch, N, d]
        """
        z_t = Z_history[:, -1]
        z_tm1 = Z_history[:, -2]

        zd_t = (z_t - z_tm1) / (dt + 1e-12)

        z_next, _ = self.integrator(z_t, zd_t, dt, self.compute_acceleration)
        return z_next
