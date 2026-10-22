"""
Numerical integrators for the physics block.

All integrators:
- Accept (z, zd, dt, accel_fn) with z, zd shape [batch, N, d]
- Return (z_next, zd_next) same shapes
- Are fully differentiable (for backprop through physics block)
"""

import torch
import torch.nn as nn
from typing import Callable


class ForwardEuler(nn.Module):
    """
    Original paper's integrator. First-order, NOT symplectic.
    Paper Eq.: z_{t+1} = z_t + dt*(z'_t - dt*(gamma_1*z'_t + gamma_0*z_t)) = z + dt*zd + dt^2*zdd,
    so the predicted position depends on gamma (gradients flow to physics params).
    z'_{t+1} = z'_t + dt * z''_t
    """

    def forward(
        self,
        z: torch.Tensor,
        zd: torch.Tensor,
        dt: float,
        accel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> tuple:
        zdd = accel_fn(z, zd)
        # Include acceleration in position update so loss has gradient w.r.t. gamma
        z_next = z + dt * zd + (dt ** 2) * zdd
        zd_next = zd + dt * zdd
        return z_next, zd_next


class StormerVerlet(nn.Module):
    """
    Störmer-Verlet (Leapfrog). SYMPLECTIC, 2nd order.
    Velocity Verlet variant.
    """

    def forward(
        self,
        z: torch.Tensor,
        zd: torch.Tensor,
        dt: float,
        accel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> tuple:
        zdd = accel_fn(z, zd)
        zd_half = zd + 0.5 * dt * zdd
        z_next = z + dt * zd_half
        zdd_next = accel_fn(z_next, zd_half)
        zd_next = zd_half + 0.5 * dt * zdd_next
        return z_next, zd_next


class RungeKutta4(nn.Module):
    """
    Classical 4th-order Runge-Kutta. NOT symplectic, high accuracy.
    """

    def forward(
        self,
        z: torch.Tensor,
        zd: torch.Tensor,
        dt: float,
        accel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> tuple:
        k1_z = zd
        k1_v = accel_fn(z, zd)

        k2_z = zd + 0.5 * dt * k1_v
        k2_v = accel_fn(z + 0.5 * dt * k1_z, zd + 0.5 * dt * k1_v)

        k3_z = zd + 0.5 * dt * k2_v
        k3_v = accel_fn(z + 0.5 * dt * k2_z, zd + 0.5 * dt * k2_v)

        k4_z = zd + dt * k3_v
        k4_v = accel_fn(z + dt * k3_z, zd + dt * k3_v)

        z_next = z + (dt / 6.0) * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)
        zd_next = zd + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        return z_next, zd_next


class Yoshida4(nn.Module):
    """
    Yoshida 4th-order symplectic integrator.
    Coefficients: w1 = 1/(2-2^{1/3}), w0 = -2^{1/3}/(2-2^{1/3})
    """

    def __init__(self):
        super().__init__()
        cbrt2 = 2.0 ** (1.0 / 3.0)
        w1 = 1.0 / (2.0 - cbrt2)
        w0 = -cbrt2 / (2.0 - cbrt2)
        self.c = [w1 / 2, (w0 + w1) / 2, (w0 + w1) / 2, w1 / 2]
        self.d = [w1, w0, w1]

    def forward(
        self,
        z: torch.Tensor,
        zd: torch.Tensor,
        dt: float,
        accel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> tuple:
        for s in range(3):
            z = z + self.c[s] * dt * zd
            zdd = accel_fn(z, zd)
            zd = zd + self.d[s] * dt * zdd
        z = z + self.c[3] * dt * zd
        return z, zd


def get_integrator(name: str) -> nn.Module:
    """Factory for integrators."""
    integrators = {
        "euler": ForwardEuler,
        "stormer_verlet": StormerVerlet,
        "verlet": StormerVerlet,
        "rk4": RungeKutta4,
        "yoshida4": Yoshida4,
    }
    if name not in integrators:
        raise ValueError(
            f"Unknown integrator: {name}. Choose from {list(integrators.keys())}"
        )
    return integrators[name]()
