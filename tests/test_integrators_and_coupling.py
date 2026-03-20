"""
Unit tests for the newly added ODE components:
- All integrators (euler, stormer_verlet, rk4, yoshida4)
- All coupling types (linear, contact, double_pendulum)
- N=1 (no coupling) and N=2 (with coupling)
Run from project root: python tests/test_integrators_and_coupling.py
Or: pytest tests/test_integrators_and_coupling.py -v
"""

import sys
import os
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.integrators.integrators import get_integrator, ForwardEuler, StormerVerlet, RungeKutta4, Yoshida4
from src.models.interaction_graph import InteractionPhysicsBlock, CouplingFunction


def _dummy_accel(z: torch.Tensor, zd: torch.Tensor) -> torch.Tensor:
    """Simple z'' = -z (harmonic oscillator)."""
    return -z


def test_integrator_euler():
    """Forward Euler: shapes and gradient flow."""
    integrator = get_integrator("euler")
    z = torch.randn(4, 1, 1, requires_grad=True)
    zd = torch.randn(4, 1, 1)
    z_next, zd_next = integrator(z, zd, 0.01, _dummy_accel)
    assert z_next.shape == z.shape and zd_next.shape == zd.shape
    z_next.sum().backward()
    assert z.grad is not None


def test_integrator_stormer_verlet():
    """Störmer-Verlet: shapes and deterministic step."""
    integrator = get_integrator("stormer_verlet")
    z = torch.randn(4, 1, 1)
    zd = torch.randn(4, 1, 1)
    z_next, zd_next = integrator(z, zd, 0.01, _dummy_accel)
    assert z_next.shape == z.shape and zd_next.shape == zd.shape


def test_integrator_rk4():
    """RK4: shapes."""
    integrator = get_integrator("rk4")
    z = torch.randn(4, 1, 1)
    zd = torch.randn(4, 1, 1)
    z_next, zd_next = integrator(z, zd, 0.01, _dummy_accel)
    assert z_next.shape == z.shape and zd_next.shape == zd.shape


def test_integrator_yoshida4():
    """Yoshida 4th-order symplectic: shapes."""
    integrator = get_integrator("yoshida4")
    z = torch.randn(4, 1, 1)
    zd = torch.randn(4, 1, 1)
    z_next, zd_next = integrator(z, zd, 0.01, _dummy_accel)
    assert z_next.shape == z.shape and zd_next.shape == zd.shape


def test_physics_block_n1_all_integrators():
    """InteractionPhysicsBlock N=1 with each integrator."""
    for name in ["euler", "stormer_verlet", "rk4", "yoshida4"]:
        block = InteractionPhysicsBlock(
            num_objects=1,
            d_per_object=1,
            ode_order=2,
            coupling_type="linear",
            integrator_type=name,
        )
        Z = torch.randn(4, 1, 1)
        Zd = torch.randn(4, 1, 1)
        Zdd = block.compute_acceleration(Z, Zd)
        assert Zdd.shape == (4, 1, 1)
        # One step: need ode_order+1 = 3 steps for order 2
        Z_hist = torch.stack([Z * 0.8, Z * 0.9, Z], dim=1)  # [4, 3, 1, 1]
        z_next = block.forward(Z_hist, 0.01)
        assert z_next.shape == (4, 1, 1)


def test_coupling_linear():
    """Linear coupling C_ij = kappa * (z_i - z_j)."""
    cf = CouplingFunction("linear", d=1)
    z_i = torch.tensor([[1.0], [2.0]])
    z_j = torch.tensor([[0.0], [1.0]])
    zd_i = zd_j = torch.zeros(2, 1)
    out = cf(z_i, z_j, zd_i, zd_j)
    assert out.shape == (2, 1)


def test_coupling_contact():
    """Contact (soft repulsion) coupling."""
    cf = CouplingFunction("contact", d=1, R_init=0.5)
    z_i = torch.tensor([[0.0], [0.0]])
    z_j = torch.tensor([[0.2], [0.8]])
    zd_i = zd_j = torch.zeros(2, 1)
    out = cf(z_i, z_j, zd_i, zd_j)
    assert out.shape == (2, 1)


def test_coupling_double_pendulum():
    """Double-pendulum (linearized) coupling."""
    cf = CouplingFunction("double_pendulum", d=1)
    z_i = torch.randn(3, 1)
    z_j = torch.randn(3, 1)
    zd_i = torch.randn(3, 1)
    zd_j = torch.randn(3, 1)
    out = cf(z_i, z_j, zd_i, zd_j)
    assert out.shape == (3, 1)


def test_physics_block_n2_linear():
    """N=2 with linear coupling: coupling term non-zero."""
    block = InteractionPhysicsBlock(
        num_objects=2,
        d_per_object=1,
        ode_order=2,
        coupling_type="linear",
        integrator_type="euler",
    )
    Z = torch.randn(4, 2, 1)
    Zd = torch.randn(4, 2, 1)
    Zdd = block.compute_acceleration(Z, Zd)
    assert Zdd.shape == (4, 2, 1)
    assert len(block.adjacency) == 2  # (0,1), (1,0)


def test_physics_block_n2_contact():
    """N=2 with contact coupling."""
    block = InteractionPhysicsBlock(
        num_objects=2,
        d_per_object=1,
        coupling_type="contact",
        integrator_type="stormer_verlet",
    )
    Z_hist = torch.randn(4, 3, 2, 1)  # batch, ode_order+1, N, d
    z_next = block.forward(Z_hist, 0.01)
    assert z_next.shape == (4, 2, 1)


if __name__ == "__main__":
    test_integrator_euler()
    test_integrator_stormer_verlet()
    test_integrator_rk4()
    test_integrator_yoshida4()
    test_physics_block_n1_all_integrators()
    test_coupling_linear()
    test_coupling_contact()
    test_coupling_double_pendulum()
    test_physics_block_n2_linear()
    test_physics_block_n2_contact()
    print("All integrator and coupling tests passed.")
