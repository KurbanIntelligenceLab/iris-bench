"""
Unit test: N=1 unified model has correct shapes and no branching.
Design invariant: [batch, N, d] with N=1 is [batch, 1, d], never [batch, d].
"""

import sys
import os
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.unified_model import UnifiedPhysicsModel
from src.models.encoder_unified import EncoderUnified
from src.models.interaction_graph import InteractionPhysicsBlock
from src.losses.loss import physics_loss


def test_n1_shapes():
    """Unified model with N=1: all tensors [batch, 1, d]."""
    config = {
        "input_dim": 5600,
        "num_objects": 1,
        "d_per_object": 1,
        "ode_order": 2,
        "dt": 0.01,
        "integrator": "euler",
        "encoder_strategy": "holistic",
        "coupling_type": "linear",
    }
    model = UnifiedPhysicsModel(config)
    batch, T = 2, 10
    frames = torch.randn(batch, T, 1, 56, 100)

    out = model(frames)
    assert out["z_encoded"].shape == (batch, T, 1, 1), out["z_encoded"].shape
    # T - ode_order - 1 predictions
    assert out["z_predicted"].shape == (batch, T - 2 - 1, 1, 1), out["z_predicted"].shape
    assert out["loss"].dim() == 0

    # Encoder alone
    x = frames[:, 0].flatten(1)
    z = model.encoder(x)
    assert z.shape == (batch, 1, 1), z.shape


def test_physics_block_n1_zero_edges():
    """InteractionPhysicsBlock with N=1 has empty adjacency."""
    block = InteractionPhysicsBlock(
        num_objects=1,
        d_per_object=1,
        ode_order=2,
        coupling_type="linear",
        integrator_type="euler",
    )
    assert block.adjacency == []
    assert len(block.couplings) == 0

    Z = torch.randn(4, 1, 1)
    Zd = torch.randn(4, 1, 1)
    Zdd = block.compute_acceleration(Z, Zd)
    assert Zdd.shape == (4, 1, 1)


def test_loss_shape_agnostic():
    """physics_loss works for [B,1,1] and [B,2,1] without branching."""
    z1 = torch.randn(10, 1, 1)
    z2 = torch.randn(10, 1, 1)
    loss1 = physics_loss(z1, z2)

    z3 = torch.randn(10, 2, 1)
    z4 = torch.randn(10, 2, 1)
    loss2 = physics_loss(z3, z4)

    assert loss1.dim() == 0 and loss2.dim() == 0


def test_n2_shapes():
    """Unified model with N=2: all tensors [batch, 2, d]."""
    config = {
        "input_dim": 5600,
        "num_objects": 2,
        "d_per_object": 1,
        "ode_order": 2,
        "dt": 0.01,
        "integrator": "stormer_verlet",
        "encoder_strategy": "holistic",
        "coupling_type": "linear",
    }
    model = UnifiedPhysicsModel(config)
    batch, T = 2, 10
    frames = torch.randn(batch, T, 1, 56, 100)

    out = model(frames)
    assert out["z_encoded"].shape == (batch, T, 2, 1)
    assert out["z_predicted"].shape == (batch, T - 3, 2, 1)


if __name__ == "__main__":
    test_n1_shapes()
    test_physics_block_n1_zero_edges()
    test_loss_shape_agnostic()
    test_n2_shapes()
    print("All tests passed.")
