"""
Unified model: one code path for N=1 and N>1.
Encoder -> [batch, T, N, d], Physics -> predictions, Loss = L1 + L2.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .encoder_unified import EncoderUnified
from .interaction_graph import InteractionPhysicsBlock
from src.losses.loss import physics_loss, physics_loss_multistep


class UnifiedPhysicsModel(nn.Module):
    """
    Single model for 1 or N objects. No branching on N.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.N = config.get("num_objects", 1)
        self.d = config.get("d_per_object", 1)
        self.ode_order = config.get("ode_order", 2)
        self.dt = config["dt"]
        self.prior_mu = config.get("prior_mu", 0.0)
        self.prior_sigma = config.get("prior_sigma", 1.0)
        self.use_multistep_loss = config.get("use_multistep_loss", False)
        self.multistep_num_steps = config.get("multistep_num_steps", 5)
        self.multistep_weights = config.get("multistep_step_weights", [1.0, 1.0, 0.5, 0.5, 0.25])

        self.encoder = EncoderUnified(
            input_dim=config["input_dim"],
            num_objects=self.N,
            d_per_object=self.d,
            strategy=config.get("encoder_strategy", "holistic"),
            hidden_dims=config.get("hidden_dims", [512, 256]),
        )

        adjacency = config.get("adjacency", None)
        self.physics = InteractionPhysicsBlock(
            num_objects=self.N,
            d_per_object=self.d,
            ode_order=self.ode_order,
            coupling_type=config.get("coupling_type", "linear"),
            adjacency=adjacency,
            integrator_type=config.get("integrator", "stormer_verlet"),
        )

    def encode_sequence(
        self,
        frames: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """frames: [batch, T, C, H, W] -> Z: [batch, T, N, d]."""
        batch, T = frames.shape[0], frames.shape[1]
        z_list = []
        for t in range(T):
            frame_flat = frames[:, t].flatten(1)
            m = masks[:, t] if masks is not None else None
            z_t = self.encoder(frame_flat, m)
            z_list.append(z_t)
        return torch.stack(z_list, dim=1)

    def predict_sequence(self, Z_encoded: torch.Tensor) -> torch.Tensor:
        """Z_encoded: [batch, T, N, d]. Predict next state for each window -> [batch, T - ode_order - 1, N, d]."""
        T = Z_encoded.shape[1]
        preds = []
        for t in range(self.ode_order, T - 1):
            z_hist = Z_encoded[:, t - self.ode_order : t + 1]
            z_pred = self.physics(z_hist, self.dt)
            preds.append(z_pred)
        return torch.stack(preds, dim=1)

    def forward(
        self,
        frames: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        frames: [batch, T, C, H, W]
        Returns: loss, z_encoded, z_predicted, parameters
        """
        Z_enc = self.encode_sequence(frames, masks)
        Z_pred = self.predict_sequence(Z_enc)

        # Z_pred[i] predicts next state after Z_enc[:, ode_order+i], i.e. target Z_enc[:, ode_order+i+1]
        Z_target = Z_enc[:, self.ode_order + 1 :]

        B, Tp, N, d = Z_target.shape
        if self.use_multistep_loss:
            loss = physics_loss_multistep(
                Z_target,
                Z_pred,
                self.prior_mu,
                self.prior_sigma,
                num_steps=self.multistep_num_steps,
                step_weights=self.multistep_weights[: self.multistep_num_steps],
            )
        else:
            z_target_flat = Z_target.reshape(B * Tp, N, d)
            z_pred_flat = Z_pred.reshape(B * Tp, N, d)
            loss = physics_loss(
                z_target_flat, z_pred_flat, self.prior_mu, self.prior_sigma
            )

        params = {}
        for name, p in self.physics.named_parameters():
            if p.numel() == 1:
                params[name] = p.item()
            else:
                params[name] = p.detach()

        return {
            "loss": loss,
            "z_encoded": Z_enc,
            "z_predicted": Z_pred,
            "parameters": params,
        }
