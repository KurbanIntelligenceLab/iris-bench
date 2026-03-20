"""
Unified physics loss. Works identically for N=1 and N>1.
For N=1: exactly the original paper's Eq. 13 (MSE + KL).
"""

import torch
from typing import Optional


def physics_loss(
    Z_encoded: torch.Tensor,
    Z_predicted: torch.Tensor,
    prior_mu: float = 0.0,
    prior_sigma: float = 1.0,
) -> torch.Tensor:
    """
    Unified loss. No branching on N.

    Args:
        Z_encoded: [batch, N, d]
        Z_predicted: [batch, N, d]
        prior_mu, prior_sigma: target N(prior_mu, prior_sigma^2)

    Returns:
        scalar: L1 + L2
    """
    # L1: prediction error
    L1 = torch.mean((Z_encoded - Z_predicted) ** 2)

    # L2: KL over batch (flatten N and d)
    z_flat = Z_encoded.reshape(Z_encoded.shape[0], -1)  # [batch, N*d]
    total_dim = z_flat.shape[1]

    mu_z = z_flat.mean(dim=0)
    var_z = z_flat.var(dim=0) + 1e-8

    prior_var = prior_sigma ** 2
    # KL(N(mu_z, var_z) || N(prior_mu, prior_var)); paper Eq. 12-13 when prior N(0,1)
    L2 = -0.5 * torch.mean(
        1
        + torch.log(var_z.clamp(min=1e-8))
        - ((mu_z - prior_mu) ** 2) / prior_var
        - var_z / prior_var
    )

    return L1 + L2


def physics_loss_multistep(
    Z_encoded: torch.Tensor,
    Z_predicted: torch.Tensor,
    prior_mu: float = 0.0,
    prior_sigma: float = 1.0,
    num_steps: int = 5,
    step_weights: Optional[list] = None,
) -> torch.Tensor:
    """
    Multi-step physics loss: weighted MSE at horizons 0..num_steps-1 plus KL.
    Encourages long-horizon consistency and can improve parameter identifiability.

    Args:
        Z_encoded: [batch, T, N, d] target latent sequence
        Z_predicted: [batch, T, N, d] predicted sequence (same T)
        prior_mu, prior_sigma: for KL on encoder latents
        num_steps: number of horizon steps (1..num_steps)
        step_weights: length num_steps, or None for uniform

    Returns:
        scalar loss
    """
    if Z_encoded.dim() != 4 or Z_predicted.dim() != 4:
        raise ValueError("physics_loss_multistep expects [B, T, N, d] for both tensors")
    B, T, N, d = Z_encoded.shape
    K = min(num_steps, T)
    if K < 1:
        return physics_loss(
            Z_encoded.reshape(-1, N, d),
            Z_predicted.reshape(-1, N, d),
            prior_mu,
            prior_sigma,
        )
    if step_weights is None:
        step_weights = [1.0] * K
    step_weights = step_weights[:K]
    total_w = sum(step_weights)
    mse_sum = 0.0
    for k in range(K):
        # Horizon k: compare from step k to end
        z_enc_k = Z_encoded[:, k:, :, :].reshape(-1, N, d)
        z_pred_k = Z_predicted[:, k:, :, :].reshape(-1, N, d)
        mse_sum = mse_sum + step_weights[k] * torch.mean((z_enc_k - z_pred_k) ** 2)
    L1 = mse_sum / total_w
    z_flat = Z_encoded.reshape(B * T * N, d)
    mu_z = z_flat.mean(dim=0)
    var_z = z_flat.var(dim=0) + 1e-8
    prior_var = prior_sigma ** 2
    L2 = -0.5 * torch.mean(
        1
        + torch.log(var_z.clamp(min=1e-8))
        - ((mu_z - prior_mu) ** 2) / prior_var
        - var_z / prior_var
    )
    return L1 + L2
