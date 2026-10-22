"""
Unified encoder: ALWAYS output [batch, N, d].
N=1 with strategy='holistic' = original paper's MLP + view(batch, 1, d).
"""

import torch
import torch.nn as nn
from typing import List, Optional


class EncoderUnified(nn.Module):
    """
    One encoder for any N. Output shape always [batch, N, d_per_object].
    """

    def __init__(
        self,
        input_dim: int,
        num_objects: int = 1,
        d_per_object: int = 1,
        strategy: str = "holistic",
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.N = num_objects
        self.d = d_per_object
        self.strategy = strategy
        hidden_dims = hidden_dims or [512, 256]

        total_output_dim = num_objects * d_per_object

        if strategy == "holistic":
            self.mlp = self._build_mlp(input_dim, total_output_dim, hidden_dims)
            self.backbone = None
            self.heads = None
            self.shared_encoder = None

        elif strategy == "per_object":
            self.mlp = None
            self.backbone = self._build_mlp(input_dim, hidden_dims[-1], hidden_dims[:-1])
            self.heads = nn.ModuleList(
                [nn.Linear(hidden_dims[-1], d_per_object) for _ in range(num_objects)]
            )
            self.shared_encoder = None

        elif strategy == "shared":
            self.mlp = None
            self.backbone = None
            self.heads = None
            self.shared_encoder = self._build_mlp(input_dim, d_per_object, hidden_dims)

        else:
            # default holistic
            self.mlp = self._build_mlp(input_dim, total_output_dim, hidden_dims)
            self.backbone = None
            self.heads = None
            self.shared_encoder = None

    def _build_mlp(
        self, in_dim: int, out_dim: int, hidden: List[int]
    ) -> nn.Sequential:
        layers = []
        prev = in_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [batch, input_dim] (flattened frame)
        masks: optional [batch, N, input_dim]
        Returns: Z [batch, N, d]
        """
        batch = x.shape[0]

        if self.strategy == "holistic":
            z_flat = self.mlp(x)
            Z = z_flat.view(batch, self.N, self.d)

        elif self.strategy == "per_object":
            assert masks is not None, "per_object strategy requires masks"
            features = self.backbone(x)
            Z = torch.stack([head(features) for head in self.heads], dim=1)

        elif self.strategy == "shared":
            assert masks is not None, "shared strategy requires masks"
            z_list = []
            for i in range(self.N):
                masked_input = x * masks[:, i]
                z_i = self.shared_encoder(masked_input)
                z_list.append(z_i)
            Z = torch.stack(z_list, dim=1)

        else:
            z_flat = self.mlp(x)
            Z = z_flat.view(batch, self.N, self.d)

        return Z
