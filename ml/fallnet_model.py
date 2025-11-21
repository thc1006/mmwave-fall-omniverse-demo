from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FallNetConfig:
    input_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    num_classes: int = 2  # 0 = normal, 1 = fall


class FallNet(nn.Module):
    """Simple MLP classifier for radar feature vectors.

    This model expects inputs of shape [batch, frames, features] or [batch, features].
    For sequence inputs, we apply mean pooling over the time dimension.
    """

    def __init__(self, cfg: FallNetConfig):
        super().__init__()
        self.cfg = cfg

        layers = []
        in_dim = cfg.input_dim
        for _ in range(cfg.num_layers):
            layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.dropout))
            in_dim = cfg.hidden_dim
        layers.append(nn.Linear(in_dim, cfg.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] or [B, F]
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)
