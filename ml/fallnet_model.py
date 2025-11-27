from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

import torch
from torch import nn


class ModelType(str, Enum):
    """Supported model architectures."""
    MLP = "mlp"
    CNN = "cnn"
    LSTM = "lstm"


@dataclass
class FallNetConfig:
    """Configuration for all FallNet model variants.

    Attributes:
        input_dim: Feature dimension (e.g., 6 for IMU, 256 for radar).
        hidden_dim: Hidden layer dimension.
        num_layers: Number of hidden layers (MLP) or conv/LSTM layers.
        dropout: Dropout rate.
        num_classes: Number of output classes.
        seq_len: Expected sequence length for CNN/LSTM (optional).
        model_type: Architecture type (mlp, cnn, lstm).
        label_map: Mapping from class index to label name.
    """
    input_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    num_classes: int = 4
    seq_len: int = 64  # Expected sequence length for CNN/LSTM
    model_type: ModelType = ModelType.MLP
    # Multi-class support; default mapping:
    # 0 = normal, 1 = fall, 2 = rehab_bad_posture, 3 = chest_abnormal
    label_map: Dict[int, str] = field(default_factory=lambda: {
        0: "normal",
        1: "fall",
        2: "rehab_bad_posture",
        3: "chest_abnormal",
    })


class FallNet(nn.Module):
    """Simple MLP classifier for radar feature vectors.

    This model supports multi-class classification. It expects inputs of
    shape [batch, frames, features] or [batch, features]. For sequence
    inputs, we apply mean pooling over the time dimension.
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


class FallNetCNN(nn.Module):
    """1D-CNN for temporal feature extraction from sequences.

    Expects input shape [B, T, F] where T is sequence length and F is features.
    Uses 1D convolutions along the time dimension to capture temporal patterns.
    """

    def __init__(self, cfg: FallNetConfig):
        super().__init__()
        self.cfg = cfg

        # Conv layers: input channels = features, conv over time
        conv_layers = []
        in_channels = cfg.input_dim
        out_channels = cfg.hidden_dim

        for i in range(cfg.num_layers):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            ])
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.classifier = nn.Linear(out_channels, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> transpose to [B, F, T] for Conv1d
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, F] -> [B, 1, F], treat as single timestep
        x = x.transpose(1, 2)  # [B, T, F] -> [B, F, T]

        x = self.conv(x)  # [B, hidden_dim, T]
        x = self.pool(x).squeeze(-1)  # [B, hidden_dim]
        return self.classifier(x)


class FallNetLSTM(nn.Module):
    """Bidirectional LSTM for sequence modeling.

    Expects input shape [B, T, F] where T is sequence length and F is features.
    Uses bidirectional LSTM to capture temporal dependencies.
    """

    def __init__(self, cfg: FallNetConfig):
        super().__init__()
        self.cfg = cfg

        self.lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        # Bidirectional doubles the hidden size
        self.classifier = nn.Linear(cfg.hidden_dim * 2, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] or [B, F]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, F] -> [B, 1, F]

        # LSTM output: [B, T, 2*hidden_dim]
        lstm_out, _ = self.lstm(x)
        # Use last timestep output
        last_out = lstm_out[:, -1, :]  # [B, 2*hidden_dim]
        last_out = self.dropout(last_out)
        return self.classifier(last_out)


class ModelFactory:
    """Factory for creating FallNet model variants."""

    _registry: Dict[ModelType, type] = {
        ModelType.MLP: FallNet,
        ModelType.CNN: FallNetCNN,
        ModelType.LSTM: FallNetLSTM,
    }

    @classmethod
    def create(cls, cfg: FallNetConfig) -> nn.Module:
        """Create a model based on the config's model_type."""
        model_cls = cls._registry.get(cfg.model_type)
        if model_cls is None:
            raise ValueError(f"Unknown model type: {cfg.model_type}")
        return model_cls(cfg)

    @classmethod
    def register(cls, model_type: ModelType, model_cls: type) -> None:
        """Register a custom model class."""
        cls._registry[model_type] = model_cls
