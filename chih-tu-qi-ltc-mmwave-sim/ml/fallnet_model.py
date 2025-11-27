"""FallNet - Neural network models for mmWave fall detection.

This module provides multiple architectures for fall detection:
- FallNetMLP: Simple multi-layer perceptron
- FallNetCNN: 1D Convolutional neural network
- FallNetLSTM: LSTM-based sequential model
- FallNet: Factory function to create models

All models support 4-class classification:
- 0: normal (正常)
- 1: fall (跌倒)
- 2: rehab_bad_posture (復健姿勢不良)
- 3: chest_abnormal (胸腔異常)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class FallNetMLP(nn.Module):
    """Multi-layer perceptron for fall detection.

    Takes averaged feature vectors and classifies into fall categories.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list[int] | None = None,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        """Initialize FallNetMLP.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch, frames, features] or [batch, features]

        Returns:
            Logits of shape [batch, num_classes]
        """
        # Handle 3D input by averaging over time
        if x.dim() == 3:
            x = x.mean(dim=1)  # [batch, frames, features] -> [batch, features]

        return self.net(x)


class FallNetCNN(nn.Module):
    """1D CNN for fall detection with temporal convolutions.

    Processes sequences of radar frames with convolutional layers.
    """

    def __init__(
        self,
        input_dim: int = 256,
        seq_len: int = 128,
        num_classes: int = 4,
        channels: list[int] | None = None,
        dropout: float = 0.3,
    ):
        """Initialize FallNetCNN.

        Args:
            input_dim: Input feature dimension per frame
            seq_len: Expected sequence length
            num_classes: Number of output classes
            channels: List of channel dimensions for conv layers
            dropout: Dropout probability
        """
        super().__init__()

        if channels is None:
            channels = [64, 128, 256]

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes

        # Build convolutional layers
        conv_layers = []
        in_channels = input_dim

        for out_channels in channels:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # Calculate output size after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, input_dim, seq_len)
            conv_out = self.conv(dummy)
            flatten_size = conv_out.numel()

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch, frames, features]

        Returns:
            Logits of shape [batch, num_classes]
        """
        # Handle 2D input by adding time dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, features] -> [batch, 1, features]

        # Transpose to [batch, features, frames] for Conv1d
        x = x.transpose(1, 2)

        # Pad or truncate to expected sequence length
        if x.size(2) < self.seq_len:
            pad_size = self.seq_len - x.size(2)
            x = F.pad(x, (0, pad_size))
        elif x.size(2) > self.seq_len:
            x = x[:, :, :self.seq_len]

        # Convolutional layers
        x = self.conv(x)

        # Flatten and classify
        x = x.flatten(1)
        return self.classifier(x)


class FallNetLSTM(nn.Module):
    """LSTM-based model for fall detection.

    Processes temporal sequences with bidirectional LSTM layers.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """Initialize FallNetLSTM.

        Args:
            input_dim: Input feature dimension per frame
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Output dimension
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention mechanism.

        Args:
            x: Input tensor of shape [batch, frames, features]

        Returns:
            Logits of shape [batch, num_classes]
        """
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, features] -> [batch, 1, features]

        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]

        # Attention weights
        attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden*2]

        # Classify
        return self.classifier(context)


class FallNet(nn.Module):
    """Factory wrapper for FallNet models.

    Provides backward compatibility and easy model selection.
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_classes: int = 4,
        model_type: Literal["mlp", "cnn", "lstm"] = "mlp",
        **kwargs,
    ):
        """Initialize FallNet.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            model_type: Model architecture to use
            **kwargs: Additional arguments passed to the model
        """
        super().__init__()

        self.model_type = model_type
        self.input_dim = input_dim
        self.num_classes = num_classes

        if model_type == "mlp":
            self.model = FallNetMLP(
                input_dim=input_dim,
                num_classes=num_classes,
                **kwargs,
            )
        elif model_type == "cnn":
            self.model = FallNetCNN(
                input_dim=input_dim,
                num_classes=num_classes,
                **kwargs,
            )
        elif model_type == "lstm":
            self.model = FallNetLSTM(
                input_dim=input_dim,
                num_classes=num_classes,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Logits of shape [batch, num_classes]
        """
        return self.model(x)


def create_model(
    model_type: str = "mlp",
    input_dim: int = 256,
    num_classes: int = 4,
    **kwargs,
) -> nn.Module:
    """Create a FallNet model.

    Args:
        model_type: One of "mlp", "cnn", "lstm"
        input_dim: Input feature dimension
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized model
    """
    return FallNet(
        input_dim=input_dim,
        num_classes=num_classes,
        model_type=model_type,
        **kwargs,
    )


# Label mapping
LABEL_MAP = {
    0: "normal",
    1: "fall",
    2: "rehab_bad_posture",
    3: "chest_abnormal",
}

LABEL_MAP_REVERSE = {v: k for k, v in LABEL_MAP.items()}
