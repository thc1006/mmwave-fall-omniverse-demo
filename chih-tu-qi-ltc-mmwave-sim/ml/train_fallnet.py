#!/usr/bin/env python3
"""Train FallNet model for mmWave fall detection.

This script trains a neural network for fall detection using
recorded radar data from Isaac Sim.

Usage:
    python -m ml.train_fallnet --data-dir ml/data --output ml/fallnet.pt

    # Train specific model type
    python -m ml.train_fallnet --model-type lstm --epochs 100

    # Quick training
    python -m ml.train_fallnet --epochs 10 --patience 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from .fallnet_model import FallNet, LABEL_MAP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RadarDataset(Dataset):
    """Dataset for radar fall detection data."""

    def __init__(self, data_dir: Path, seq_len: int = 128):
        """Initialize dataset.

        Args:
            data_dir: Directory containing .npz files
            seq_len: Expected sequence length
        """
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.samples: list[tuple[np.ndarray, int]] = []

        self._load_data()

    def _load_data(self) -> None:
        """Load all .npz files from data directory."""
        for npz_path in self.data_dir.rglob("*.npz"):
            try:
                data = np.load(npz_path, allow_pickle=True)
                features = data["data"].astype(np.float32)
                label = int(data["label"])
                self.samples.append((features, label))
            except Exception as e:
                logger.warning(f"Failed to load {npz_path}: {e}")

        logger.info(f"Loaded {len(self.samples)} samples from {self.data_dir}")

        # Log class distribution
        labels = [s[1] for s in self.samples]
        for label_idx, label_name in LABEL_MAP.items():
            count = labels.count(label_idx)
            logger.info(f"  Class {label_idx} ({label_name}): {count} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        features, label = self.samples[idx]

        # Pad or truncate to seq_len
        if len(features) < self.seq_len:
            padding = np.zeros((self.seq_len - len(features), features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, padding], axis=0)
        elif len(features) > self.seq_len:
            features = features[:self.seq_len]

        return torch.from_numpy(features), label


class SyntheticRadarDataset(Dataset):
    """Synthetic dataset for testing when no recorded data is available."""

    def __init__(
        self,
        num_samples: int = 1000,
        seq_len: int = 128,
        feature_dim: int = 256,
        num_classes: int = 4,
    ):
        """Initialize synthetic dataset.

        Args:
            num_samples: Number of samples to generate
            seq_len: Sequence length
            feature_dim: Feature dimension
            num_classes: Number of classes
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.samples = self._generate_samples()

    def _generate_samples(self) -> list[tuple[torch.Tensor, int]]:
        """Generate synthetic samples."""
        samples = []
        samples_per_class = self.num_samples // self.num_classes

        for label in range(self.num_classes):
            for _ in range(samples_per_class):
                # Generate class-specific patterns
                data = np.random.randn(self.seq_len, self.feature_dim).astype(np.float32) * 0.1

                if label == 0:  # normal
                    # Stable pattern
                    data[:, 30:50] += 0.5
                elif label == 1:  # fall
                    # High velocity signature
                    t = np.linspace(0, 1, self.seq_len)
                    fall_pattern = np.exp(-5 * (t - 0.3) ** 2) * 3
                    data[:, 50:80] += fall_pattern[:, None]
                elif label == 2:  # rehab_bad_posture
                    # Irregular rhythmic pattern
                    t = np.linspace(0, 4 * np.pi, self.seq_len)
                    irregular = np.sin(t) + 0.3 * np.sin(3 * t)
                    data[:, 20:40] += irregular[:, None] * 0.8
                elif label == 3:  # chest_abnormal
                    # Irregular micro-motion
                    t = np.linspace(0, 10 * np.pi, self.seq_len)
                    breathing = np.sin(t * 0.7) + 0.5 * np.random.randn(self.seq_len)
                    data[:, 10:20] += breathing[:, None] * 0.5

                samples.append((torch.from_numpy(data), label))

        np.random.shuffle(samples)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.samples[idx]


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch.

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    """Evaluate model.

    Returns:
        Tuple of (average loss, accuracy, per-class accuracy dict)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # Per-class tracking
    class_correct = {k: 0 for k in LABEL_MAP}
    class_total = {k: 0 for k in LABEL_MAP}

    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Per-class accuracy
            for i in range(len(targets)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == targets[i]:
                    class_correct[label] += 1

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    per_class_acc = {
        LABEL_MAP[k]: 100.0 * class_correct[k] / max(class_total[k], 1)
        for k in LABEL_MAP
    }

    return avg_loss, accuracy, per_class_acc


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    patience: int = 15,
    lr: float = 0.001,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Train the model.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Maximum number of epochs
        patience: Early stopping patience
        lr: Learning rate
        output_path: Path to save best model

    Returns:
        Training history dict
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    logger.info(f"Training for up to {epochs} epochs with patience {patience}")
    logger.info(f"Learning rate: {lr}")

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate
        val_loss, val_acc, per_class_acc = evaluate(
            model, val_loader, criterion, device
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            no_improve = 0

            # Save best model
            if output_path is not None:
                save_checkpoint(model, output_path, epoch, val_acc, history)
                logger.info(f"  Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    logger.info(f"\nBest validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

    return history


def save_checkpoint(
    model: nn.Module,
    path: Path,
    epoch: int,
    val_acc: float,
    history: dict[str, Any],
) -> None:
    """Save model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
        "history": history,
        "config": {
            "input_dim": getattr(model, "input_dim", 256),
            "num_classes": getattr(model, "num_classes", 4),
            "model_type": getattr(model, "model_type", "mlp"),
        },
    }

    torch.save(checkpoint, path)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train FallNet model for fall detection."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("ml/data"),
        help="Directory containing training data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/fallnet.pt"),
        help="Output model path",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="lstm",
        choices=["mlp", "cnn", "lstm"],
        help="Model architecture",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (for testing)",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=2000,
        help="Number of synthetic samples",
    )

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataset
    if args.synthetic or not args.data_dir.exists():
        logger.info("Using synthetic dataset")
        dataset = SyntheticRadarDataset(
            num_samples=args.synthetic_samples,
            seq_len=args.seq_len,
        )
    else:
        dataset = RadarDataset(args.data_dir, seq_len=args.seq_len)

    if len(dataset) == 0:
        logger.error("No data found. Using synthetic data.")
        dataset = SyntheticRadarDataset(
            num_samples=args.synthetic_samples,
            seq_len=args.seq_len,
        )

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Create model
    model_kwargs = {"input_dim": 256, "num_classes": 4, "model_type": args.model_type}
    if args.model_type == "cnn":
        model_kwargs["seq_len"] = args.seq_len
    model = FallNet(**model_kwargs)
    model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Trainable parameters: {num_params:,}")

    # Train
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("=" * 60)

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        output_path=args.output,
    )

    logger.info("Training complete!")
    logger.info(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
