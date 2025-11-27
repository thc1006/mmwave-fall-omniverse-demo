from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fallnet_model import FallNetConfig, ModelFactory, ModelType


def discover_labels(data_dir: Path) -> Dict[str, int]:
    """Discover subdirectories under data_dir as class labels.

    The mapping is sorted alphabetically for reproducibility, but we make a
    best effort to keep "normal" at index 0 if it exists.
    """
    subdirs = [p for p in data_dir.iterdir() if p.is_dir()]
    names = sorted(p.name for p in subdirs)
    if "normal" in names:
        names.remove("normal")
        names.insert(0, "normal")
    return {name: idx for idx, name in enumerate(names)}


def load_dataset(data_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, str]]:
    """Load .npz episodes from class subdirectories under data_dir.

    Each .npz file is expected to contain `data` (frames x features) and `label`
    (string or ignored). Labels are inferred from the directory structure.
    """
    label_to_idx = discover_labels(data_dir)
    idx_to_label = {idx: name for name, idx in label_to_idx.items()}

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for label_name, label_idx in label_to_idx.items():
        subdir = data_dir / label_name
        if not subdir.exists():
            continue
        for npz_path in sorted(subdir.glob("*.npz")):
            arr = np.load(npz_path)
            data = arr["data"]  # [frames, features] or [features]
            if data.ndim == 1:
                data = data[None, :]
            xs.append(data.astype("float32"))
            ys.append(np.full((data.shape[0],), label_idx, dtype="int64"))

    if not xs:
        raise RuntimeError(f"No data found under {data_dir}. Did you run record_fall_data.py?")

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return torch.from_numpy(x), torch.from_numpy(y), idx_to_label


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train(
    data_dir: Path,
    output_path: Path,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    model_type: str = "mlp",
    val_split: float = 0.2,
    patience: int = 10,
) -> None:
    """Train a FallNet model with early stopping and LR scheduling.

    Args:
        data_dir: Directory containing class subdirectories with .npz files.
        output_path: Path to save the trained model.
        epochs: Maximum number of training epochs.
        batch_size: Batch size for training.
        lr: Initial learning rate.
        model_type: Model architecture (mlp, cnn, lstm).
        val_split: Fraction of data to use for validation.
        patience: Early stopping patience (epochs without improvement).
    """
    x, y, idx_to_label = load_dataset(data_dir)

    # Z-score normalization (critical for good performance)
    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True) + 1e-8  # avoid div by zero
    x = (x - x_mean) / x_std

    # Handle both 2D [samples, features] and 3D [samples, frames, features]
    if x.dim() == 2:
        num_samples, features = x.shape
        seq_len = 1
    else:
        num_samples, seq_len, features = x.shape

    # Create config with selected model type
    cfg = FallNetConfig(
        input_dim=features,
        hidden_dim=256,
        num_layers=2,
        num_classes=len(idx_to_label),
        seq_len=seq_len,
        model_type=ModelType(model_type),
        label_map=idx_to_label,
    )

    # Create model using factory
    model = ModelFactory.create(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[train_fallnet] Using model: {model_type.upper()}, device: {device}")
    print(f"[train_fallnet] Input shape: {x.shape}, classes: {len(idx_to_label)}")

    # Train/validation split
    dataset = TensorDataset(x, y)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[train_fallnet] Train samples: {train_size}, Val samples: {val_size}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning rate scheduler: reduce LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=patience)

    # Track best model
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == batch_y).sum().item()

        avg_train_loss = train_loss / train_size
        train_acc = train_correct / train_size

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == batch_y).sum().item()

        avg_val_loss = val_loss / val_size if val_size > 0 else 0.0
        val_acc = val_correct / val_size if val_size > 0 else 0.0

        # Update LR scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[train_fallnet] Epoch {epoch}: "
            f"train_loss={avg_train_loss:.4f}, train_acc={train_acc:.3f}, "
            f"val_loss={avg_val_loss:.4f}, val_acc={val_acc:.3f}, lr={current_lr:.2e}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            print(f"[train_fallnet] New best model (val_loss={best_val_loss:.4f})")

        # Check early stopping
        if early_stopping(avg_val_loss):
            print(f"[train_fallnet] Early stopping at epoch {epoch}")
            break

    # Restore best model and save
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert config to serializable dict
    config_dict = asdict(cfg)
    config_dict["model_type"] = cfg.model_type.value  # Convert enum to string

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config_dict,
        "norm_mean": x_mean.cpu(),
        "norm_std": x_std.cpu(),
        "idx_to_label": idx_to_label,
    }, output_path)
    print(f"[train_fallnet] Saved best model to {output_path}")
    print(f"[train_fallnet] Final val_loss={best_val_loss:.4f}")
    print(f"[train_fallnet] Label map: {idx_to_label}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fall detection model from RTX Radar data.")
    parser.add_argument("--data-dir", type=str, default="ml/data", help="Directory with class subdirectories.")
    parser.add_argument("--output", type=str, default="ml/fallnet.pt", help="Output path for the trained model.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument(
        "--model-type",
        type=str,
        default="mlp",
        choices=["mlp", "cnn", "lstm"],
        help="Model architecture: mlp (simple MLP), cnn (1D-CNN), lstm (BiLSTM).",
    )
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_type=args.model_type,
        val_split=args.val_split,
        patience=args.patience,
    )
