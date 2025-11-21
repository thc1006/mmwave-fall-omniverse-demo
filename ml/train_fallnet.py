from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim

from .fallnet_model import FallNet, FallNetConfig


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


def train(
    data_dir: Path,
    output_path: Path,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> None:
    x, y, idx_to_label = load_dataset(data_dir)
    num_samples, frames, features = x.shape

    cfg = FallNetConfig(input_dim=features, num_classes=len(idx_to_label), label_map=idx_to_label)
    model = FallNet(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()

        avg_loss = total_loss / num_samples
        acc = correct / num_samples
        print(f"[train_fallnet] Epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.3f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__}, output_path)
    print(f"[train_fallnet] Saved model to {output_path}")
    print(f"[train_fallnet] Label map: {idx_to_label}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fall detection model from RTX Radar data.")
    parser.add_argument("--data-dir", type=str, default="ml/data", help="Directory with class subdirectories.")
    parser.add_argument("--output", type=str, default="ml/fallnet.pt", help="Output path for the trained model.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(Path(args.data_dir), Path(args.output), args.epochs, args.batch_size, args.lr)
