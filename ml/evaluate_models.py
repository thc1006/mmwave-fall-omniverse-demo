"""Model evaluation script for fall detection models.

This script evaluates all trained models (MLP, CNN, LSTM) and generates
comprehensive metrics including accuracy, precision, recall, F1-score,
confusion matrices, and ROC curve data.

Usage:
    python -m ml.evaluate_models
    python -m ml.evaluate_models --save-results
    python -m ml.evaluate_models --data-dir ml/data --output ml/evaluation_results.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .fallnet_model import FallNetConfig, ModelFactory, ModelType


@dataclass
class ClassMetrics:
    """Metrics for a single class."""
    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass
class ModelEvaluationResult:
    """Evaluation results for a single model."""
    model_name: str
    model_type: str
    accuracy: float
    class_metrics: Dict[str, ClassMetrics]
    confusion_matrix: List[List[int]]
    roc_data: Optional[Dict[str, Any]] = None
    macro_avg: Optional[Dict[str, float]] = None
    weighted_avg: Optional[Dict[str, float]] = None


@dataclass
class EvaluationSummary:
    """Summary of all model evaluations."""
    models: List[ModelEvaluationResult] = field(default_factory=list)
    best_model: Optional[str] = None
    comparison_table: Optional[str] = None


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


def load_test_data(data_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, str]]:
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


def load_model_checkpoint(
    checkpoint_path: Path,
) -> Tuple[torch.nn.Module, FallNetConfig, torch.Tensor, torch.Tensor, Dict[int, str]]:
    """Load a model checkpoint and return the model, config, and normalization params.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.

    Returns:
        Tuple of (model, config, norm_mean, norm_std, idx_to_label)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config_dict = checkpoint["config"]
    # Convert model_type string back to enum
    if isinstance(config_dict.get("model_type"), str):
        config_dict["model_type"] = ModelType(config_dict["model_type"])

    cfg = FallNetConfig(**config_dict)
    model = ModelFactory.create(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    norm_mean = checkpoint["norm_mean"]
    norm_std = checkpoint["norm_std"]
    idx_to_label = checkpoint["idx_to_label"]

    return model, cfg, norm_mean, norm_std, idx_to_label


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    """Compute confusion matrix from predictions.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape [num_classes, num_classes].
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    return cm


def compute_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_idx: int
) -> ClassMetrics:
    """Compute precision, recall, and F1-score for a single class.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_idx: Index of the class to compute metrics for.

    Returns:
        ClassMetrics dataclass with precision, recall, f1_score, and support.
    """
    # True positives: predicted as class_idx and actually class_idx
    tp = np.sum((y_pred == class_idx) & (y_true == class_idx))
    # False positives: predicted as class_idx but not actually class_idx
    fp = np.sum((y_pred == class_idx) & (y_true != class_idx))
    # False negatives: not predicted as class_idx but actually class_idx
    fn = np.sum((y_pred != class_idx) & (y_true == class_idx))
    # Support: total number of actual instances of class_idx
    support = int(np.sum(y_true == class_idx))

    # Compute metrics with zero-division handling
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ClassMetrics(
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1_score),
        support=support,
    )


def compute_roc_data(
    y_true: np.ndarray, y_probs: np.ndarray, num_classes: int
) -> Dict[str, Any]:
    """Compute ROC curve data for each class (one-vs-rest).

    Args:
        y_true: Ground truth labels.
        y_probs: Predicted probabilities of shape [N, num_classes].
        num_classes: Number of classes.

    Returns:
        Dictionary with ROC curve data for each class.
    """
    roc_data: Dict[str, Any] = {}

    for class_idx in range(num_classes):
        # Binary labels for this class (one-vs-rest)
        y_binary = (y_true == class_idx).astype(np.int64)
        scores = y_probs[:, class_idx]

        # Sort by scores descending
        sorted_indices = np.argsort(-scores)
        y_sorted = y_binary[sorted_indices]
        scores_sorted = scores[sorted_indices]

        # Compute TPR and FPR at different thresholds
        total_positives = np.sum(y_binary)
        total_negatives = len(y_binary) - total_positives

        if total_positives == 0 or total_negatives == 0:
            # Cannot compute ROC if there are no positives or negatives
            roc_data[f"class_{class_idx}"] = {
                "fpr": [0.0, 1.0],
                "tpr": [0.0, 1.0],
                "thresholds": [1.0, 0.0],
                "auc": 0.5,
            }
            continue

        # Use unique thresholds
        unique_thresholds = np.unique(scores_sorted)
        # Add endpoints
        thresholds = np.concatenate([[scores_sorted[0] + 1e-6], unique_thresholds[::-1], [0.0]])

        fpr_list = []
        tpr_list = []

        for threshold in thresholds:
            # Predictions at this threshold
            y_pred_binary = (scores >= threshold).astype(np.int64)

            tp = np.sum((y_pred_binary == 1) & (y_binary == 1))
            fp = np.sum((y_pred_binary == 1) & (y_binary == 0))

            tpr = tp / total_positives
            fpr = fp / total_negatives

            tpr_list.append(float(tpr))
            fpr_list.append(float(fpr))

        # Compute AUC using trapezoidal rule
        fpr_arr = np.array(fpr_list)
        tpr_arr = np.array(tpr_list)
        # Sort by FPR for proper AUC computation
        sorted_idx = np.argsort(fpr_arr)
        fpr_arr = fpr_arr[sorted_idx]
        tpr_arr = tpr_arr[sorted_idx]
        # Use trapezoid (numpy >= 2.0) or fall back to trapz
        try:
            auc = float(np.trapezoid(tpr_arr, fpr_arr))
        except AttributeError:
            auc = float(np.trapz(tpr_arr, fpr_arr))

        roc_data[f"class_{class_idx}"] = {
            "fpr": fpr_list,
            "tpr": tpr_list,
            "thresholds": [float(t) for t in thresholds],
            "auc": auc,
        }

    return roc_data


def evaluate_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    idx_to_label: Dict[int, str],
    model_name: str,
    model_type: str,
    batch_size: int = 64,
) -> ModelEvaluationResult:
    """Evaluate a single model on the test data.

    Args:
        model: The model to evaluate.
        x: Input features tensor.
        y: Ground truth labels tensor.
        norm_mean: Normalization mean from training.
        norm_std: Normalization std from training.
        idx_to_label: Mapping from class index to label name.
        model_name: Name of the model for reporting.
        model_type: Type of model (mlp, cnn, lstm).
        batch_size: Batch size for inference.

    Returns:
        ModelEvaluationResult with all computed metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Apply z-score normalization using saved parameters
    x_normalized = (x - norm_mean) / norm_std

    # Create dataloader
    dataset = TensorDataset(x_normalized, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Collect predictions
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch_y.numpy())

    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    num_classes = len(idx_to_label)

    # Compute accuracy
    accuracy = float(np.mean(y_pred == y_true))

    # Compute confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)

    # Compute per-class metrics
    class_metrics: Dict[str, ClassMetrics] = {}
    for class_idx, class_name in idx_to_label.items():
        metrics = compute_class_metrics(y_true, y_pred, class_idx)
        class_metrics[class_name] = metrics

    # Compute ROC data
    roc_data = compute_roc_data(y_true, y_probs, num_classes)

    # Compute macro and weighted averages
    precisions = [m.precision for m in class_metrics.values()]
    recalls = [m.recall for m in class_metrics.values()]
    f1s = [m.f1_score for m in class_metrics.values()]
    supports = [m.support for m in class_metrics.values()]
    total_support = sum(supports)

    macro_avg = {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1_score": float(np.mean(f1s)),
    }

    weighted_avg = {
        "precision": float(np.average(precisions, weights=supports)) if total_support > 0 else 0.0,
        "recall": float(np.average(recalls, weights=supports)) if total_support > 0 else 0.0,
        "f1_score": float(np.average(f1s, weights=supports)) if total_support > 0 else 0.0,
    }

    return ModelEvaluationResult(
        model_name=model_name,
        model_type=model_type,
        accuracy=accuracy,
        class_metrics=class_metrics,
        confusion_matrix=cm.tolist(),
        roc_data=roc_data,
        macro_avg=macro_avg,
        weighted_avg=weighted_avg,
    )


def format_confusion_matrix(cm: List[List[int]], labels: List[str]) -> str:
    """Format confusion matrix as a readable string.

    Args:
        cm: Confusion matrix as list of lists.
        labels: Class labels.

    Returns:
        Formatted string representation.
    """
    # Determine column widths
    max_label_len = max(len(label) for label in labels)
    max_val_len = max(len(str(val)) for row in cm for val in row)
    col_width = max(max_label_len, max_val_len) + 2

    lines = []
    # Header
    header = " " * (max_label_len + 2) + "Predicted".center(col_width * len(labels))
    lines.append(header)
    header2 = " " * (max_label_len + 2) + "".join(label.center(col_width) for label in labels)
    lines.append(header2)
    lines.append("-" * (max_label_len + 2 + col_width * len(labels)))

    # Rows
    for i, (row, label) in enumerate(zip(cm, labels)):
        prefix = "Actual " if i == len(labels) // 2 else "       "
        row_str = f"{prefix}{label:<{max_label_len}} " + "".join(str(val).center(col_width) for val in row)
        lines.append(row_str)

    return "\n".join(lines)


def format_comparison_table(results: List[ModelEvaluationResult]) -> str:
    """Create a comparison table of all models.

    Args:
        results: List of evaluation results.

    Returns:
        Formatted comparison table string.
    """
    lines = []
    lines.append("=" * 100)
    lines.append("MODEL COMPARISON TABLE")
    lines.append("=" * 100)
    lines.append("")

    # Header
    header = f"{'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Fall AUC':>10}"
    lines.append(header)
    lines.append("-" * 65)

    for result in results:
        # Get fall class AUC if available
        fall_auc = "N/A"
        if result.roc_data:
            for key, data in result.roc_data.items():
                # Find the fall class (class_1 typically)
                if "class_1" in key:
                    fall_auc = f"{data.get('auc', 0.0):.4f}"
                    break

        # Use weighted average for overall metrics
        precision = result.weighted_avg.get("precision", 0.0) if result.weighted_avg else 0.0
        recall = result.weighted_avg.get("recall", 0.0) if result.weighted_avg else 0.0
        f1 = result.weighted_avg.get("f1_score", 0.0) if result.weighted_avg else 0.0

        row = f"{result.model_name:<15} {result.accuracy:>10.4f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {fall_auc:>10}"
        lines.append(row)

    lines.append("-" * 65)
    lines.append("")

    # Per-class breakdown
    lines.append("PER-CLASS METRICS")
    lines.append("-" * 65)

    for result in results:
        lines.append(f"\n{result.model_name} ({result.model_type.upper()}):")
        lines.append(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        lines.append(f"  {'-' * 60}")

        for class_name, metrics in result.class_metrics.items():
            lines.append(
                f"  {class_name:<20} {metrics.precision:>10.4f} {metrics.recall:>10.4f} "
                f"{metrics.f1_score:>10.4f} {metrics.support:>10}"
            )

        lines.append(f"  {'-' * 60}")
        if result.macro_avg:
            lines.append(
                f"  {'macro avg':<20} {result.macro_avg['precision']:>10.4f} "
                f"{result.macro_avg['recall']:>10.4f} {result.macro_avg['f1_score']:>10.4f}"
            )
        if result.weighted_avg:
            lines.append(
                f"  {'weighted avg':<20} {result.weighted_avg['precision']:>10.4f} "
                f"{result.weighted_avg['recall']:>10.4f} {result.weighted_avg['f1_score']:>10.4f}"
            )

    return "\n".join(lines)


def print_detailed_results(result: ModelEvaluationResult, idx_to_label: Dict[int, str]) -> None:
    """Print detailed evaluation results for a single model.

    Args:
        result: Evaluation result to print.
        idx_to_label: Mapping from class index to label name.
    """
    print(f"\n{'=' * 60}")
    print(f"MODEL: {result.model_name} ({result.model_type.upper()})")
    print(f"{'=' * 60}")

    print(f"\nOverall Accuracy: {result.accuracy:.4f} ({result.accuracy * 100:.2f}%)")

    print(f"\nPer-Class Metrics:")
    print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"  {'-' * 60}")

    for class_name, metrics in result.class_metrics.items():
        print(
            f"  {class_name:<20} {metrics.precision:>10.4f} {metrics.recall:>10.4f} "
            f"{metrics.f1_score:>10.4f} {metrics.support:>10}"
        )

    print(f"  {'-' * 60}")
    if result.macro_avg:
        print(
            f"  {'macro avg':<20} {result.macro_avg['precision']:>10.4f} "
            f"{result.macro_avg['recall']:>10.4f} {result.macro_avg['f1_score']:>10.4f}"
        )
    if result.weighted_avg:
        print(
            f"  {'weighted avg':<20} {result.weighted_avg['precision']:>10.4f} "
            f"{result.weighted_avg['recall']:>10.4f} {result.weighted_avg['f1_score']:>10.4f}"
        )

    # Print confusion matrix
    labels = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    print(f"\nConfusion Matrix:")
    print(format_confusion_matrix(result.confusion_matrix, labels))

    # Print ROC AUC scores
    if result.roc_data:
        print(f"\nROC AUC Scores:")
        for class_idx, class_name in idx_to_label.items():
            key = f"class_{class_idx}"
            if key in result.roc_data:
                auc = result.roc_data[key].get("auc", 0.0)
                print(f"  {class_name}: {auc:.4f}")


def serialize_evaluation_result(result: ModelEvaluationResult) -> Dict[str, Any]:
    """Convert ModelEvaluationResult to a JSON-serializable dictionary.

    Args:
        result: Evaluation result to serialize.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "model_name": result.model_name,
        "model_type": result.model_type,
        "accuracy": result.accuracy,
        "class_metrics": {
            name: asdict(metrics) for name, metrics in result.class_metrics.items()
        },
        "confusion_matrix": result.confusion_matrix,
        "roc_data": result.roc_data,
        "macro_avg": result.macro_avg,
        "weighted_avg": result.weighted_avg,
    }


def evaluate_all_models(
    data_dir: Path,
    model_paths: Dict[str, Path],
    save_results: bool = False,
    output_path: Optional[Path] = None,
) -> EvaluationSummary:
    """Evaluate all models and create comparison.

    Args:
        data_dir: Directory containing test data.
        model_paths: Dictionary mapping model names to checkpoint paths.
        save_results: Whether to save results to JSON.
        output_path: Path to save results (if save_results is True).

    Returns:
        EvaluationSummary with all results.
    """
    print("=" * 60)
    print("FALL DETECTION MODEL EVALUATION")
    print("=" * 60)

    # Load test data
    print(f"\nLoading test data from {data_dir}...")
    x, y, idx_to_label = load_test_data(data_dir)
    print(f"  Loaded {len(y)} samples")
    print(f"  Classes: {idx_to_label}")
    print(f"  Input shape: {x.shape}")

    # Class distribution
    unique, counts = np.unique(y.numpy(), return_counts=True)
    print(f"  Class distribution:")
    for class_idx, count in zip(unique, counts):
        print(f"    {idx_to_label[class_idx]}: {count} ({count / len(y) * 100:.1f}%)")

    results: List[ModelEvaluationResult] = []

    # Evaluate each model
    for model_name, model_path in model_paths.items():
        if not model_path.exists():
            print(f"\n[WARNING] Model checkpoint not found: {model_path}")
            continue

        print(f"\nEvaluating {model_name}...")
        try:
            model, cfg, norm_mean, norm_std, saved_idx_to_label = load_model_checkpoint(model_path)
            result = evaluate_model(
                model=model,
                x=x,
                y=y,
                norm_mean=norm_mean,
                norm_std=norm_std,
                idx_to_label=idx_to_label,
                model_name=model_name,
                model_type=cfg.model_type.value,
            )
            results.append(result)
            print_detailed_results(result, idx_to_label)
        except Exception as e:
            print(f"  [ERROR] Failed to evaluate {model_name}: {e}")

    if not results:
        print("\n[ERROR] No models were successfully evaluated.")
        return EvaluationSummary()

    # Create comparison table
    comparison_table = format_comparison_table(results)
    print(f"\n{comparison_table}")

    # Determine best model (by weighted F1-score)
    best_model = max(
        results,
        key=lambda r: r.weighted_avg.get("f1_score", 0.0) if r.weighted_avg else 0.0
    )
    print(f"\nBest Model: {best_model.model_name} (weighted F1: {best_model.weighted_avg['f1_score']:.4f})")

    summary = EvaluationSummary(
        models=results,
        best_model=best_model.model_name,
        comparison_table=comparison_table,
    )

    # Save results if requested
    if save_results and output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_dict = {
            "summary": {
                "best_model": summary.best_model,
                "num_models_evaluated": len(results),
                "test_samples": len(y),
                "classes": idx_to_label,
            },
            "models": [serialize_evaluation_result(r) for r in results],
        }
        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate fall detection models and generate comparison metrics."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="ml/data",
        help="Directory containing test data with class subdirectories.",
    )
    parser.add_argument(
        "--mlp-model",
        type=str,
        default="ml/fallnet.pt",
        help="Path to MLP model checkpoint.",
    )
    parser.add_argument(
        "--cnn-model",
        type=str,
        default="ml/fallnet_cnn.pt",
        help="Path to CNN model checkpoint.",
    )
    parser.add_argument(
        "--lstm-model",
        type=str,
        default="ml/fallnet_lstm.pt",
        help="Path to LSTM model checkpoint.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save evaluation results to JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ml/evaluation_results.json",
        help="Output path for evaluation results JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for model evaluation."""
    args = parse_args()

    model_paths = {
        "FallNet-MLP": Path(args.mlp_model),
        "FallNet-CNN": Path(args.cnn_model),
        "FallNet-LSTM": Path(args.lstm_model),
    }

    evaluate_all_models(
        data_dir=Path(args.data_dir),
        model_paths=model_paths,
        save_results=args.save_results,
        output_path=Path(args.output) if args.save_results else None,
    )


if __name__ == "__main__":
    main()
