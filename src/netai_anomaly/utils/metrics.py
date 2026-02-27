"""Evaluation metrics for anomaly detection models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_anomaly_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """Compute comprehensive anomaly detection metrics.

    Args:
        y_true: Binary ground truth labels (1 = anomaly, 0 = normal).
        y_scores: Anomaly scores (higher = more anomalous).
        threshold: Decision threshold. If None, uses optimal F1 threshold.

    Returns:
        Dictionary of metric names to values.
    """
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores).ravel()

    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_scores)

    y_pred = (y_scores >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }

    # ROC AUC requires both classes present
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Find the threshold that maximizes F1 score."""
    if len(np.unique(y_true)) < 2:
        return float(np.percentile(y_scores, 95))

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # Compute F1 for each threshold
    f1s = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0,
    )

    # precision_recall_curve returns n+1 values; thresholds has n values
    best_idx = np.argmax(f1s[:-1])
    return float(thresholds[best_idx])
