"""Training-only threshold optimization for probabilistic classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from utils.exceptions import DataValidationError


@dataclass(frozen=True)
class ThresholdResult:
    """One threshold and its requested metric summary."""

    threshold: float
    metric_name: str
    metric_value: float
    precision: float
    recall: float
    f1: float
    confusion_matrix_values: dict[str, int]


@dataclass(frozen=True)
class ThresholdOptimizationResult:
    """Selection result across candidate thresholds."""

    selected_threshold: float
    selected_metric: str
    selected_metric_value: float
    threshold_results: list[ThresholdResult]


def optimize_classification_thresholds(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    candidate_thresholds: list[float] | None = None,
    positive_label: str = "Y",
    metric: str = "f1",
) -> ThresholdOptimizationResult:
    """Select a classification threshold using training-only OOF probabilities.

    The passed probabilities must be derived only from training-data folds and
    must not come from the final holdout. This function is meant for selecting a
    decision threshold before any holdout evaluation is done.
    """
    if probabilities.ndim != 1:
        raise DataValidationError("Probability vector must be one-dimensional")
    if not np.all((probabilities >= 0.0) & (probabilities <= 1.0)):
        raise DataValidationError("Threshold optimization probabilities must be in [0, 1]")
    if candidate_thresholds is None:
        candidate_thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    thresholds = sorted({float(threshold) for threshold in candidate_thresholds})
    if not thresholds:
        raise DataValidationError("At least one candidate threshold is required")
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise DataValidationError(f"Threshold must be within [0, 1]: {threshold!r}")

    y_values = np.asarray(y_true)
    y_binary = (y_values == positive_label).astype(int)
    if np.unique(y_binary).size < 2:
        raise DataValidationError("Threshold optimization requires both classes in y_true")

    metric_name = metric.lower()
    supported = {"precision", "recall", "f1", "accuracy"}
    if metric_name not in supported:
        raise DataValidationError(
            f"Unsupported threshold metric '{metric}' ; expected one of {sorted(supported)}"
        )

    results: list[ThresholdResult] = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        matrix = confusion_matrix(y_binary, predictions, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        precision = precision_score(y_binary, predictions, zero_division=0)
        recall = recall_score(y_binary, predictions, zero_division=0)
        f1 = f1_score(y_binary, predictions, zero_division=0)
        accuracy = (tn + tp) / (tn + fp + fn + tp) if (tn + fp + fn + tp) else 0.0
        metric_value = {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}[metric_name]
        results.append(
            ThresholdResult(
                threshold=float(threshold),
                metric_name=metric_name,
                metric_value=float(metric_value),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                confusion_matrix_values={
                    "true_negative": int(tn),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "true_positive": int(tp),
                },
            )
        )

    selected = max(results, key=lambda item: item.metric_value)
    return ThresholdOptimizationResult(
        selected_threshold=selected.threshold,
        selected_metric=metric_name,
        selected_metric_value=selected.metric_value,
        threshold_results=results,
    )


__all__ = [
    "ThresholdOptimizationResult",
    "ThresholdResult",
    "optimize_classification_thresholds",
]
