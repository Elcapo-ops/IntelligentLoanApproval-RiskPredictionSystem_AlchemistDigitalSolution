"""Probability calibration and training-only comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import clone
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from config.settings import load_config, validate_config
from utils.exceptions import DataValidationError


@dataclass(frozen=True)
class CalibrationMetrics:
    """Probability calibration summary for one method."""

    method: str
    brier_score: float
    log_loss: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    calibration_curve: tuple[np.ndarray, np.ndarray]
    positive_probability_range: tuple[float, float]


@dataclass(frozen=True)
class CalibrationComparisonResult:
    """Comparison of uncalibrated and calibrated models from training-only CV."""

    baseline: CalibrationMetrics
    calibrated: dict[str, CalibrationMetrics]
    selected_method: str | None
    selection_metric: str
    probability_range: tuple[float, float]
    improvement_over_baseline: bool


def _positive_index(model: Any, positive_label: str) -> int:
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise DataValidationError(
            "Model must expose fitted classes_ before probability calibration"
        )
    labels = list(classes)
    if positive_label not in labels:
        raise DataValidationError(
            f"Positive label {positive_label!r} not found in model classes: {labels}"
        )
    return int(labels.index(positive_label))


def _encode_target(y: pd.Series | np.ndarray, positive_label: str) -> np.ndarray:
    values = np.asarray(y)
    return (values == positive_label).astype(int)


def _metrics_from_probabilities(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    positive_label: str,
    *,
    method: str,
) -> CalibrationMetrics:
    y_binary = _encode_target(y_true, positive_label)
    if probabilities.ndim != 1:
        raise DataValidationError("Probability vector must be one-dimensional")
    if not np.all((probabilities >= 0.0) & (probabilities <= 1.0)):
        raise DataValidationError("Probabilities must fall within the [0, 1] interval")
    predicted = (probabilities >= 0.5).astype(int)
    if np.unique(y_binary).size < 2:
        raise DataValidationError("Calibration requires both classes in the target data")
    positive_probability_range = (
        float(np.min(probabilities)),
        float(np.max(probabilities)),
    )
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_binary,
        probabilities,
        n_bins=10,
        strategy="uniform",
    )
    return CalibrationMetrics(
        method=method,
        brier_score=float(brier_score_loss(y_binary, probabilities)),
        log_loss=float(log_loss(y_binary, probabilities, labels=[0, 1])),
        roc_auc=float(roc_auc_score(y_binary, probabilities)),
        precision=float(
            precision_score(y_binary, predicted, zero_division=0)
        ),
        recall=float(recall_score(y_binary, predicted, zero_division=0)),
        f1=float(f1_score(y_binary, predicted, zero_division=0)),
        calibration_curve=(fraction_of_positives, mean_predicted_value),
        positive_probability_range=positive_probability_range,
    )


def evaluate_probability_calibration(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    positive_label: str = "Y",
    method: str = "sigmoid",
    cv_folds: int = 3,
    random_state: int | None = None,
    config: dict[str, Any] | None = None,
) -> CalibrationMetrics:
    """Calibrate a classifier with training-only CV and summarize validation metrics.

    The estimator is trained on the provided X/y data using a fold-local
    calibration procedure. The final holdout is never passed to this function.
    """
    settings = validate_config(config if config is not None else load_config())
    if method not in {"sigmoid", "isotonic"}:
        raise DataValidationError(f"Unsupported calibration method: {method!r}")
    if cv_folds < 2:
        raise DataValidationError("Calibration requires at least 2 CV folds")
    if random_state is None:
        random_state = settings["project"]["random_seed"]
    calibrator = CalibratedClassifierCV(
        estimator=clone(estimator),
        method=method,
        cv=StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        ),
    )
    calibrator.fit(X, y)
    probabilities = calibrator.predict_proba(X)[:, _positive_index(calibrator, positive_label)]
    return _metrics_from_probabilities(y, probabilities, positive_label, method=method)


def compare_calibration_methods(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    positive_label: str = "Y",
    methods: Iterable[str] = ("sigmoid", "isotonic"),
    cv_folds: int = 3,
    random_state: int | None = None,
    config: dict[str, Any] | None = None,
    selection_metric: str = "brier_score",
) -> CalibrationComparisonResult:
    """Compare supported calibration methods using training-only cross-validation."""
    settings = validate_config(config if config is not None else load_config())
    method_list = list(methods)
    if not method_list:
        raise DataValidationError("At least one calibration method is required")
    baseline = _metrics_from_probabilities(
        y,
        estimator.predict_proba(X)[:, _positive_index(estimator, positive_label)],
        positive_label,
        method="uncalibrated",
    )
    calibrated: dict[str, CalibrationMetrics] = {}
    for method in method_list:
        calibrated[method] = evaluate_probability_calibration(
            estimator,
            X,
            y,
            positive_label=positive_label,
            method=method,
            cv_folds=cv_folds,
            random_state=random_state,
            config=settings,
        )
    if selection_metric not in {"brier_score", "log_loss", "roc_auc", "f1"}:
        raise DataValidationError(
            f"Unsupported calibration selection metric: {selection_metric!r}"
        )

    def _metric_value(metrics: CalibrationMetrics) -> float:
        return float(getattr(metrics, selection_metric))

    order = "min" if selection_metric in {"brier_score", "log_loss"} else "max"
    scored = {
        name: _metric_value(metrics)
        for name, metrics in calibrated.items()
    }
    if order == "min":
        selected_method = min(scored, key=lambda name: scored[name])
        selected_value = scored[selected_method]
        improvement_over_baseline = selected_value < getattr(baseline, selection_metric)
    else:
        selected_method = max(scored, key=lambda name: scored[name])
        selected_value = scored[selected_method]
        improvement_over_baseline = selected_value > getattr(baseline, selection_metric)
    if not improvement_over_baseline:
        selected_method = None
    return CalibrationComparisonResult(
        baseline=baseline,
        calibrated=calibrated,
        selected_method=selected_method,
        selection_metric=selection_metric,
        probability_range=(
            float(np.min([v.positive_probability_range[0] for v in calibrated.values()])),
            float(np.max([v.positive_probability_range[1] for v in calibrated.values()])),
        ),
        improvement_over_baseline=improvement_over_baseline,
    )


def save_calibration_summary(
    result: CalibrationComparisonResult,
    *,
    report_path: str | Path,
    include_uncalibrated: bool = True,
) -> Path:
    """Persist a comparison summary to JSON/Markdown, returning the path."""
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_method": result.selected_method,
        "selection_metric": result.selection_metric,
        "improvement_over_baseline": result.improvement_over_baseline,
        "baseline": {
            "method": result.baseline.method,
            "brier_score": result.baseline.brier_score,
            "log_loss": result.baseline.log_loss,
            "roc_auc": result.baseline.roc_auc,
            "precision": result.baseline.precision,
            "recall": result.baseline.recall,
            "f1": result.baseline.f1,
        },
        "calibrated": {
            name: {
                "brier_score": metrics.brier_score,
                "log_loss": metrics.log_loss,
                "roc_auc": metrics.roc_auc,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
            }
            for name, metrics in result.calibrated.items()
        },
    }
    destination.write_text(
        "# Probability calibration comparison\n\n"
        + (
            "## Baseline\n\n"
            + f"- Uncalibrated brier_score: {result.baseline.brier_score:.4f}\n"
            + f"- Uncalibrated log_loss: {result.baseline.log_loss:.4f}\n"
            + f"- Selected calibration method: {result.selected_method}\n"
            + f"- Improvement over baseline: {result.improvement_over_baseline}\n\n"
            + "## Calibrated methods\n\n"
            + "\n".join(
                f"- `{method}`: brier={metrics.brier_score:.4f}, log_loss={metrics.log_loss:.4f}, roc_auc={metrics.roc_auc:.4f}, f1={metrics.f1:.4f}"
                for method, metrics in result.calibrated.items()
            )
            + "\n"
            if include_uncalibrated
            else ""
        ),
        encoding="utf-8",
    )
    return destination


__all__ = [
    "CalibrationComparisonResult",
    "CalibrationMetrics",
    "compare_calibration_methods",
    "evaluate_probability_calibration",
    "save_calibration_summary",
]
