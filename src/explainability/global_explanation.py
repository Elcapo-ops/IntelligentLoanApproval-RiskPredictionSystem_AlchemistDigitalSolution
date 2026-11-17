"""Fold-held-out, original-feature importance for fitted loan pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from utils.exceptions import DataValidationError


@dataclass(frozen=True)
class GlobalExplanationResult:
    """Out-of-fold outputs, fitted fold models, and raw-feature importances."""

    importance: pd.DataFrame
    oof_predictions: pd.Series
    oof_approval_score: pd.Series
    fold_models: tuple[Any, ...]
    fold_train_indices: tuple[np.ndarray, ...]
    fold_metrics: pd.DataFrame
    oof_fold: pd.Series


def cross_validated_permutation_importance(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    random_state: int,
    n_repeats: int = 5,
) -> GlobalExplanationResult:
    """Fit on CV training folds and permute unencoded input features per fold.

    Permutation is performed on the original DataFrame passed to the complete
    pipeline. This preserves application-facing feature names and lets the
    pipeline consistently regenerate derived and encoded features.
    """
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise DataValidationError("Global explanations require a non-empty DataFrame")
    if not X.columns.is_unique:
        raise DataValidationError("Explanation input columns must be unique")
    if len(X) != len(y) or not X.index.equals(y.index):
        raise DataValidationError("Explanation features and targets must align")
    if n_repeats < 1:
        raise DataValidationError("Permutation repeats must be positive")

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    predictions = np.empty(len(X), dtype=object)
    probabilities = np.full(len(X), np.nan, dtype=float)
    fold_assignments = np.zeros(len(X), dtype=int)
    importance_rows: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, float | int]] = []
    fold_models: list[Any] = []
    fold_train_indices: list[np.ndarray] = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(X, y),
        start=1,
    ):
        model = clone(estimator)
        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_validation = X.iloc[validation_indices]
        y_validation = y.iloc[validation_indices]
        model.fit(X_train, y_train)
        fold_models.append(model)
        fold_train_indices.append(train_indices)
        fold_assignments[validation_indices] = fold_number

        model_classes = list(model.classes_)
        if "Y" not in model_classes:
            raise DataValidationError(
                f"Fitted explanation model has no approved class: {model_classes}"
            )
        approved_index = model_classes.index("Y")
        fold_probabilities = model.predict_proba(X_validation)[:, approved_index]
        predictions[validation_indices] = model.predict(X_validation)
        probabilities[validation_indices] = fold_probabilities

        permutation = permutation_importance(
            model,
            X_validation,
            y_validation,
            scoring="balanced_accuracy",
            n_repeats=n_repeats,
            random_state=random_state + fold_number,
            n_jobs=1,
        )
        importance_rows.append(
            pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance_mean": permutation.importances_mean,
                    "importance_repeat_std": permutation.importances_std,
                    "fold": fold_number,
                }
            )
        )
        fold_metrics.append(
            {
                "fold": fold_number,
                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        y_validation,
                        model.predict(X_validation),
                    )
                ),
                "validation_rows": len(validation_indices),
            }
        )

    if not np.isfinite(probabilities).all():
        raise DataValidationError("Cross-validation did not predict every row")

    all_importances = pd.concat(importance_rows, ignore_index=True)
    importance = (
        all_importances.groupby("feature", sort=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_fold_std=("importance_mean", "std"),
            importance_repeat_std=("importance_repeat_std", "mean"),
        )
        .reset_index()
        .sort_values("importance_mean", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    return GlobalExplanationResult(
        importance=importance,
        oof_predictions=pd.Series(predictions, index=X.index, name="prediction"),
        oof_approval_score=pd.Series(
            probabilities,
            index=X.index,
            name="uncalibrated_approval_class_score",
        ),
        fold_models=tuple(fold_models),
        fold_train_indices=tuple(fold_train_indices),
        fold_metrics=pd.DataFrame(fold_metrics),
        oof_fold=pd.Series(fold_assignments, index=X.index, name="fold"),
    )


def compute_shap_values(
    model: Any,
    background: pd.DataFrame,
    examples: pd.DataFrame,
    *,
    background_rows: int = 20,
) -> tuple[float, pd.DataFrame] | None:
    """Compute optional model-agnostic SHAP values on raw named inputs.

    Returns ``None`` if SHAP is not installed. The predictor wraps the entire
    pipeline, so SHAP inputs remain the original, unencoded columns.
    """
    try:
        shap = importlib.import_module("shap")
    except ImportError:
        return None

    if background.empty or examples.empty:
        raise DataValidationError("SHAP needs non-empty background and example rows")
    columns = list(background.columns)
    if list(examples.columns) != columns:
        raise DataValidationError("SHAP examples must match background feature schema")
    sample = background.sample(
        n=min(background_rows, len(background)),
        random_state=42,
    )
    model_classes = list(model.classes_)
    if "Y" not in model_classes:
        raise DataValidationError("SHAP model has no approved class 'Y'")
    approved_index = model_classes.index("Y")

    def approval_class_score(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=columns)
        return np.asarray(model.predict_proba(frame)[:, approved_index], dtype=float)

    explainer = shap.Explainer(
        approval_class_score,
        sample.to_numpy(),
        algorithm="permutation",
        feature_names=columns,
    )
    max_evaluations = 2 * len(columns) + 1
    explanation = explainer(
        examples.to_numpy(),
        max_evals=max_evaluations,
    )
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, approved_index]
    if values.shape != (len(examples), len(columns)):
        raise DataValidationError(
            f"SHAP returned unexpected values shape: {values.shape}"
        )
    base_values = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    return (
        float(base_values[0]),
        pd.DataFrame(values, columns=columns, index=examples.index),
    )


def save_global_importance_chart(
    importance: pd.DataFrame,
    output_path: Path,
    *,
    top_n: int = 12,
) -> None:
    """Save a readable horizontal permutation-importance chart."""
    if importance.empty:
        raise DataValidationError("Cannot plot empty permutation importance")
    shown = importance.head(top_n).sort_values("importance_mean")
    figure, axis = plt.subplots(figsize=(10, max(5, 0.42 * len(shown))))
    errors = shown["importance_fold_std"].fillna(0.0).to_numpy()
    axis.barh(
        shown["feature"],
        shown["importance_mean"],
        xerr=errors,
        color="#3478a8",
        alpha=0.9,
        capsize=3,
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_title("Cross-validated permutation importance")
    axis.set_xlabel("Decrease in balanced accuracy after permutation")
    axis.set_ylabel("Original input feature")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
