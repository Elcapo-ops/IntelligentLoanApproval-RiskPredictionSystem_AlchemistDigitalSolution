"""Local, one-feature-at-a-time sensitivity explanations for raw applicants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.exceptions import DataValidationError


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _reference_values(reference: pd.DataFrame) -> pd.Series:
    values: dict[str, Any] = {}
    for column in reference:
        series = reference[column]
        if pd.api.types.is_numeric_dtype(series):
            values[column] = series.median()
        else:
            modes = series.mode(dropna=True)
            values[column] = modes.iloc[0] if not modes.empty else np.nan
    return pd.Series(values)


def explain_local_prediction(
    model: Any,
    applicant: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict[str, Any]:
    """Measure raw model-score changes when each input is replaced by reference.

    This is a one-feature-at-a-time sensitivity check, not an additive
    attribution or a causal/counterfactual claim. Replacing correlated inputs
    independently can create unrealistic applicant profiles. Scores are raw
    and uncalibrated outputs for the historical approval class.
    """
    if len(applicant) != 1:
        raise DataValidationError("A local explanation requires exactly one row")
    if reference.empty or list(applicant.columns) != list(reference.columns):
        raise DataValidationError(
            "Local explanation reference must match the applicant feature schema"
        )
    if "Y" not in list(model.classes_):
        raise DataValidationError("Local explanation model has no approved class 'Y'")

    approved_index = list(model.classes_).index("Y")
    baseline_score = float(
        model.predict_proba(applicant)[:, approved_index][0]
    )
    predicted_loan_status = str(model.predict(applicant)[0])
    replacements = _reference_values(reference)
    sensitivities: list[dict[str, Any]] = []
    for feature in applicant.columns:
        changed = applicant.copy()
        if pd.api.types.is_numeric_dtype(changed[feature]):
            changed[feature] = pd.to_numeric(changed[feature]).astype("float64")
        changed.loc[:, feature] = replacements[feature]
        score = float(model.predict_proba(changed)[:, approved_index][0])
        sensitivities.append(
            {
                "feature": feature,
                "applicant_value": _json_value(applicant.iloc[0][feature]),
                "reference_value": _json_value(replacements[feature]),
                "uncalibrated_approval_score_after_replacement": score,
                "uncalibrated_approval_score_delta": score - baseline_score,
            }
        )
    sensitivities.sort(
        key=lambda item: abs(item["uncalibrated_approval_score_delta"]),
        reverse=True,
    )
    return {
        "uncalibrated_approval_class_score": baseline_score,
        "predicted_loan_status": predicted_loan_status,
        "prediction_rule": "pipeline's default predict method",
        "reference_rows": int(len(reference)),
        "sensitivity_method": (
            "one original feature at a time replaced with reference median "
            "for numeric features or reference mode for categorical features"
        ),
        "interpretation_warning": (
            "Descriptive model sensitivity only; not causal, not additive, "
            "and not a lending rationale by itself."
        ),
        "features": sensitivities,
    }


def save_local_explanation_chart(
    explanations: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Save local raw-score-change charts with original feature labels."""
    if not explanations:
        raise DataValidationError("Cannot plot empty local explanations")
    figure, axes = plt.subplots(
        len(explanations),
        1,
        figsize=(10, max(4, 3.8 * len(explanations))),
        squeeze=False,
    )
    for axis, (case_name, explanation) in zip(axes[:, 0], explanations.items()):
        features = explanation["features"][:10]
        features = list(reversed(features))
        deltas = [
            row["uncalibrated_approval_score_delta"] for row in features
        ]
        labels = [row["feature"] for row in features]
        colors = ["#3c8d6d" if delta >= 0 else "#bd5a56" for delta in deltas]
        axis.barh(labels, deltas, color=colors)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(
            f"{case_name}: uncalibrated historical approval-class score "
            f"{explanation['uncalibrated_approval_class_score']:.3f}"
        )
        axis.set_xlabel(
            "Change in raw, uncalibrated score for historical approval class"
        )
        axis.set_ylabel("Original input feature")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
