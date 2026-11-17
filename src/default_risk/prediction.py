"""Inference helpers for the separately registered LendingClub model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from default_risk.data import FEATURE_COLUMNS
from default_risk.modeling import (
    DEFAULT_TARGET_MAPPING,
    MODEL_NAME,
    MODEL_VERSION,
    RISK_THRESHOLDS,
)
from models.registry import load_model_artifact
from utils.exceptions import DataValidationError

DEFAULT_RISK_DISCLAIMER = (
    "This output is a statistical estimate of loan default risk based on "
    "historical loan performance data. It is not a guarantee of future repayment "
    "or default, a regulated credit rating, or an automatic lending decision."
)
NUMERIC_INPUTS = {
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "credit_history_months",
}


def load_default_risk_model(
    project_root: Path | str,
) -> tuple[Any, dict[str, Any]]:
    """Load only the registered default-risk artifact after integrity checks."""
    root = Path(project_root).resolve()
    model, metadata = load_model_artifact(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        model_dir=root / "artifacts" / "models" / "default_risk",
        expected_feature_schema=FEATURE_COLUMNS,
        expected_target_mapping={"0": "Fully Paid", "1": "Charged Off"},
    )
    if metadata.get("target_definition") != DEFAULT_TARGET_MAPPING:
        raise DataValidationError("Registered default-risk target definition mismatch")
    return model, metadata


def validate_default_risk_input(applicant: dict[str, Any]) -> pd.DataFrame:
    """Validate one applicant-time input row without rejecting unseen categories."""
    if not isinstance(applicant, dict):
        raise DataValidationError("Applicant input must be a mapping")
    missing = sorted(set(FEATURE_COLUMNS).difference(applicant))
    extra = sorted(set(applicant).difference(FEATURE_COLUMNS))
    if missing or extra:
        raise DataValidationError(
            f"Default-risk input schema mismatch; missing={missing}, extra={extra}"
        )

    row: dict[str, Any] = {}
    for column in FEATURE_COLUMNS:
        value = applicant[column]
        if value is None or value is pd.NA or (
            not isinstance(value, (str, list, tuple, dict))
            and bool(pd.isna(value))
        ):
            row[column] = np.nan
            continue
        if isinstance(value, str) and not value.strip():
            row[column] = np.nan
            continue
        if column in NUMERIC_INPUTS:
            if isinstance(value, (bool, np.bool_)):
                raise DataValidationError(f"{column} must be a finite number")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise DataValidationError(
                    f"{column} must be a finite number"
                ) from exc
            if not np.isfinite(number):
                raise DataValidationError(f"{column} must be a finite number")
            if column not in {"dti", "revol_util"} and number < 0:
                raise DataValidationError(f"{column} must be non-negative")
            if number < 0:
                raise DataValidationError(f"{column} must be non-negative")
            if column == "dti" and number > 200:
                raise DataValidationError("dti must be between 0 and 200")
            if column == "revol_util" and number > 200:
                raise DataValidationError("revol_util must be between 0 and 200")
            if column in {"fico_range_low", "fico_range_high"} and not 300 <= number <= 850:
                raise DataValidationError(f"{column} must be between 300 and 850")
            row[column] = number
            continue
        if not isinstance(value, str):
            raise DataValidationError(f"{column} must be text")
        row[column] = value.strip()
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def assign_risk_category(
    probability: float,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Apply illustrative probability bands configured for this academic model."""
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise DataValidationError("Default probability must be within [0, 1]")
    bands = thresholds or RISK_THRESHOLDS
    if probability < bands["low_risk_max"]:
        return "Low Risk"
    if probability < bands["moderate_risk_max"]:
        return "Moderate Risk"
    if probability < bands["high_risk_max"]:
        return "High Risk"
    return "Very High Risk"


def predict_default_risk(
    model: Any,
    applicant: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return historical default probabilities and provenance for one input."""
    frame = validate_default_risk_input(applicant)
    classes = list(model.classes_)
    if 1 not in classes or 0 not in classes:
        raise DataValidationError(
            f"Model must expose both target classes 0 and 1; found {classes}"
        )
    probabilities = model.predict_proba(frame)[0]
    default_probability = float(probabilities[classes.index(1)])
    non_default_probability = float(probabilities[classes.index(0)])
    if (
        not np.isfinite([default_probability, non_default_probability]).all()
        or not 0 <= default_probability <= 1
        or not 0 <= non_default_probability <= 1
        or not np.isclose(default_probability + non_default_probability, 1.0, atol=1e-6)
    ):
        raise DataValidationError("Model produced invalid class probabilities")
    return {
        "default_probability": default_probability,
        "non_default_probability": non_default_probability,
        "risk_category": assign_risk_category(
            default_probability,
            metadata.get("risk_category_thresholds"),
        ),
        "model_name": metadata.get("model_name", MODEL_NAME),
        "model_version": metadata.get("model_version", MODEL_VERSION),
        "target": "loan_status",
        "disclaimer": DEFAULT_RISK_DISCLAIMER,
    }


def local_probability_sensitivity(
    model: Any,
    applicant: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute one-input-at-a-time probability changes against training defaults."""
    baseline = validate_default_risk_input(applicant)
    defaults = metadata.get("application_time_reference_values")
    if not isinstance(defaults, dict):
        return []
    classes = list(model.classes_)
    default_index = classes.index(1)
    base_probability = float(model.predict_proba(baseline)[0, default_index])
    results: list[dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        if feature not in defaults:
            continue
        changed = baseline.copy()
        changed.loc[0, feature] = defaults[feature]
        probability = float(model.predict_proba(changed)[0, default_index])
        results.append(
            {
                "feature": feature,
                "entered_value": applicant[feature],
                "training_reference_value": defaults[feature],
                "reference_default_probability": probability,
                "change_vs_input": probability - base_probability,
            }
        )
    return results


__all__ = [
    "DEFAULT_RISK_DISCLAIMER",
    "assign_risk_category",
    "load_default_risk_model",
    "local_probability_sensitivity",
    "predict_default_risk",
    "validate_default_risk_input",
]
