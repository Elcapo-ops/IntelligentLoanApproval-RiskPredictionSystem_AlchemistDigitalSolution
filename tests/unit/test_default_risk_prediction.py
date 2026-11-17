"""Inference validation and risk-category tests for the default-risk module."""

import numpy as np
import pytest

from default_risk.data import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from default_risk.prediction import (
    DEFAULT_RISK_DISCLAIMER,
    assign_risk_category,
    predict_default_risk,
    validate_default_risk_input,
)
from utils.exceptions import DataValidationError


def _applicant() -> dict[str, object]:
    row: dict[str, object] = {feature: 0.0 for feature in NUMERIC_FEATURES}
    row.update({feature: "Other" for feature in CATEGORICAL_FEATURES})
    row["fico_range_low"] = 700.0
    row["fico_range_high"] = 704.0
    return row


class _ProbabilityStub:
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        assert list(frame.columns) == FEATURE_COLUMNS
        return np.array([[0.7, 0.3]])


def test_unknown_categories_and_missing_optional_values_are_allowed() -> None:
    applicant = _applicant()
    applicant["purpose"] = "previously_unseen_category"
    applicant["annual_inc"] = None

    frame = validate_default_risk_input(applicant)

    assert frame.loc[0, "purpose"] == "previously_unseen_category"
    assert np.isnan(frame.loc[0, "annual_inc"])


@pytest.mark.parametrize(
    "feature,value",
    [
        ("annual_inc", "not-a-number"),
        ("annual_inc", float("inf")),
        ("loan_amnt", -1),
        ("dti", 201),
        ("fico_range_low", 299),
    ],
)
def test_invalid_numeric_inputs_are_rejected(feature: str, value: object) -> None:
    applicant = _applicant()
    applicant[feature] = value

    with pytest.raises(DataValidationError):
        validate_default_risk_input(applicant)


def test_input_schema_requires_all_features_and_rejects_extras() -> None:
    applicant = _applicant()
    del applicant["annual_inc"]

    with pytest.raises(DataValidationError, match="schema mismatch"):
        validate_default_risk_input(applicant)

    applicant = _applicant()
    applicant["loan_status"] = "Charged Off"
    with pytest.raises(DataValidationError, match="schema mismatch"):
        validate_default_risk_input(applicant)


@pytest.mark.parametrize(
    "probability,expected",
    [
        (0.0, "Low Risk"),
        (0.0999, "Low Risk"),
        (0.10, "Moderate Risk"),
        (0.25, "High Risk"),
        (0.50, "Very High Risk"),
        (1.0, "Very High Risk"),
    ],
)
def test_risk_categories_use_configured_boundaries(
    probability: float,
    expected: str,
) -> None:
    assert assign_risk_category(probability) == expected


def test_prediction_uses_explicit_class_order_and_disclaimer() -> None:
    result = predict_default_risk(
        _ProbabilityStub(),
        _applicant(),
        {"model_name": "lendingclub_default_risk", "model_version": "0.1.0"},
    )

    assert result["default_probability"] == pytest.approx(0.3)
    assert result["non_default_probability"] == pytest.approx(0.7)
    assert result["risk_category"] == "High Risk"
    assert result["disclaimer"] == DEFAULT_RISK_DISCLAIMER
