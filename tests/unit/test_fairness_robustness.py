import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from evaluation.fairness_robustness import (
    build_fairness_robustness_report,
    evaluate_historical_fairness,
    evaluate_robustness,
)
from utils.exceptions import DataValidationError


def _sample_frame() -> pd.DataFrame:
    data = {
        "Gender": ["Female", "Female", "Male", "Male", "Female", "Male"] * 4,
        "ApplicantIncome": [1200, 3500, 1600, 4200, 1800, 5300] * 4,
    }
    return pd.DataFrame(data)


def _model() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", ["ApplicantIncome"]),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["Gender"]),
        ]
    )
    estimator = RandomForestClassifier(
        n_estimators=12,
        random_state=42,
        class_weight="balanced",
    )
    return Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])


def test_fairness_summary_reports_available_attributes_and_group_gaps() -> None:
    frame = _sample_frame()
    y_true = pd.Series(["Y", "N", "Y", "N", "Y", "N"] * 4, index=frame.index)
    y_pred = pd.Series(["Y", "Y", "N", "N", "Y", "N"] * 4, index=frame.index)

    result = evaluate_historical_fairness(
        frame,
        y_true,
        y_pred,
        attributes=["Gender", "ApplicantIncome"],
        minimum_group_size=2,
    )

    assert result["available_attributes"] == ["Gender"]
    assert result["missing_attributes"] == []
    assert set(result["metrics"]["attribute"]) == {"Gender"}
    assert result["summary"]["selection_rate_gap"].notna().all()
    assert "historical approval" in build_fairness_robustness_report(
        result,
        {"results": [{"scenario": "missing_values", "status": "PASS", "details": "ok"}]},
    ).lower()


def test_robustness_checks_cover_missing_and_invalid_inputs() -> None:
    X = _sample_frame()
    model = _model().fit(X, pd.Series(["Y", "N"] * 12, index=X.index))

    result = evaluate_robustness(model, X)

    assert len(result["results"]) >= 5
    assert all(item["status"] in {"PASS", "FAIL"} for item in result["results"])
    assert any(item["scenario"] == "missing_values" for item in result["results"])
    assert any(item["scenario"] == "invalid_numeric_type" for item in result["results"])


def test_fairness_evaluation_rejects_missing_attributes_and_misaligned_inputs() -> None:
    frame = _sample_frame()
    y_true = pd.Series(["Y", "N", "Y", "N"] * 6, index=range(1, 25))

    with pytest.raises(DataValidationError, match="aligned indices"):
        evaluate_historical_fairness(
            frame,
            y_true,
            pd.Series(["Y", "N"] * 12, index=frame.index),
            attributes=["Gender"],
        )

    with pytest.raises(DataValidationError, match="No configured fairness attributes"):
        evaluate_historical_fairness(
            frame,
            pd.Series(["Y", "N"] * 12, index=frame.index),
            pd.Series(["Y", "N"] * 12, index=frame.index),
            attributes=["Unknown_Attribute"],
        )
