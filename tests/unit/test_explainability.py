import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from explainability.fairness import (
    calculate_group_metrics,
    summarize_group_gaps,
)
from explainability.global_explanation import (
    compute_shap_values,
    cross_validated_permutation_importance,
)
from explainability.local_explanation import explain_local_prediction
from utils.exceptions import DataValidationError


def _features(rows: int = 90) -> pd.DataFrame:
    values = np.arange(rows)
    return pd.DataFrame(
        {
            "ApplicantIncome": 1500 + (values * 137) % 9000,
            "Gender": np.where(values % 2, "Female", "Male"),
        }
    )


def _model() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", ["ApplicantIncome"]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["Gender"],
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "estimator",
                RandomForestClassifier(
                    n_estimators=8,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def test_global_importance_preserves_raw_features_and_oof_predictions() -> None:
    X = _features()
    y = pd.Series(
        np.where(
            (X["ApplicantIncome"] > X["ApplicantIncome"].median())
            | X["Gender"].eq("Female"),
            "Y",
            "N",
        ),
        index=X.index,
    )

    result = cross_validated_permutation_importance(
        _model(),
        X,
        y,
        n_splits=3,
        random_state=42,
        n_repeats=2,
    )

    assert set(result.importance["feature"]) == set(X.columns)
    assert not result.importance["feature"].str.contains("__").any()
    assert result.oof_predictions.index.equals(X.index)
    assert result.oof_approval_score.notna().all()
    assert result.oof_fold.nunique() == 3
    assert len(result.fold_models) == 3
    assert result.fold_metrics["balanced_accuracy"].between(0.0, 1.0).all()


def test_local_sensitivity_matches_pipeline_predictions_and_keeps_raw_names() -> None:
    X = _features()
    y = pd.Series(
        np.where(X["ApplicantIncome"] > X["ApplicantIncome"].median(), "Y", "N"),
        index=X.index,
    )
    model = _model().fit(X.iloc[10:], y.iloc[10:])
    applicant = X.iloc[[0]].copy()
    original = applicant.copy(deep=True)
    explanation = explain_local_prediction(model, applicant, X.iloc[10:])

    assert {row["feature"] for row in explanation["features"]} == set(X.columns)
    assert explanation["predicted_loan_status"] in {"N", "Y"}
    assert explanation["sensitivity_method"].startswith("one original feature")
    pd.testing.assert_frame_equal(applicant, original)
    assert all(
        np.isfinite(row["uncalibrated_approval_score_delta"])
        for row in explanation["features"]
    )


def test_local_explanation_rejects_multiple_applicants() -> None:
    X = _features(30)
    y = pd.Series(np.where(X["ApplicantIncome"] > 5000, "Y", "N"))
    model = _model().fit(X, y)

    with pytest.raises(DataValidationError, match="exactly one row"):
        explain_local_prediction(model, X.iloc[:2], X)


def test_fairness_rates_and_minimum_group_handling() -> None:
    frame = pd.DataFrame(
        {"Gender": ["Female"] * 12 + ["Male"] * 8 + [None] * 2}
    )
    y_true = pd.Series(["Y", "N"] * 11)
    y_pred = pd.Series(["Y", "Y", "N", "N"] * 5 + ["Y", "N"])

    metrics = calculate_group_metrics(
        frame,
        y_true,
        y_pred,
        attributes=["Gender"],
        minimum_group_size=10,
    )
    gaps = summarize_group_gaps(metrics)

    female = metrics.loc[metrics["group"] == "Female"].iloc[0]
    male = metrics.loc[metrics["group"] == "Male"].iloc[0]
    missing = metrics.loc[metrics["group"] == "Missing"].iloc[0]
    assert bool(female["eligible_for_comparison"])
    assert not bool(male["eligible_for_comparison"])
    assert not bool(missing["eligible_for_comparison"])
    assert female["approval_rate"] == pytest.approx(0.5)
    assert gaps.iloc[0]["eligible_groups"] == 1
    assert np.isnan(gaps.iloc[0]["approval_rate_gap"])


def test_fairness_rejects_misaligned_inputs() -> None:
    frame = pd.DataFrame({"Gender": ["Female", "Male"]})
    y_true = pd.Series(["Y", "N"], index=[1, 2])
    y_pred = pd.Series(["Y", "N"], index=[1, 2])

    with pytest.raises(DataValidationError, match="aligned indices"):
        calculate_group_metrics(
            frame,
            y_true,
            y_pred,
            attributes=["Gender"],
        )


def test_optional_shap_path_uses_raw_feature_names() -> None:
    X = _features(60)
    y = pd.Series(
        np.where(X["ApplicantIncome"] > X["ApplicantIncome"].median(), "Y", "N"),
        index=X.index,
    )
    model = _model().fit(X, y)

    result = compute_shap_values(
        model,
        X.iloc[:20],
        X.iloc[[25]],
        background_rows=5,
    )
    if result is not None:
        _, values = result
        assert list(values.columns) == list(X.columns)
        assert values.shape == (1, len(X.columns))
