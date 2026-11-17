from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config.settings import load_config
from data.splitting import get_model_feature_columns, split_labeled_data
from features.engineering import LoanFeatureEngineer
from preprocessing.pipeline import build_model_pipeline, build_preprocessor
from utils.exceptions import DataValidationError


@pytest.fixture
def labeled_frame() -> pd.DataFrame:
    config = load_config()
    rows = 40
    data: dict[str, object] = {
        "Loan_ID": [f"LP{index:04}" for index in range(rows)],
        "Loan_Status": ["Y" if index % 2 else "N" for index in range(rows)],
        "ApplicantIncome": np.arange(1000, 1000 + rows * 100, 100),
        "CoapplicantIncome": np.arange(rows, dtype=float) * 50,
        "LoanAmount": np.arange(50, 50 + rows, dtype=float),
        "Loan_Amount_Term": [360.0 if index % 3 else 180.0 for index in range(rows)],
        "Gender": ["Female" if index % 2 else "Male" for index in range(rows)],
        "Married": ["Yes" if index % 2 else "No" for index in range(rows)],
        "Dependents": [
            "3+" if index % 4 == 3 else str(index % 4) for index in range(rows)
        ],
        "Education": ["Graduate" if index % 2 else "Not Graduate" for index in range(rows)],
        "Self_Employed": ["Yes" if index % 3 else "No" for index in range(rows)],
        "Credit_History": [float(index % 2) for index in range(rows)],
        "Property_Area": ["Urban", "Rural", "Semiurban", "Urban"] * (rows // 4),
        "Approval_Proxy": ["derived"] * rows,
    }
    frame = pd.DataFrame(data)
    frame.loc[1, "ApplicantIncome"] = np.nan
    frame.loc[2, "Gender"] = None
    frame.loc[3, "Credit_History"] = np.nan
    assert set(config["dataset"]["numeric_columns"]).issubset(frame.columns)
    return frame


def test_split_is_stratified_and_excludes_target_id_and_unavailable_columns(
    labeled_frame: pd.DataFrame,
) -> None:
    split = split_labeled_data(labeled_frame)

    assert len(split.X_train) + len(split.X_test) == len(labeled_frame)
    assert split.X_train.index.intersection(split.X_test.index).empty
    assert split.y_train.index.equals(split.X_train.index)
    assert split.y_test.index.equals(split.X_test.index)
    assert tuple(split.X_train.columns) == get_model_feature_columns()
    assert "Loan_ID" not in split.X_train
    assert "Loan_Status" not in split.X_train
    assert "Approval_Proxy" not in split.X_train
    assert split.y_train.value_counts(normalize=True).to_dict() == (
        split.y_test.value_counts(normalize=True).to_dict()
    )


def test_split_is_deterministic_from_centralized_seed(
    labeled_frame: pd.DataFrame,
) -> None:
    first = split_labeled_data(labeled_frame)
    second = split_labeled_data(labeled_frame)

    assert first.X_train.index.tolist() == second.X_train.index.tolist()
    assert first.X_test.index.tolist() == second.X_test.index.tolist()


def test_split_rejects_missing_configured_features(
    labeled_frame: pd.DataFrame,
) -> None:
    frame = labeled_frame.drop(columns="ApplicantIncome")

    with pytest.raises(DataValidationError, match="missing configured model features"):
        split_labeled_data(frame)


def test_split_rejects_duplicate_model_feature_rows(
    labeled_frame: pd.DataFrame,
) -> None:
    duplicate = labeled_frame.iloc[[0]].copy()
    duplicate.loc[:, "Loan_ID"] = "LP_DUPLICATE"
    frame = pd.concat([labeled_frame, duplicate], ignore_index=True)

    with pytest.raises(DataValidationError, match="Duplicate model-feature rows"):
        split_labeled_data(frame)


def test_preprocessor_fits_statistics_only_from_training_rows(
    labeled_frame: pd.DataFrame,
) -> None:
    split = split_labeled_data(labeled_frame)
    preprocessor = build_preprocessor()
    split.X_train.loc[split.X_train.index[0], "ApplicantIncome"] = np.nan
    split.X_test.loc[:, "ApplicantIncome"] = 1_000_000_000.0
    feature_engineer = LoanFeatureEngineer(
        numeric_columns=load_config()["dataset"]["numeric_columns"],
        log1p_columns=load_config()["feature_engineering"]["log1p_columns"],
    )
    train_features = feature_engineer.fit_transform(split.X_train)
    preprocessor.fit(train_features)
    numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps[
        "imputer"
    ]
    income_position = list(
        load_config()["dataset"]["numeric_columns"]
    ).index("ApplicantIncome")
    assert numeric_imputer.statistics_[income_position] == pytest.approx(
        train_features["ApplicantIncome"].median()
    )
    assert split.X_test["ApplicantIncome"].median() != pytest.approx(
        numeric_imputer.statistics_[income_position]
    )


def test_unknown_categories_are_ignored_without_changing_output_shape(
    labeled_frame: pd.DataFrame,
) -> None:
    split = split_labeled_data(labeled_frame)
    feature_engineer = LoanFeatureEngineer(
        numeric_columns=load_config()["dataset"]["numeric_columns"],
        log1p_columns=load_config()["feature_engineering"]["log1p_columns"],
    ).fit(split.X_train)
    train_features = feature_engineer.transform(split.X_train)
    preprocessor = build_preprocessor().fit(train_features)
    unseen = split.X_test.iloc[[0]].copy()
    unseen.loc[:, "Gender"] = "Unseen"
    unseen.loc[:, "Property_Area"] = "Unseen"
    unseen_features = feature_engineer.transform(unseen)

    transformed = preprocessor.transform(unseen_features)

    assert transformed.shape[0] == 1
    assert transformed.shape[1] == preprocessor.transform(
        feature_engineer.transform(split.X_test)
    ).shape[1]
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert np.isfinite(dense).all()


def test_model_pipeline_supports_fold_local_preprocessing(
    labeled_frame: pd.DataFrame,
) -> None:
    split = split_labeled_data(labeled_frame)
    model_pipeline = build_model_pipeline(LogisticRegression(max_iter=500))
    scores = cross_val_score(
        model_pipeline,
        split.X_train,
        split.y_train,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring="balanced_accuracy",
    )

    assert len(scores) == 3
    assert np.isfinite(scores).all()
    assert "preprocessor" in model_pipeline.named_steps
    assert "estimator" in model_pipeline.named_steps
    assert "feature_engineering" in model_pipeline.named_steps


def test_preprocessor_rejects_feature_configuration_conflicts() -> None:
    config = deepcopy(load_config())
    config["dataset"]["numeric_columns"].append("Loan_ID")

    with pytest.raises(DataValidationError, match="target or identifier"):
        build_preprocessor(config)
