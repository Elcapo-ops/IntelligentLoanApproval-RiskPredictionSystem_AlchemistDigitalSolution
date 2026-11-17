from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.dashboard_service import load_historical_dashboard_data
from app.services.prediction_service import (
    explain_historical_approval_prediction,
    get_form_defaults,
    load_prediction_model,
    predict_historical_approval,
)
from config.paths import get_project_paths
from config.settings import load_config
from data.loader import load_datasets
from data.splitting import get_model_feature_columns, split_labeled_data
from evaluation.evaluate import evaluate_selected_model
from models.registry import load_model_artifact
from models.train import train_models
from utils.exceptions import DataValidationError

_PROJECT_PATHS = get_project_paths(load_config())
_ACQUIRED_DATA_AVAILABLE = (
    _PROJECT_PATHS.labeled_dataset.is_file()
    and _PROJECT_PATHS.unlabeled_dataset.is_file()
)

@pytest.mark.skipif(
    not _ACQUIRED_DATA_AVAILABLE,
    reason="Acquired Kaggle CSVs are not included in source control",
)
def test_real_dataset_trains_evaluates_exports_and_serves_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = deepcopy(load_config())
    config["training"]["cv_folds"] = 3
    config["training"]["model_version"] = "8.1.0"
    config["models"]["parameters"]["logistic_regression"]["max_iter"] = 200
    config["models"]["parameters"]["random_forest"]["n_estimators"] = 8
    config["models"]["parameters"]["gradient_boosting"]["n_estimators"] = 6
    config["paths"]["models_dir"] = str(tmp_path / "models")

    monkeypatch.chdir(tmp_path)
    paths = get_project_paths(config)
    datasets = load_datasets(config=config)

    assert len(datasets.labeled) == 614
    assert len(datasets.unlabeled) == 367
    assert config["dataset"]["target_column"] not in datasets.unlabeled.columns
    assert set(datasets.labeled["Loan_Status"]) == {"N", "Y"}

    split = split_labeled_data(datasets.labeled, config=config)
    feature_columns = get_model_feature_columns(config)
    assert len(split.X_train) == 491
    assert len(split.X_test) == 123
    assert list(split.X_train.columns) == list(feature_columns)
    assert split.X_train.index.intersection(split.X_test.index).empty
    assert "Loan_ID" not in feature_columns
    assert "Loan_Status" not in feature_columns

    cv_path = tmp_path / "artifacts" / "model_cv_results.json"
    training = train_models(
        datasets.labeled,
        config=config,
        output_dir=paths.models_dir,
        report_path=tmp_path / "reports" / "training.md",
        results_path=cv_path,
    )
    assert set(training.fitted_models) == set(config["models"]["candidates"])
    assert np.isfinite(training.cv_results.select_dtypes(include="number")).all().all()
    assert all(path.is_file() for path in training.model_paths.values())

    evaluation = evaluate_selected_model(
        datasets.labeled,
        config=config,
        cv_results_path=cv_path,
        model_dir=paths.models_dir,
        output_dir=tmp_path / "reports" / "evaluation",
    )
    assert evaluation.metrics["positive_class"] == "Y"
    assert evaluation.metrics["negative_class"] == "N"
    assert evaluation.metrics["positive_probability_column_index"] == 1
    assert evaluation.metrics["confusion_matrix"]["tn"] + (
        evaluation.metrics["confusion_matrix"]["fp"]
        + evaluation.metrics["confusion_matrix"]["fn"]
        + evaluation.metrics["confusion_matrix"]["tp"]
    ) == len(split.X_test)
    assert evaluation.report_path.is_file()
    assert all(path.is_file() for path in evaluation.figure_paths.values())

    config["explainability"]["model_name"] = evaluation.selected_model
    config["explainability"]["model_version"] = evaluation.selected_version
    expected_schema = {
        column: str(datasets.labeled[column].dtype) for column in feature_columns
    }
    loaded, metadata = load_model_artifact(
        model_name=evaluation.selected_model,
        model_version=evaluation.selected_version,
        model_dir=paths.models_dir,
        expected_feature_schema=expected_schema,
        expected_target_mapping={"N": 0, "Y": 1},
    )
    assert metadata["holdout_evaluated"] is False
    assert metadata["dataset_hash"]
    assert list(loaded.classes_) == ["N", "Y"]
    assert list(loaded.predict_proba(split.X_test.iloc[:1]).shape) == [1, 2]
    feature_engineer = loaded.named_steps["feature_engineering"]
    training_engineered = feature_engineer.transform(split.X_train)
    holdout_engineered = feature_engineer.transform(split.X_test)
    numeric_pipeline = (
        loaded.named_steps["preprocessor"]
        .named_transformers_["numeric"]
    )
    numeric_columns = list(
        loaded.named_steps["preprocessor"].transformers_[0][2]
    )
    imputer_statistics = numeric_pipeline.named_steps["imputer"].statistics_
    np.testing.assert_allclose(
        imputer_statistics,
        training_engineered[numeric_columns].median().to_numpy(dtype=float),
    )
    applicant_income_index = numeric_columns.index("ApplicantIncome")
    assert imputer_statistics[applicant_income_index] != pytest.approx(
        holdout_engineered["ApplicantIncome"].median()
    )

    prediction_model, prediction_metadata = load_prediction_model(config)
    applicant = get_form_defaults(config)
    applicant_row = pd.DataFrame([applicant], columns=feature_columns)
    prediction = predict_historical_approval(prediction_model, applicant, config)
    assert prediction_metadata["model_version"] == evaluation.selected_version
    assert prediction["predicted_loan_status"] == str(
        prediction_model.predict(applicant_row)[0]
    )
    assert prediction["predicted_loan_status"] in {"N", "Y"}

    explanation = explain_historical_approval_prediction(
        prediction_model,
        applicant,
        config,
    )
    assert (
        explanation["predicted_loan_status"]
        == prediction["predicted_loan_status"]
    )
    assert explanation["reference_rows"] == len(split.X_train)
    assert {item["feature"] for item in explanation["features"]} == set(
        feature_columns
    )

    dashboard = load_historical_dashboard_data(config)
    assert dashboard["application_count"] == 614
    assert dashboard["approval_count"] + dashboard["rejection_count"] == 614

    with pytest.raises(DataValidationError, match="cannot be negative"):
        predict_historical_approval(
            prediction_model,
            {**applicant, "ApplicantIncome": -1},
            config,
        )
