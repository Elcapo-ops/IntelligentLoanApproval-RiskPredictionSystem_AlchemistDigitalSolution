import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import load_config
from data.splitting import split_labeled_data
from models.optimize import optimize_selected_model
from models.registry import load_model_artifact
from models.train import train_models
from utils.exceptions import DataValidationError


def _training_frame(rows: int = 80) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Loan_ID": [f"OP{index:04}" for index in range(rows)],
            "Loan_Status": ["Y" if index % 3 else "N" for index in range(rows)],
            "ApplicantIncome": 1500 + (np.arange(rows) * 137) % 9000,
            "CoapplicantIncome": (np.arange(rows) * 41) % 3200,
            "LoanAmount": 40 + (np.arange(rows) * 7) % 260,
            "Loan_Amount_Term": [360 if index % 5 else 180 for index in range(rows)],
            "Gender": ["Female" if index % 2 else "Male" for index in range(rows)],
            "Married": ["Yes" if index % 3 else "No" for index in range(rows)],
            "Dependents": [
                "3+" if index % 4 == 3 else str(index % 4) for index in range(rows)
            ],
            "Education": [
                "Graduate" if index % 3 else "Not Graduate" for index in range(rows)
            ],
            "Self_Employed": [
                "Yes" if index % 4 == 0 else "No" for index in range(rows)
            ],
            "Credit_History": [float(index % 2) for index in range(rows)],
            "Property_Area": ["Rural", "Urban", "Semiurban", "Rural"] * (rows // 4),
            "Availability_Leak": ["unavailable"] * rows,
        }
    )


def _optimization_config() -> dict:
    config = deepcopy(load_config())
    config["training"]["test_size"] = 0.25
    config["training"]["cv_folds"] = 3
    config["optimization"]["n_iter"] = 2
    config["optimization"]["parameter_distributions"] = {
        "estimator__n_estimators": [4, 6],
        "estimator__max_depth": [None],
        "estimator__min_samples_leaf": [1],
        "estimator__max_features": ["sqrt"],
        "estimator__class_weight": ["balanced"],
    }
    return config


def test_optimization_uses_training_cv_and_saves_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _training_frame()
    config = _optimization_config()
    baseline = train_models(
        frame,
        config=config,
        output_dir=tmp_path / "baseline_models",
        report_path=tmp_path / "baseline_report.md",
        results_path=tmp_path / "cv_results.json",
    )

    cv_payload = json.loads(baseline.results_path.read_text(encoding="utf-8"))
    for row in cv_payload["models"]:
        row["balanced_accuracy_mean"] = (
            0.9 if row["model"] == "random_forest" else 0.1
        )
        row["roc_auc_mean"] = 0.9 if row["model"] == "random_forest" else 0.1
    baseline.results_path.write_text(
        json.dumps(cv_payload, indent=2), encoding="utf-8"
    )

    import models.optimize as optimization_module

    original_fit = optimization_module.RandomizedSearchCV.fit
    search_fit_rows: list[set[int]] = []

    def record_search_fit(self, X, y=None, **fit_params):
        search_fit_rows.append(set(X.index))
        return original_fit(self, X, y, **fit_params)

    monkeypatch.setattr(
        optimization_module.RandomizedSearchCV,
        "fit",
        record_search_fit,
    )
    result = optimize_selected_model(
        frame,
        config=config,
        cv_results_path=baseline.results_path,
        model_dir=tmp_path / "optimized_models",
        results_path=tmp_path / "reports" / "tuning.json",
        report_path=tmp_path / "reports" / "tuning.md",
    )
    split = split_labeled_data(frame, config=config)
    persisted = json.loads(result.results_path.read_text(encoding="utf-8"))
    report = result.report_path.read_text(encoding="utf-8")

    assert result.selected_model == "random_forest"
    assert result.model_version == "1.1.0"
    assert search_fit_rows == [set(split.X_train.index)]
    assert set(result.best_params).issubset(
        config["optimization"]["parameter_distributions"]
    )
    assert len(result.cv_results) == config["optimization"]["n_iter"]
    assert len(persisted["candidate_cv_results"]) == config["optimization"]["n_iter"]
    assert np.isfinite(result.cv_results["mean_test_balanced_accuracy"]).all()
    assert persisted["holdout_used"] is False
    assert persisted["training_rows"] == len(split.X_train)
    assert persisted["holdout_rows_reserved"] == len(split.X_test)
    assert persisted["best_params"] == result.best_params
    assert len(persisted["cv_fold_scores"]) == config["training"]["cv_folds"]
    assert not any("threshold" in key.lower() for key in persisted)
    assert "same folds" in report
    assert "No threshold optimization" in report

    loaded, metadata = load_model_artifact(
        model_name="random_forest",
        model_version="1.1.0",
        model_dir=tmp_path / "optimized_models",
        expected_target_mapping={"N": 0, "Y": 1},
    )
    assert loaded.predict(split.X_train.iloc[:2]).shape == (2,)
    assert metadata["training_configuration"]["holdout_used"] is False
    assert metadata["training_configuration"]["training_rows"] == len(split.X_train)


def test_optimization_rejects_cv_results_that_used_holdout(tmp_path: Path) -> None:
    frame = _training_frame()
    config = _optimization_config()
    baseline = train_models(
        frame,
        config=config,
        output_dir=tmp_path / "models",
        report_path=tmp_path / "training.md",
        results_path=tmp_path / "cv_results.json",
    )
    payload = json.loads(baseline.results_path.read_text(encoding="utf-8"))
    payload["holdout_evaluated"] = True
    baseline.results_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        optimize_selected_model(
            frame,
            config=config,
            cv_results_path=baseline.results_path,
            model_dir=tmp_path / "optimized_models",
            results_path=tmp_path / "tuning.json",
            report_path=tmp_path / "tuning.md",
        )
    except DataValidationError as error:
        assert "holdout" in str(error).lower()
    else:
        raise AssertionError("Expected optimization to reject holdout-used CV data")
