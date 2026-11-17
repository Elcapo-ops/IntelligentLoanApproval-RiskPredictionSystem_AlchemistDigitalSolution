import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import load_config
from models.classifiers import BalancedGradientBoostingClassifier, build_classifiers
from models.registry import load_model_artifact
from models.train import train_models
from utils.exceptions import DataValidationError


def _training_frame(rows: int = 80) -> pd.DataFrame:
    config = load_config()
    frame: dict[str, object] = {
        "Loan_ID": [f"TR{index:04}" for index in range(rows)],
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
        "Education": ["Graduate" if index % 3 else "Not Graduate" for index in range(rows)],
        "Self_Employed": ["Yes" if index % 4 == 0 else "No" for index in range(rows)],
        "Credit_History": [float(index % 2) for index in range(rows)],
        "Property_Area": ["Rural", "Urban", "Semiurban", "Rural"] * (rows // 4),
        "Availability_Leak": ["unavailable"] * rows,
    }
    assert set(config["dataset"]["numeric_columns"]).issubset(frame)
    return pd.DataFrame(frame)


def _small_config() -> dict:
    config = deepcopy(load_config())
    config["training"]["test_size"] = 0.25
    config["training"]["cv_folds"] = 3
    config["models"]["parameters"]["logistic_regression"]["max_iter"] = 200
    config["models"]["parameters"]["random_forest"]["n_estimators"] = 8
    config["models"]["parameters"]["gradient_boosting"]["n_estimators"] = 6
    return config


def test_builds_all_four_fixed_seed_classifiers() -> None:
    config = _small_config()
    classifiers = build_classifiers(config)

    assert list(classifiers) == [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
    ]
    assert classifiers["logistic_regression"].random_state == 42
    assert classifiers["decision_tree"].random_state == 42
    assert classifiers["random_forest"].random_state == 42
    assert isinstance(
        classifiers["gradient_boosting"],
        BalancedGradientBoostingClassifier,
    )
    assert classifiers["gradient_boosting"].random_state == 42


def test_gradient_boosting_uses_balanced_fit_sample_weights(
    monkeypatch,
) -> None:
    import models.classifiers as classifiers_module

    class RecordingEstimator:
        def __init__(self, **_: object) -> None:
            pass

        def fit(self, X, y, sample_weight):
            self.classes_ = np.unique(y)
            self.n_features_in_ = X.shape[1]
            self.sample_weight_ = sample_weight
            return self

    monkeypatch.setattr(
        classifiers_module,
        "GradientBoostingClassifier",
        RecordingEstimator,
    )
    estimator = BalancedGradientBoostingClassifier(
        n_estimators=5,
        random_state=42,
    )
    X = np.arange(60, dtype=float).reshape(30, 2)
    y = np.array(["Y"] * 20 + ["N"] * 10)
    estimator.fit(X, y)

    weights = estimator.estimator_.sample_weight_
    assert estimator.classes_.tolist() == ["N", "Y"]
    assert weights[y == "N"].sum() == weights[y == "Y"].sum()


def test_training_uses_common_stratified_cv_and_reserves_holdout(
    tmp_path: Path,
) -> None:
    frame = _training_frame()
    config = _small_config()
    result = train_models(
        frame,
        config=config,
        output_dir=tmp_path / "models",
        report_path=tmp_path / "reports" / "training.md",
        results_path=tmp_path / "reports" / "results.json",
    )

    assert result.cv_results["model"].tolist() == config["models"]["candidates"]
    assert set(result.cv_results.columns) == {
        "model",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "roc_auc_mean",
        "roc_auc_std",
        "specificity_mean",
        "specificity_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
    }
    assert np.isfinite(result.cv_results.select_dtypes(include="number")).all().all()
    assert set(result.fitted_models) == set(config["models"]["candidates"])
    assert all(path.is_file() for path in result.model_paths.values())

    saved_results = json.loads(result.results_path.read_text(encoding="utf-8"))
    assert saved_results["holdout_evaluated"] is False
    assert saved_results["selection_metric"] is None
    assert saved_results["training_rows"] == 60
    assert saved_results["holdout_rows_reserved"] == 20
    assert saved_results["dataset_hash"]
    assert saved_results["model_version"] == config["training"]["model_version"]
    for model_name in config["models"]["candidates"]:
        loaded, metadata = load_model_artifact(
            model_name=model_name,
            model_version=config["training"]["model_version"],
            model_dir=tmp_path / "models",
            expected_target_mapping={"N": 0, "Y": 1},
        )
        assert loaded is not None
        assert metadata["dataset_hash"] == saved_results["dataset_hash"]
        assert metadata["random_seed"] == 42
        assert metadata["feature_schema"]["ApplicantIncome"] == "int64"
        assert set(metadata["dependency_versions"]) == {
            "python",
            "numpy",
            "pandas",
            "scikit-learn",
            "joblib",
        }
    report = result.report_path.read_text(encoding="utf-8")
    assert "training accuracy" in report
    assert "not passed to cross-validation, fit" in report
    assert "Gradient boosting uses balanced" in report
    assert "GradientBoosting" not in report or "gradient_boosting" in report


def test_training_results_repeat_with_fixed_seed(tmp_path: Path) -> None:
    frame = _training_frame()
    config = _small_config()
    first = train_models(
        frame,
        config=config,
        output_dir=tmp_path / "first_models",
        report_path=tmp_path / "first.md",
        results_path=tmp_path / "first.json",
    )
    second = train_models(
        frame,
        config=config,
        output_dir=tmp_path / "second_models",
        report_path=tmp_path / "second.md",
        results_path=tmp_path / "second.json",
    )

    pd.testing.assert_frame_equal(first.cv_results, second.cv_results)


def test_training_rejects_too_many_cv_folds_for_minority_class(
    tmp_path: Path,
) -> None:
    frame = _training_frame(40)
    config = _small_config()
    config["training"]["cv_folds"] = 50

    try:
        train_models(
            frame,
            config=config,
            output_dir=tmp_path / "models",
            report_path=tmp_path / "report.md",
            results_path=tmp_path / "results.json",
        )
    except DataValidationError as error:
        assert "smallest class" in str(error)
    else:
        raise AssertionError("Expected CV fold validation to reject this setup")
