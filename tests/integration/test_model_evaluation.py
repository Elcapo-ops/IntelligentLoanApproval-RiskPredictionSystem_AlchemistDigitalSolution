import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.settings import load_config
from evaluation.evaluate import (
    _calculate_metrics,
    evaluate_selected_model,
    select_model_from_cv,
)
from models.train import train_models
from utils.exceptions import DataValidationError


def _training_frame(rows: int = 80) -> pd.DataFrame:
    return pd.DataFrame(
        {
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
            "Education": [
                "Graduate" if index % 3 else "Not Graduate" for index in range(rows)
            ],
            "Self_Employed": [
                "Yes" if index % 4 == 0 else "No" for index in range(rows)
            ],
            "Credit_History": [float(index % 2) for index in range(rows)],
            "Property_Area": ["Rural", "Urban", "Semiurban", "Rural"] * (rows // 4),
        }
    )


def _small_config() -> dict:
    config = load_config()
    config["training"]["test_size"] = 0.25
    config["training"]["cv_folds"] = 3
    config["models"]["parameters"]["logistic_regression"]["max_iter"] = 200
    config["models"]["parameters"]["random_forest"]["n_estimators"] = 8
    config["models"]["parameters"]["gradient_boosting"]["n_estimators"] = 6
    return config


def test_cv_selector_uses_mean_balanced_accuracy_and_roc_auc_tiebreak() -> None:
    cv_results = pd.DataFrame(
        [
            {
                "model": "lower_auc",
                "balanced_accuracy_mean": 0.8,
                "roc_auc_mean": 0.7,
            },
            {
                "model": "higher_auc",
                "balanced_accuracy_mean": 0.8,
                "roc_auc_mean": 0.75,
            },
            {
                "model": "lower_balanced",
                "balanced_accuracy_mean": 0.79,
                "roc_auc_mean": 0.99,
            },
        ]
    )

    assert select_model_from_cv(cv_results) == "higher_auc"


def test_cv_selector_is_invariant_to_holdout_metrics_and_threshold_columns() -> None:
    cv_results = pd.DataFrame(
        [
            {
                "model": "cv_winner",
                "balanced_accuracy_mean": 0.8,
                "roc_auc_mean": 0.75,
                "holdout_balanced_accuracy": 0.1,
                "holdout_threshold": 0.2,
            },
            {
                "model": "holdout_winner",
                "balanced_accuracy_mean": 0.7,
                "roc_auc_mean": 0.7,
                "holdout_balanced_accuracy": 1.0,
                "holdout_threshold": 0.99,
            },
        ]
    )

    assert select_model_from_cv(cv_results) == "cv_winner"


def test_cv_selector_rejects_nonfinite_metrics() -> None:
    with pytest.raises(DataValidationError, match="must be finite"):
        select_model_from_cv(
            pd.DataFrame(
                [
                    {
                        "model": "invalid",
                        "balanced_accuracy_mean": np.nan,
                        "roc_auc_mean": 0.8,
                    }
                ]
            )
        )


def test_undefined_metrics_are_handled_explicitly() -> None:
    y_true = pd.Series(["Y", "Y", "Y"])
    y_pred = np.array(["N", "Y", "Y"])
    probabilities = np.array([0.1, 0.7, 0.9])

    metrics, _ = _calculate_metrics(y_true, y_pred, probabilities)

    assert metrics["specificity"] is None
    assert metrics["roc_auc"] is None
    assert metrics["precision"] == pytest.approx(1.0)


def test_evaluation_selects_then_scores_only_selected_holdout_model(
    tmp_path: Path,
) -> None:
    frame = _training_frame()
    config = _small_config()
    model_dir = tmp_path / "models"
    cv_path = tmp_path / "artifacts" / "model_cv_results.json"
    train_models(
        frame,
        config=config,
        output_dir=model_dir,
        report_path=tmp_path / "train.md",
        results_path=cv_path,
    )

    result = evaluate_selected_model(
        frame,
        config=config,
        cv_results_path=cv_path,
        model_dir=model_dir,
        output_dir=tmp_path / "reports",
    )

    cv_payload = json.loads(cv_path.read_text(encoding="utf-8"))
    expected_selection = select_model_from_cv(pd.DataFrame(cv_payload["models"]))
    assert result.selected_model == expected_selection
    assert result.metrics["positive_class"] == "Y"
    assert result.metrics["negative_class"] == "N"
    assert result.metrics["positive_probability_column_index"] == 1
    matrix = result.metrics["confusion_matrix"]
    tn, fp, fn, tp = matrix["tn"], matrix["fp"], matrix["fn"], matrix["tp"]
    assert result.metrics["accuracy"] == pytest.approx((tn + tp) / (tn + fp + fn + tp))
    assert result.metrics["precision"] == pytest.approx(tp / (tp + fp))
    assert result.metrics["recall"] == pytest.approx(tp / (tp + fn))
    assert result.metrics["specificity"] == pytest.approx(tn / (tn + fp))
    assert result.metrics["balanced_accuracy"] == pytest.approx(
        (tp / (tp + fn) + tn / (tn + fp)) / 2
    )

    comparison = pd.read_csv(result.comparison_path)
    selected_rows = comparison[comparison["model"] == result.selected_model]
    assert len(selected_rows) == 1
    assert selected_rows.iloc[0]["holdout_accuracy"] == pytest.approx(
        result.metrics["accuracy"]
    )
    assert comparison.loc[
        comparison["model"] != result.selected_model, "holdout_accuracy"
    ].isna().all()
    assert result.report_path.is_file()
    assert result.selection_path.is_file()
    assert all(path.is_file() for path in result.figure_paths.values())

    selection = json.loads(result.selection_path.read_text(encoding="utf-8"))
    assert selection["selection_metric"] == "balanced_accuracy_mean"
    assert selection["holdout_evaluated_after_selection"] is True
    report = result.report_path.read_text(encoding="utf-8")
    assert "Positive class: `Y`" in report
    assert "column 1" in report
    assert "fold variation" in report


def test_evaluation_rejects_cv_results_for_different_dataset(
    tmp_path: Path,
) -> None:
    frame = _training_frame()
    config = _small_config()
    cv_path = tmp_path / "model_cv_results.json"
    cv_path.write_text(
        json.dumps(
            {
                "holdout_evaluated": False,
                "target_column": "Loan_Status",
                "target_classes": ["Y", "N"],
                "dataset_hash": "different",
                "models": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="dataset hash"):
        evaluate_selected_model(
            frame,
            config=config,
            cv_results_path=cv_path,
            model_dir=tmp_path / "models",
            output_dir=tmp_path / "reports",
        )
