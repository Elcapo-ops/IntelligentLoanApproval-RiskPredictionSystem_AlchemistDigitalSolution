"""Cross-validated model training with a reserved final holdout."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from config.paths import get_project_paths
from config.settings import load_config, validate_config
from data.splitting import DataSplit, split_labeled_data
from models.classifiers import build_classifiers
from models.registry import (
    dataframe_sha256,
    dependency_versions,
    save_model_artifact,
)
from preprocessing.pipeline import build_model_pipeline
from utils.exceptions import DataValidationError


def _specificity_score(y_true: Any, y_pred: Any) -> float:
    """Calculate specificity treating N as the negative class."""
    negatives = y_true == "N"
    true_negatives = ((y_pred == "N") & negatives).sum()
    false_positives = ((y_pred == "Y") & negatives).sum()
    denominator = true_negatives + false_positives
    return float(true_negatives / denominator) if denominator else 0.0


@dataclass(frozen=True)
class TrainingResult:
    """Cross-validation results, training-fitted candidates, and reserved split."""

    split: DataSplit
    cv_results: pd.DataFrame
    fitted_models: dict[str, Any]
    model_paths: dict[str, Path]
    results_path: Path
    report_path: Path


def _json_safe_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Convert estimator parameters into JSON-serializable values."""
    return {
        key: value if value is None or isinstance(value, (str, int, float, bool))
        else str(value)
        for key, value in parameters.items()
    }


def _render_training_report(
    *,
    config: dict[str, Any],
    split: DataSplit,
    cv_results: pd.DataFrame,
    model_parameters: dict[str, dict[str, Any]],
) -> str:
    """Render measured CV results and explicit holdout safeguards."""
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "specificity",
        "precision",
        "recall",
        "f1",
    ]
    lines = [
        "# Phase 4.1 — Model training and cross-validation",
        "",
        "## Scope and leakage safeguards",
        "",
        f"- Labeled source: `{config['paths']['labeled_dataset']}` only.",
        f"- Target: `{split.target_column}`; classes: "
        f"`{', '.join(config['dataset']['target_classes'])}`.",
        f"- Training rows: {len(split.X_train)}; holdout rows reserved: "
        f"{len(split.X_test)}.",
        f"- Random seed: {config['project']['random_seed']}; stratified "
        f"{config['training']['cv_folds']}-fold CV on training rows only.",
        f"- Artifact version: `{config['training']['model_version']}`.",
        "- All candidates used the same target labels and identical CV folds.",
        "- The final holdout was not passed to cross-validation, fit, "
        "scoring, or model selection.",
        "- No training-accuracy metric was calculated or used for selection.",
        "- Preprocessing and feature engineering ran inside each estimator "
        "pipeline and were fit separately within CV folds.",
        "- Class weighting was used instead of pre-split resampling; no "
        "SMOTE was applied.",
        "- All four candidate pipelines were refit on the complete training "
        "partition for later holdout evaluation. No candidate is declared "
        "selected in this phase.",
        "",
        "## Fixed model configurations",
        "",
        "| Model | Parameters |",
        "|---|---|",
    ]
    for model_name, parameters in model_parameters.items():
        lines.append(f"| `{model_name}` | `{json.dumps(parameters, sort_keys=True)}` |")

    lines.extend(
        [
            "",
            "## Training-partition cross-validation results",
            "",
            "Values are mean ± standard deviation across the same stratified "
            "folds. These are CV estimates on training data, not final "
            "holdout performance.",
            "",
            "| Model | " + " | ".join(
                f"{metric} (mean ± SD)" for metric in metrics
            ) + " |",
            "|---|" + "|".join("---" for _ in metrics) + "|",
        ]
    )
    for _, row in cv_results.iterrows():
        values = [
            f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}"
            for metric in metrics
        ]
        lines.append(f"| `{row['model']}` | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "- Models are compared using stratified cross-validation metrics, "
            "not training accuracy.",
            "- Balanced accuracy and per-class precision/recall/F1 complement "
            "ROC-AUC because the labeled target is imbalanced.",
            "- Logistic regression, decision tree, and random forest use "
            "`class_weight='balanced'`. Gradient boosting uses balanced "
            "sample weights computed from the target labels supplied to each "
            "fold's `fit` call.",
            "- No threshold tuning, final model selection, or holdout "
            "evaluation was performed. Those decisions require a separate "
            "evaluation phase with the holdout used only as specified.",
            "- Results describe this small Kaggle labeled sample and are not "
            "evidence of causal effects or lending suitability.",
        ]
    )
    return "\n".join(lines) + "\n"


def train_models(
    labeled_frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    report_path: Path | None = None,
    results_path: Path | None = None,
) -> TrainingResult:
    """Evaluate four candidates on common CV folds, then fit on training only.

    The stratified holdout is reserved and never transformed, scored, or used
    to choose a model here. Each CV fold gets a fresh complete preprocessing
    and estimator pipeline. Candidate pipelines are finally refit on all
    training-partition rows for a later final-evaluation phase.
    """
    settings = validate_config(config if config is not None else load_config())
    paths = get_project_paths(settings)
    split = split_labeled_data(labeled_frame, config=settings)
    n_splits = settings["training"]["cv_folds"]
    least_populated_class = int(split.y_train.value_counts().min())
    if least_populated_class < n_splits:
        raise DataValidationError(
            f"Training partition has only {least_populated_class} samples in "
            f"its smallest class; {n_splits}-fold stratified CV requires at "
            "least one sample per fold"
        )

    seed = settings["project"]["random_seed"]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    common_folds = list(cv.split(split.X_train, split.y_train))
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
        "specificity": make_scorer(_specificity_score),
        "precision": make_scorer(
            precision_score,
            pos_label="Y",
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            pos_label="Y",
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            pos_label="Y",
            zero_division=0,
        ),
    }

    classifiers = build_classifiers(settings)
    fitted_models: dict[str, Any] = {}
    model_paths: dict[str, Path] = {}
    model_parameters: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    model_output_dir = Path(output_dir) if output_dir is not None else paths.models_dir
    model_output_dir.mkdir(parents=True, exist_ok=True)

    model_version = settings["training"]["model_version"]
    training_date = datetime.now(timezone.utc).isoformat()
    dataset_hash = dataframe_sha256(labeled_frame)
    feature_schema = {
        column: str(labeled_frame[column].dtype)
        for column in split.X_train.columns
    }
    target_mapping = {
        target: index
        for index, target in enumerate(
            sorted(settings["dataset"]["target_classes"])
        )
    }
    for model_name, classifier in classifiers.items():
        pipeline = build_model_pipeline(classifier, config=settings)
        cv_output = cross_validate(
            pipeline,
            split.X_train,
            split.y_train,
            cv=common_folds,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,
            error_score="raise",
        )
        row: dict[str, Any] = {"model": model_name}
        for metric in scoring:
            scores = cv_output[f"test_{metric}"]
            row[f"{metric}_mean"] = float(scores.mean())
            row[f"{metric}_std"] = float(scores.std(ddof=1))
        rows.append(row)

        class_weight = settings["models"]["parameters"][model_name].get(
            "class_weight"
        )
        model_parameters[model_name] = _json_safe_parameters(
            {
                **settings["models"]["parameters"][model_name],
                "random_state": seed,
                "imbalance_handling": (
                    "balanced sample weights computed inside each fit"
                    if model_name == "gradient_boosting"
                    else f"class_weight={class_weight!r}"
                ),
            }
        )
        pipeline.fit(split.X_train, split.y_train)
        model_path, _ = save_model_artifact(
            pipeline,
            model_name=model_name,
            model_version=model_version,
            model_dir=model_output_dir,
            metadata={
                "training_date": training_date,
                "dataset_hash": dataset_hash,
                "dataset_hash_algorithm": "sha256(pandas dataframe schema and values)",
                "random_seed": seed,
                "feature_schema": feature_schema,
                "target_mapping": target_mapping,
                "dependency_versions": dependency_versions(),
                "training_configuration": {
                    "test_size": settings["training"]["test_size"],
                    "cv_folds": n_splits,
                    "random_seed": seed,
                    "model_parameters": model_parameters[model_name],
                    "feature_engineering": settings["feature_engineering"],
                    "target_column": split.target_column,
                },
                "cross_validation_metrics": {
                    key: value
                    for key, value in row.items()
                    if key != "model"
                },
                "holdout_evaluated": False,
            },
        )
        fitted_models[model_name] = pipeline
        model_paths[model_name] = model_path

    cv_results = pd.DataFrame(rows)
    report_output = (
        Path(report_path)
        if report_path is not None
        else paths.root / "reports" / "model_training_report.md"
    )
    results_output = (
        Path(results_path)
        if results_path is not None
        else paths.reports_dir / "model_cv_results.json"
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    results_output.parent.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "target_column": split.target_column,
        "target_classes": settings["dataset"]["target_classes"],
        "random_seed": seed,
        "training_date": training_date,
        "dataset_hash": dataset_hash,
        "model_version": model_version,
        "cv_folds": n_splits,
        "training_rows": len(split.X_train),
        "holdout_rows_reserved": len(split.X_test),
        "holdout_evaluated": False,
        "selection_metric": None,
        "models": [
            {
                **row,
                "parameters": model_parameters[row["model"]],
                "artifact": model_paths[row["model"]].name,
                "model_version": model_version,
            }
            for row in rows
        ],
    }
    results_output.write_text(
        json.dumps(results_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    report_output.write_text(
        _render_training_report(
            config=settings,
            split=split,
            cv_results=cv_results,
            model_parameters=model_parameters,
        ),
        encoding="utf-8",
    )
    return TrainingResult(
        split=split,
        cv_results=cv_results,
        fitted_models=fitted_models,
        model_paths=model_paths,
        results_path=results_output,
        report_path=report_output,
    )
