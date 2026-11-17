"""Training-only randomized hyperparameter search for the CV-selected model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate

from config.paths import get_project_paths
from config.settings import load_config, validate_config
from data.splitting import DataSplit, split_labeled_data
from evaluation.evaluate import _read_cv_results, select_model_from_cv
from models.classifiers import build_classifiers
from models.registry import (
    dataframe_sha256,
    dependency_versions,
    save_model_artifact,
)
from preprocessing.pipeline import build_model_pipeline
from utils.exceptions import DataValidationError


@dataclass(frozen=True)
class OptimizationResult:
    """Selected and optimized pipeline with training-only search results."""

    selected_model: str
    model_version: str
    best_estimator: Any
    best_params: dict[str, Any]
    best_cv_score: float
    baseline_cv_score: float
    cv_results: pd.DataFrame
    artifact_path: Path
    results_path: Path
    report_path: Path


def _render_optimization_report(
    *,
    selected_model: str,
    model_version: str,
    n_splits: int,
    seed: int,
    n_iter: int,
    parameter_distributions: dict[str, list[Any]],
    baseline_mean: float,
    baseline_std: float,
    baseline_roc_auc_mean: float,
    baseline_roc_auc_std: float,
    best_params: dict[str, Any],
    best_mean: float,
    best_std: float,
    best_roc_auc_mean: float,
    best_roc_auc_std: float,
    training_rows: int,
    holdout_rows: int,
) -> str:
    """Render search methodology and observed training-only results."""
    lines = [
        "# Phase 5.2 — Hyperparameter optimization",
        "",
        "## Protocol",
        "",
        f"- Candidate selected from the Phase 4.1 training-only CV results: "
        f"`{selected_model}`.",
        "- No holdout features, labels, or scores were used in this search, "
        "comparison, or parameter selection.",
        "- The same holdout was evaluated during Phase 5.1 for the baseline "
        "model before this optimization. Those results were not consulted for "
        "Phase 5.2, but the holdout is not pristine and any future evaluation "
        "must be interpreted with that limitation.",
        f"- Training rows: {training_rows}; reserved holdout rows: {holdout_rows}.",
        f"- Search used {n_splits}-fold shuffled stratified CV, seed {seed}, "
        "on the training partition only.",
        "- Search scoring: mean CV balanced accuracy (`refit` criterion); "
        "mean CV ROC-AUC is reported as a secondary comparison.",
        "- Each search candidate was a complete pipeline; feature engineering, "
        "imputation, scaling, and encoding were fitted within each CV fold.",
        "- The untuned baseline was scored on the same folds and metric.",
        "- No threshold optimization was performed.",
        "",
        "## Search space and budget",
        "",
        f"RandomizedSearchCV sampled {n_iter} parameter combinations from:",
        "",
        "| Pipeline parameter | Values |",
        "|---|---|",
    ]
    for parameter, values in parameter_distributions.items():
        lines.append(
            f"| `{parameter}` | `{json.dumps(values, sort_keys=True)}` |"
        )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Model | CV balanced accuracy (mean ± SD) | CV ROC-AUC (mean ± SD) |",
            "|---|---:|---:|",
            f"| Untuned `{selected_model}` | {baseline_mean:.4f} ± "
            f"{baseline_std:.4f} | {baseline_roc_auc_mean:.4f} ± "
            f"{baseline_roc_auc_std:.4f} |",
            f"| Tuned `{selected_model}` | {best_mean:.4f} ± {best_std:.4f} | "
            f"{best_roc_auc_mean:.4f} ± {best_roc_auc_std:.4f} |",
            "",
            f"Balanced-accuracy change: **{best_mean - baseline_mean:+.4f}** "
            "CV points (tuned mean minus untuned mean).",
            "",
            "Best parameters:",
            "",
            "```json",
            json.dumps(best_params, indent=2, sort_keys=True),
            "```",
            "",
            f"Saved optimized complete pipeline as version `{model_version}`.",
            "This optimized version has not yet been evaluated on the holdout. "
            "No holdout-based model selection or threshold tuning was conducted.",
            "",
            "CV scores are estimates from this dataset and search protocol; "
            "they do not guarantee generalization or establish suitability "
            "for lending decisions.",
        ]
    )
    return "\n".join(lines) + "\n"


def optimize_selected_model(
    labeled_frame: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
    cv_results_path: Path | None = None,
    model_dir: Path | None = None,
    results_path: Path | None = None,
    report_path: Path | None = None,
) -> OptimizationResult:
    """Tune the CV-selected Random Forest using training partition only.

    Selection and candidate search use CV records and the training partition;
    no holdout variable is passed to the search or baseline scorer.
    """
    settings = validate_config(config if config is not None else load_config())
    paths = get_project_paths(settings)
    cv_path = (
        Path(cv_results_path)
        if cv_results_path is not None
        else paths.reports_dir / "model_cv_results.json"
    )
    cv_payload = _read_cv_results(cv_path)
    if cv_payload.get("holdout_evaluated") is not False:
        raise DataValidationError("CV results must record an untouched holdout")
    if cv_payload.get("target_column") != settings["dataset"]["target_column"]:
        raise DataValidationError("CV results target column does not match config")
    if cv_payload.get("target_classes") != settings["dataset"]["target_classes"]:
        raise DataValidationError("CV results target classes do not match config")

    dataset_hash = dataframe_sha256(labeled_frame)
    if cv_payload.get("dataset_hash") != dataset_hash:
        raise DataValidationError(
            "CV results dataset hash does not match the labeled input"
        )
    selected_model = select_model_from_cv(
        pd.DataFrame(cv_payload["models"]),
        model_names=settings["models"]["candidates"],
    )
    if selected_model != "random_forest":
        raise DataValidationError(
            "Phase 5.2 search is configured for the CV-selected Random Forest; "
            f"CV selected {selected_model!r}"
        )

    split: DataSplit = split_labeled_data(labeled_frame, config=settings)
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

    untuned_classifier = build_classifiers(settings)[selected_model]
    if not isinstance(untuned_classifier, RandomForestClassifier):
        raise DataValidationError(
            "Random Forest classifier factory returned an unexpected estimator"
        )
    untuned_pipeline = build_model_pipeline(untuned_classifier, config=settings)
    baseline_cv = cross_validate(
        untuned_pipeline,
        split.X_train,
        split.y_train,
        cv=common_folds,
        scoring={
            "balanced_accuracy": "balanced_accuracy",
            "roc_auc": "roc_auc",
        },
        return_train_score=False,
        n_jobs=1,
        error_score="raise",
    )
    baseline_mean = float(baseline_cv["test_balanced_accuracy"].mean())
    baseline_std = float(
        baseline_cv["test_balanced_accuracy"].std(ddof=1)
    )
    baseline_roc_auc_mean = float(baseline_cv["test_roc_auc"].mean())
    baseline_roc_auc_std = float(baseline_cv["test_roc_auc"].std(ddof=1))

    optimization = settings["optimization"]
    parameter_distributions = optimization["parameter_distributions"]
    search = RandomizedSearchCV(
        estimator=build_model_pipeline(
            build_classifiers(settings)[selected_model],
            config=settings,
        ),
        param_distributions=parameter_distributions,
        n_iter=optimization["n_iter"],
        scoring={
            "balanced_accuracy": "balanced_accuracy",
            "roc_auc": "roc_auc",
        },
        refit="balanced_accuracy",
        cv=common_folds,
        random_state=seed,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(split.X_train, split.y_train)

    best_index = int(search.best_index_)
    cv_results = pd.DataFrame(search.cv_results_)
    best_mean = float(search.best_score_)
    best_std = float(search.cv_results_["std_test_balanced_accuracy"][best_index])
    best_roc_auc_mean = float(search.cv_results_["mean_test_roc_auc"][best_index])
    best_roc_auc_std = float(search.cv_results_["std_test_roc_auc"][best_index])
    cv_fold_scores = [
        {
            "fold": fold_index + 1,
            "untuned_balanced_accuracy": float(
                baseline_cv["test_balanced_accuracy"][fold_index]
            ),
            "untuned_roc_auc": float(baseline_cv["test_roc_auc"][fold_index]),
            "tuned_balanced_accuracy": float(
                search.cv_results_[f"split{fold_index}_test_balanced_accuracy"][
                    best_index
                ]
            ),
            "tuned_roc_auc": float(
                search.cv_results_[f"split{fold_index}_test_roc_auc"][best_index]
            ),
        }
        for fold_index in range(n_splits)
    ]
    candidate_cv_results = [
        {
            "rank": int(search.cv_results_["rank_test_balanced_accuracy"][index]),
            "params": parameters,
            "balanced_accuracy_mean": float(
                search.cv_results_["mean_test_balanced_accuracy"][index]
            ),
            "balanced_accuracy_std": float(
                search.cv_results_["std_test_balanced_accuracy"][index]
            ),
            "roc_auc_mean": float(search.cv_results_["mean_test_roc_auc"][index]),
            "roc_auc_std": float(search.cv_results_["std_test_roc_auc"][index]),
        }
        for index, parameters in enumerate(search.cv_results_["params"])
    ]

    model_version = settings["training"]["optimized_model_version"]
    output_model_dir = Path(model_dir) if model_dir is not None else paths.models_dir
    artifact_path, _ = save_model_artifact(
        search.best_estimator_,
        model_name=selected_model,
        model_version=model_version,
        model_dir=output_model_dir,
        metadata={
            "training_date": datetime.now(timezone.utc).isoformat(),
            "dataset_hash": dataset_hash,
            "dataset_hash_algorithm": "sha256(pandas dataframe schema and values)",
            "random_seed": seed,
            "feature_schema": {
                column: str(labeled_frame[column].dtype)
                for column in split.X_train.columns
            },
            "target_mapping": {
                target: index
                for index, target in enumerate(
                    sorted(settings["dataset"]["target_classes"])
                )
            },
            "dependency_versions": dependency_versions(),
            "training_configuration": {
                "target_column": split.target_column,
                "training_rows": len(split.X_train),
                "holdout_rows_reserved": len(split.X_test),
                "holdout_used": False,
                "cv_folds": n_splits,
                "random_seed": seed,
                "selection_metric": "balanced_accuracy_mean",
                "search": "RandomizedSearchCV",
                "n_iter": optimization["n_iter"],
                "scoring": optimization["scoring"],
                "parameter_distributions": parameter_distributions,
                "best_params": search.best_params_,
                "best_balanced_accuracy": best_mean,
                "best_balanced_accuracy_std": best_std,
                "best_roc_auc": best_roc_auc_mean,
                "best_roc_auc_std": best_roc_auc_std,
                "untuned_balanced_accuracy": baseline_mean,
                "untuned_balanced_accuracy_std": baseline_std,
                "untuned_roc_auc": baseline_roc_auc_mean,
                "untuned_roc_auc_std": baseline_roc_auc_std,
            },
            "holdout_evaluated": False,
        },
    )

    output_results_path = (
        Path(results_path)
        if results_path is not None
        else paths.tuning_results
    )
    output_report_path = (
        Path(report_path)
        if report_path is not None
        else paths.tuning_report
    )
    output_results_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "selected_model": selected_model,
        "model_version": model_version,
        "selection_source": "training-only cross-validation",
        "selection_metric": "balanced_accuracy_mean",
        "holdout_used": False,
        "target_column": split.target_column,
        "dataset_hash": dataset_hash,
        "random_seed": seed,
        "cv_folds": n_splits,
        "training_rows": len(split.X_train),
        "holdout_rows_reserved": len(split.X_test),
        "search": "RandomizedSearchCV",
        "n_iter": optimization["n_iter"],
        "scoring": optimization["scoring"],
        "parameter_distributions": parameter_distributions,
        "best_params": search.best_params_,
        "baseline_balanced_accuracy_mean": baseline_mean,
        "baseline_balanced_accuracy_std": baseline_std,
        "baseline_roc_auc_mean": baseline_roc_auc_mean,
        "baseline_roc_auc_std": baseline_roc_auc_std,
        "tuned_balanced_accuracy_mean": best_mean,
        "tuned_balanced_accuracy_std": best_std,
        "tuned_roc_auc_mean": best_roc_auc_mean,
        "tuned_roc_auc_std": best_roc_auc_std,
        "balanced_accuracy_delta": best_mean - baseline_mean,
        "cv_fold_scores": cv_fold_scores,
        "candidate_cv_results": candidate_cv_results,
        "best_index": best_index,
        "artifact": artifact_path.name,
    }
    output_results_path.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_report_path.write_text(
        _render_optimization_report(
            selected_model=selected_model,
            model_version=model_version,
            n_splits=n_splits,
            seed=seed,
            n_iter=optimization["n_iter"],
            parameter_distributions=parameter_distributions,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            baseline_roc_auc_mean=baseline_roc_auc_mean,
            baseline_roc_auc_std=baseline_roc_auc_std,
            best_params=search.best_params_,
            best_mean=best_mean,
            best_std=best_std,
            best_roc_auc_mean=best_roc_auc_mean,
            best_roc_auc_std=best_roc_auc_std,
            training_rows=len(split.X_train),
            holdout_rows=len(split.X_test),
        ),
        encoding="utf-8",
    )
    return OptimizationResult(
        selected_model=selected_model,
        model_version=model_version,
        best_estimator=search.best_estimator_,
        best_params=search.best_params_,
        best_cv_score=best_mean,
        baseline_cv_score=baseline_mean,
        cv_results=cv_results,
        artifact_path=artifact_path,
        results_path=output_results_path,
        report_path=output_report_path,
    )
