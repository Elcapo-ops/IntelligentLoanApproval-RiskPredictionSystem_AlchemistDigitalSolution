"""Chronological, train-only model comparison for LendingClub default risk."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from default_risk.data import (
    CATEGORICAL_FEATURES,
    DATASET_FILENAME,
    DATASET_SOURCE,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RAW_FEATURE_COLUMNS,
    audit_lendingclub_dataset,
    prepare_lendingclub_frame,
    save_audit_report,
)
from config.settings import load_config
from models.registry import dependency_versions, save_model_artifact
from utils.exceptions import DataValidationError

MODEL_NAME = "lendingclub_default_risk"
MODEL_VERSION = "0.1.0"
DEFAULT_TARGET_MAPPING = {
    "positive_label": 1,
    "positive_status": "Charged Off",
    "negative_label": 0,
    "negative_status": "Fully Paid",
    "excluded_statuses": [
        "Current",
        "Late (16-30 days)",
        "Late (31-120 days)",
        "In Grace Period",
        "Default",
        "Does not meet the credit policy. Status:Fully Paid",
        "Does not meet the credit policy. Status:Charged Off",
    ],
}
RISK_THRESHOLDS = {
    "low_risk_max": 0.10,
    "moderate_risk_max": 0.25,
    "high_risk_max": 0.50,
}


def build_preprocessor() -> ColumnTransformer:
    """Create one fold-local preprocessing graph for all candidate estimators."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_candidate_models(random_state: int = 42) -> dict[str, Any]:
    """Return four reproducible models with the same input/preprocessing schema."""
    return {
        "logistic_regression": LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=120,
            solver="saga",
            random_state=random_state,
            n_jobs=1,
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=14,
            min_samples_leaf=50,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            max_depth=18,
            min_samples_leaf=40,
            n_estimators=80,
            n_jobs=2,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            l2_regularization=1.0,
            max_iter=50,
            max_leaf_nodes=15,
            min_samples_leaf=60,
            random_state=random_state,
        ),
    }


def _allocate_quotas(
    counts: dict[tuple[int, int, str], int],
    maximum: int,
) -> dict[tuple[int, int, str], int]:
    """Allocate deterministic proportional row quotas across year/class strata."""
    total = sum(counts.values())
    if total <= maximum:
        return dict(counts)
    exact = {key: count * maximum / total for key, count in counts.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remainder = maximum - sum(quotas.values())
    for key in sorted(exact, key=lambda item: exact[item] - quotas[item], reverse=True):
        if remainder <= 0:
            break
        if quotas[key] < counts[key]:
            quotas[key] += 1
            remainder -= 1
    return quotas


def _yearly_target_counts(audit: dict[str, Any], test_start_year: int) -> dict[tuple[int, int, str], int]:
    counts: dict[tuple[int, int, str], int] = {}
    for record in audit["monthly_target_counts"]:
        parsed = pd.to_datetime(record["issue_month"], format="%b-%Y", errors="coerce")
        if pd.isna(parsed):
            continue
        year = int(parsed.year)
        target = 1 if record["loan_status"] == "Charged Off" else 0
        partition = "test" if year >= test_start_year else "train"
        key = (year, target, partition)
        counts[key] = counts.get(key, 0) + int(record["count"])
    return counts


def _build_processed_data_and_samples(
    dataset_path: Path,
    processed_path: Path,
    audit: dict[str, Any],
    *,
    test_start_year: int,
    max_train_rows: int,
    max_test_rows: int,
    chunksize: int = 50_000,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Stream clean rows to disk and select deterministic temporal samples."""
    stratum_counts = _yearly_target_counts(audit, test_start_year)
    train_counts = {
        key: value for key, value in stratum_counts.items() if key[2] == "train"
    }
    test_counts = {
        key: value for key, value in stratum_counts.items() if key[2] == "test"
    }
    train_quotas = _allocate_quotas(train_counts, max_train_rows)
    test_quotas = _allocate_quotas(test_counts, max_test_rows)
    candidates: dict[tuple[int, int, str], pd.DataFrame] = {}
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    raw_columns = ["id", "loan_status", "issue_d", *RAW_FEATURE_COLUMNS]
    processed_rows = 0

    try:
        with gzip.open(processed_path, "wt", encoding="utf-8", newline="") as output:
            wrote_header = False
            for chunk in pd.read_csv(
                dataset_path,
                compression="gzip",
                usecols=raw_columns,
                chunksize=chunksize,
                low_memory=False,
            ):
                prepared = prepare_lendingclub_frame(chunk)
                if prepared.empty:
                    continue
                prepared = prepared.dropna(subset=["issue_d"])
                prepared["_year"] = prepared["issue_d"].dt.year
                prepared["_sample_key"] = pd.util.hash_pandas_object(
                    prepared["id"].astype(str), index=False
                ).to_numpy(dtype="uint64")
                processed_rows += len(prepared)
                prepared.drop(columns=["id", "_year", "_sample_key"]).to_csv(
                    output,
                    index=False,
                    header=not wrote_header,
                )
                wrote_header = True

                for (year, target), group in prepared.groupby(
                    ["_year", "default_target"], sort=False
                ):
                    year_value = int(year)
                    partition = "test" if year_value >= test_start_year else "train"
                    stratum = (year_value, int(target), partition)
                    quota = (test_quotas if partition == "test" else train_quotas).get(
                        stratum, 0
                    )
                    if not quota:
                        continue
                    previous = candidates.get(stratum)
                    merged = (
                        pd.concat([previous, group], ignore_index=True)
                        if previous is not None
                        else group.copy()
                    )
                    candidates[stratum] = merged.nsmallest(
                        quota, "_sample_key"
                    ).reset_index(drop=True)
    except (OSError, EOFError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise DataValidationError(
            f"Unable to stream the LendingClub dataset: {dataset_path.name}"
        ) from exc

    train_parts = [
        frame.drop(columns=["_year", "_sample_key"])
        for key, frame in candidates.items()
        if key[2] == "train"
    ]
    test_parts = [
        frame.drop(columns=["_year", "_sample_key"])
        for key, frame in candidates.items()
        if key[2] == "test"
    ]
    if not train_parts or not test_parts:
        raise DataValidationError(
            "Temporal split produced an empty partition; check issue dates and cutoff"
        )
    training = pd.concat(train_parts, ignore_index=True).sort_values("issue_d")
    testing = pd.concat(test_parts, ignore_index=True).sort_values("issue_d")
    if training["issue_d"].max() >= testing["issue_d"].min():
        raise DataValidationError("Temporal train/test partitions overlap in issue dates")
    return training.reset_index(drop=True), testing.reset_index(drop=True), processed_rows


def _build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("classifier", clone(estimator)),
        ]
    )


def _positive_probability(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise DataValidationError(
            f"Default class 1 is absent from fitted estimator classes {classes}"
        )
    probabilities = estimator.predict_proba(X)[:, classes.index(1)]
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise DataValidationError("Estimator produced invalid default probabilities")
    return probabilities


def _score_probabilities(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "specificity": specificity,
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "rows": int(len(y)),
        "default_prevalence": float(y.mean()),
    }


def compare_candidates(
    training: pd.DataFrame,
    *,
    random_state: int = 42,
    cv_folds: int = 3,
) -> tuple[dict[str, dict[str, float]], str]:
    """Compare four models on expanding time splits from training data only."""
    if cv_folds < 2:
        raise DataValidationError("At least two temporal CV folds are required")
    ordered = training.sort_values("issue_d").reset_index(drop=True)
    X = ordered.loc[:, FEATURE_COLUMNS]
    y = ordered["default_target"].astype(int)
    splitter = TimeSeriesSplit(n_splits=cv_folds)
    candidates = build_candidate_models(random_state)
    comparison: dict[str, dict[str, float]] = {}
    for name, estimator in candidates.items():
        fold_metrics: list[dict[str, float]] = []
        for train_indices, validation_indices in splitter.split(X):
            X_train, X_validation = X.iloc[train_indices], X.iloc[validation_indices]
            y_train, y_validation = y.iloc[train_indices], y.iloc[validation_indices]
            if y_train.nunique() != 2 or y_validation.nunique() != 2:
                raise DataValidationError(
                    f"Temporal CV fold for {name} does not contain both classes"
                )
            pipeline = _build_pipeline(estimator)
            pipeline.fit(X_train, y_train)
            probabilities = _positive_probability(pipeline, X_validation)
            metrics = _score_probabilities(y_validation, probabilities)
            fold_metrics.append(
                {
                    "pr_auc": metrics["pr_auc"],
                    "roc_auc": metrics["roc_auc"],
                    "recall": metrics["recall"],
                    "precision": metrics["precision"],
                    "f1": metrics["f1"],
                    "f2": float(
                        fbeta_score(
                            y_validation,
                            (probabilities >= 0.5).astype(int),
                            beta=2,
                            zero_division=0,
                        )
                    ),
                    "balanced_accuracy": metrics["balanced_accuracy"],
                }
            )
        comparison[name] = {
            f"{metric}_mean": float(np.mean([fold[metric] for fold in fold_metrics]))
            for metric in fold_metrics[0]
        }
        comparison[name].update(
            {
                f"{metric}_std": float(np.std([fold[metric] for fold in fold_metrics]))
                for metric in fold_metrics[0]
            }
        )
    selected = max(
        comparison,
        key=lambda name: (
            comparison[name]["f2_mean"],
            comparison[name]["pr_auc_mean"],
            comparison[name]["recall_mean"],
        ),
    )
    return comparison, selected


def _risk_category(probability: float) -> str:
    if probability < RISK_THRESHOLDS["low_risk_max"]:
        return "Low Risk"
    if probability < RISK_THRESHOLDS["moderate_risk_max"]:
        return "Moderate Risk"
    if probability < RISK_THRESHOLDS["high_risk_max"]:
        return "High Risk"
    return "Very High Risk"


def train_default_risk_model(
    *,
    project_root: Path,
    dataset_path: Path | None = None,
    test_start_year: int | None = None,
    max_train_rows: int | None = None,
    max_test_rows: int | None = None,
    cv_folds: int | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Audit, compare, calibrate, evaluate, explain, and register the model."""
    project_root = Path(project_root).resolve()
    config = load_config(project_root / "config.yaml")
    default_settings = config["default_risk"]
    test_start_year = (
        default_settings["test_start_year"]
        if test_start_year is None
        else test_start_year
    )
    max_train_rows = (
        default_settings["max_train_rows"]
        if max_train_rows is None
        else max_train_rows
    )
    max_test_rows = (
        default_settings["max_test_rows"]
        if max_test_rows is None
        else max_test_rows
    )
    cv_folds = default_settings["cv_folds"] if cv_folds is None else cv_folds
    random_state = config["project"]["random_seed"] if random_state is None else random_state
    risk_thresholds = dict(default_settings["risk_category_thresholds"])
    calibration_method = default_settings["calibration_method"]
    if max_train_rows < 1 or max_test_rows < 1:
        raise DataValidationError("Training and holdout sample limits must be positive")
    if cv_folds < 2:
        raise DataValidationError("At least two temporal CV folds are required")
    if dataset_path is None:
        dataset_path = (
            project_root / "data" / "default_risk" / "raw" / DATASET_FILENAME
        )
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.is_file() or dataset_path.is_symlink():
        raise DataValidationError(
            f"LendingClub data file not found or unsafe: {dataset_path}"
        )

    report_dir = project_root / "reports" / "default_risk"
    artifact_dir = project_root / "artifacts" / "models" / "default_risk"
    processed_path = (
        project_root
        / "data"
        / "default_risk"
        / "processed"
        / "lendingclub_matured.csv.gz"
    )
    audit = audit_lendingclub_dataset(dataset_path)
    audit_json, audit_markdown = save_audit_report(audit, report_dir)
    training, testing, processed_rows = _build_processed_data_and_samples(
        dataset_path,
        processed_path,
        audit,
        test_start_year=test_start_year,
        max_train_rows=max_train_rows,
        max_test_rows=max_test_rows,
    )
    if training["default_target"].nunique() != 2 or testing["default_target"].nunique() != 2:
        raise DataValidationError("Temporal train/test split must contain both classes")

    cv_results, selected_name = compare_candidates(
        training,
        random_state=random_state,
        cv_folds=cv_folds,
    )
    candidates = build_candidate_models(random_state)
    uncalibrated = _build_pipeline(candidates[selected_name])
    X_train = training.loc[:, FEATURE_COLUMNS]
    y_train = training["default_target"].astype(int)
    X_test = testing.loc[:, FEATURE_COLUMNS]
    y_test = testing["default_target"].astype(int)
    uncalibrated.fit(X_train, y_train)

    # Calibration uses only chronological training folds; the final holdout is
    # not supplied to the calibrator or used to select the model.
    calibration_cv = list(
        TimeSeriesSplit(n_splits=cv_folds).split(
            training.sort_values("issue_d").loc[:, FEATURE_COLUMNS]
        )
    )
    calibrated = CalibratedClassifierCV(
        estimator=_build_pipeline(candidates[selected_name]),
        method=calibration_method,
        cv=calibration_cv,
        ensemble=True,
    )
    ordered_training = training.sort_values("issue_d").reset_index(drop=True)
    calibrated.fit(
        ordered_training.loc[:, FEATURE_COLUMNS],
        ordered_training["default_target"].astype(int),
    )
    baseline_probabilities = _positive_probability(uncalibrated, X_test)
    calibrated_probabilities = _positive_probability(calibrated, X_test)
    baseline_metrics = _score_probabilities(y_test, baseline_probabilities)
    calibrated_metrics = _score_probabilities(y_test, calibrated_probabilities)
    final_metrics = {
        "uncalibrated": baseline_metrics,
        "calibrated": calibrated_metrics,
    }

    # Use training rows only for permutation importance. No holdout-based
    # feature/model/threshold selection is performed.
    explanation_rows = training.sample(
        n=min(1_000, len(training)), random_state=random_state
    )
    importance = permutation_importance(
        calibrated,
        explanation_rows.loc[:, FEATURE_COLUMNS],
        explanation_rows["default_target"].astype(int),
        scoring="average_precision",
        n_repeats=3,
        random_state=random_state,
        n_jobs=1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    report_dir.mkdir(parents=True, exist_ok=True)
    importance_path = report_dir / "permutation_importance.csv"
    importance_frame.to_csv(importance_path, index=False)
    plots_dir = project_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    calibration_plot_path = plots_dir / "calibration_curve.png"
    _save_calibration_plot(
        y_test,
        baseline_probabilities,
        calibrated_probabilities,
        calibration_plot_path,
    )
    importance_plot_path = plots_dir / "permutation_importance.png"
    importance_frame.head(15).sort_values("importance_mean").plot.barh(
        x="feature",
        y="importance_mean",
        legend=False,
        figsize=(9, 6),
        title="Training-sample permutation importance",
    )
    plt.xlabel("Decrease in average precision after permutation")
    plt.tight_layout()
    plt.savefig(importance_plot_path, dpi=150)
    plt.close()
    application_references: dict[str, Any] = {}
    for feature in NUMERIC_FEATURES:
        median = X_train[feature].median()
        application_references[feature] = (
            float(median) if pd.notna(median) else None
        )
    for feature in CATEGORICAL_FEATURES:
        modes = X_train[feature].mode(dropna=True)
        application_references[feature] = (
            str(modes.iloc[0]) if not modes.empty else None
        )

    metadata = {
        "dataset_source": DATASET_SOURCE,
        "dataset_file": dataset_path.name,
        "dataset_sha256": audit["sha256"],
        "target_variable": "loan_status",
        "target_definition": DEFAULT_TARGET_MAPPING,
        "target_mapping": {"0": "Fully Paid", "1": "Charged Off"},
        "feature_schema": list(FEATURE_COLUMNS),
        "feature_list": list(FEATURE_COLUMNS),
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "training_configuration": {
            "temporal_test_start_year": test_start_year,
            "max_train_sample_rows": max_train_rows,
            "max_test_sample_rows": max_test_rows,
            "actual_train_sample_rows": len(training),
            "actual_test_sample_rows": len(testing),
            "temporal_cv_folds": cv_folds,
            "selection_metric": (
                "training-only temporal CV F2 at threshold 0.50, "
                "with PR-AUC as the tie-breaker"
            ),
            "random_state": random_state,
            "processed_rows": processed_rows,
        },
        "preprocessing_configuration": {
            "numeric": "median imputation then StandardScaler, fitted inside each CV fold",
            "categorical": "most-frequent imputation then OneHotEncoder(handle_unknown='ignore'), fitted inside each CV fold",
            "feature_eligibility": "application/origination-time only; no target or post-origination fields",
        },
        "calibration_configuration": {
            "method": calibration_method,
            "cv": "expanding TimeSeriesSplit folds within training data",
            "holdout_used_for_calibration": False,
        },
        "risk_category_thresholds": risk_thresholds,
        "application_time_reference_values": application_references,
        "selected_model": selected_name,
        "evaluation_metrics": calibrated_metrics,
        "baseline_evaluation_metrics": baseline_metrics,
        "dependency_versions": dependency_versions(),
        "evaluation_date_utc": datetime.now(timezone.utc).isoformat(),
    }
    artifact_path, metadata_path = save_model_artifact(
        calibrated,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        model_dir=artifact_dir,
        metadata=metadata,
    )

    cv_path = report_dir / "model_comparison.json"
    cv_path.write_text(
        json.dumps(
            {
                "selected_model": selected_name,
                "selection_metric": "f2_mean at 0.50 threshold, then pr_auc_mean",
                "training_only_temporal_cv": cv_results,
                "final_holdout": final_metrics,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = report_dir / "model_evaluation.md"
    summary_path.write_text(
        _render_evaluation_report(
            audit=audit,
            training=training,
            testing=testing,
            selected_name=selected_name,
            cv_results=cv_results,
            final_metrics=final_metrics,
            artifact_path=artifact_path,
        ),
        encoding="utf-8",
    )
    calibration_path = report_dir / "calibration_report.md"
    calibration_path.write_text(
        _render_calibration_report(
            baseline_metrics,
            calibrated_metrics,
            calibration_plot_path.name,
        ),
        encoding="utf-8",
    )
    explainability_path = report_dir / "explainability_report.md"
    explainability_path.write_text(
        _render_explainability_report(importance_frame, importance_plot_path.name),
        encoding="utf-8",
    )
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "selected_model": selected_name,
        "dataset_audit_json": str(audit_json),
        "dataset_audit_markdown": str(audit_markdown),
        "processed_dataset": str(processed_path),
        "model_artifact": str(artifact_path),
        "model_metadata": str(metadata_path),
        "evaluation_report": str(summary_path),
        "calibration_report": str(calibration_path),
        "comparison_report": str(cv_path),
        "permutation_importance": str(importance_path),
        "permutation_importance_plot": str(importance_plot_path),
        "calibration_plot": str(calibration_plot_path),
        "explainability_report": str(explainability_path),
        "training_rows": len(training),
        "test_rows": len(testing),
        "cv_results": cv_results,
        "holdout_metrics": final_metrics,
        "target_counts": audit["target_class_counts"],
    }


def _render_evaluation_report(
    *,
    audit: dict[str, Any],
    training: pd.DataFrame,
    testing: pd.DataFrame,
    selected_name: str,
    cv_results: dict[str, dict[str, float]],
    final_metrics: dict[str, dict[str, Any]],
    artifact_path: Path,
) -> str:
    rows = [
        "# LendingClub default-risk evaluation",
        "",
        "## Protocol",
        "",
        f"- Source rows: {audit['row_count']:,}; resolved target rows: {audit['resolved_target_rows']:,}.",
        f"- Training sample: {len(training):,}; temporal holdout sample: {len(testing):,}.",
        f"- Training issue dates: {training.issue_d.min():%Y-%m} to {training.issue_d.max():%Y-%m}.",
        f"- Holdout issue dates: {testing.issue_d.min():%Y-%m} to {testing.issue_d.max():%Y-%m}.",
        "- Unresolved statuses are excluded, never mapped to non-default.",
        "- Candidate selection used expanding time-series CV on training rows only; the final holdout was not used for model selection, tuning, or calibration.",
        "- Holdout metrics represent the sampled later-issue-date rows with resolved labels; because outcomes are censored and unresolved loans were excluded, these are not fixed-horizon default-incidence estimates.",
        "",
        "## Temporal CV comparison (training data only)",
        "",
        "| Model | PR-AUC | ROC-AUC | Default recall | Precision | F1 | F2 | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in cv_results.items():
        rows.append(
            "| {name} | {pr:.4f} | {roc:.4f} | {recall:.4f} | {precision:.4f} | {f1:.4f} | {f2:.4f} | {bal:.4f} |".format(
                name=name,
                pr=metrics["pr_auc_mean"],
                roc=metrics["roc_auc_mean"],
                recall=metrics["recall_mean"],
                precision=metrics["precision_mean"],
                f1=metrics["f1_mean"],
                f2=metrics["f2_mean"],
                bal=metrics["balanced_accuracy_mean"],
            )
        )
    rows.extend(["", f"Selected model: **{selected_name}**, based on training-only temporal CV F2 at threshold 0.50, with PR-AUC as the tie-breaker.", ""])
    rows.extend(
        [
            "## Final temporal holdout metrics",
            "",
            "| Metric | Uncalibrated | Calibrated |",
            "|---|---:|---:|",
        ]
    )
    keys = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Default recall", "recall"),
        ("F1", "f1"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("Specificity", "specificity"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("Brier score", "brier_score"),
        ("Log loss", "log_loss"),
    ]
    for label, key in keys:
        rows.append(
            f"| {label} | {final_metrics['uncalibrated'][key]:.4f} | {final_metrics['calibrated'][key]:.4f} |"
        )
    rows.extend(
        [
            "",
            "### Confusion matrices at threshold 0.50",
            "",
            f"- Uncalibrated: `{final_metrics['uncalibrated']['confusion_matrix']}`",
            f"- Calibrated: `{final_metrics['calibrated']['confusion_matrix']}`",
            "",
            "Accuracy alone can conceal poor default detection when non-default loans are more common. Review default-class precision, recall, PR-AUC, calibration, and confusion matrices together. The 0.50 threshold is an analytical reference, not a lending policy.",
            "",
            f"Registered artifact: `{artifact_path}`",
            "",
            "Feature importance is descriptive predictive contribution, not causation. See `permutation_importance.csv`.",
            "",
        ]
    )
    return "\n".join(rows)


def _save_calibration_plot(
    y_true: pd.Series,
    baseline_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    output_path: Path,
) -> None:
    """Plot final-holdout reliability curves after the model is fixed."""
    baseline_fraction, baseline_mean = calibration_curve(
        y_true, baseline_probabilities, n_bins=10, strategy="quantile"
    )
    calibrated_fraction, calibrated_mean = calibration_curve(
        y_true, calibrated_probabilities, n_bins=10, strategy="quantile"
    )
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    axis.plot(
        baseline_mean,
        baseline_fraction,
        marker="o",
        label="Uncalibrated",
    )
    axis.plot(
        calibrated_mean,
        calibrated_fraction,
        marker="o",
        label="Sigmoid calibrated",
    )
    axis.set(
        xlabel="Mean predicted default probability",
        ylabel="Observed default fraction",
        title="Temporal holdout calibration",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _render_calibration_report(
    baseline: dict[str, Any],
    calibrated: dict[str, Any],
    plot_filename: str,
) -> str:
    improved_brier = calibrated["brier_score"] < baseline["brier_score"]
    improved_log_loss = calibrated["log_loss"] < baseline["log_loss"]
    return "\n".join(
        [
            "# Default-probability calibration report",
            "",
            "Calibration is estimated historical default probability among LendingClub loans with resolved `Charged Off` or `Fully Paid` labels. It is not a guarantee of future performance or a lending decision.",
            "",
            "The sigmoid calibrator was fit only on expanding temporal folds within the training sample. The final temporal holdout was not used for fitting.",
            "",
            "| Metric | Uncalibrated | Sigmoid calibrated |",
            "|---|---:|---:|",
            f"| Brier score | {baseline['brier_score']:.4f} | {calibrated['brier_score']:.4f} |",
            f"| Log loss | {baseline['log_loss']:.4f} | {calibrated['log_loss']:.4f} |",
            f"| ROC-AUC | {baseline['roc_auc']:.4f} | {calibrated['roc_auc']:.4f} |",
            f"| PR-AUC | {baseline['pr_auc']:.4f} | {calibrated['pr_auc']:.4f} |",
            "",
            f"- Brier score improved on the holdout: **{improved_brier}**.",
            f"- Log loss improved on the holdout: **{improved_log_loss}**.",
            "- This is one temporal holdout comparison and does not establish calibration stability across future populations or periods.",
            "",
            f"![Holdout calibration curve]({plot_filename})",
            "",
        ]
    )


def _render_explainability_report(
    importance: pd.DataFrame,
    plot_filename: str,
) -> str:
    top = importance.head(10)
    return "\n".join(
        [
            "# Default-risk explainability report",
            "",
            "Global permutation importance measures how much average precision changes when one input column is shuffled on a fixed, training-only sample. This measures predictive contribution in this dataset and is not causal evidence.",
            "",
            "Individual explanations are one-feature-at-a-time sensitivity checks against training-sample median/mode values. They can create unrealistic input combinations and are not causal or lending rationales.",
            "",
            "| Feature | Mean importance | Standard deviation |",
            "|---|---:|---:|",
            *[
                f"| `{row.feature}` | {row.importance_mean:.6f} | {row.importance_std:.6f} |"
                for row in top.itertuples(index=False)
            ],
            "",
            f"![Training-sample permutation importance]({plot_filename})",
            "",
        ]
    )


__all__ = [
    "DEFAULT_TARGET_MAPPING",
    "MODEL_NAME",
    "MODEL_VERSION",
    "RISK_THRESHOLDS",
    "build_candidate_models",
    "build_preprocessor",
    "compare_candidates",
    "train_default_risk_model",
]
