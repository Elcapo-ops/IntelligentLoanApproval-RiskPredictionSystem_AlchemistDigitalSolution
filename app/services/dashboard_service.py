"""Read-only services for dashboard statistics and model results."""

import json
import math
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import pandas as pd

from config.paths import get_project_paths
from config.settings import load_config, validate_config
from data.loader import load_csv
from utils.exceptions import DataValidationError


def load_historical_dashboard_data(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate historical application counts and observed label breakdowns."""
    settings = validate_config(config if config is not None else load_config())
    paths = get_project_paths(settings)
    frame = load_csv(paths.labeled_dataset)
    target_column = settings["dataset"]["target_column"]
    target_classes = settings["dataset"]["target_classes"]
    if target_column not in frame or frame[target_column].isna().any():
        raise DataValidationError("Historical dashboard target is missing or invalid")

    counts = frame[target_column].value_counts().reindex(target_classes, fill_value=0)
    if int(counts.sum()) != len(frame) or (counts <= 0).any():
        raise DataValidationError("Historical target counts do not match dataset rows")
    approval_count = int(counts["Y"])
    rejection_count = int(counts["N"])
    approval_by_area = (
        frame.groupby(["Property_Area", target_column], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=target_classes, fill_value=0)
        .rename(columns={"Y": "Y", "N": "N"})
        .rename_axis("Property_Area")
        .reset_index()
    )
    return {
        "application_count": int(len(frame)),
        "approval_count": approval_count,
        "rejection_count": rejection_count,
        "approval_rate": approval_count / len(frame),
        "rejection_rate": rejection_count / len(frame),
        "status_counts": counts.rename(index={"Y": "Approved", "N": "Rejected"}),
        "approval_by_area": approval_by_area,
        "source": str(paths.labeled_dataset),
        "target_column": target_column,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"Unable to read model results file: {path}") from exc
    if not isinstance(result, dict):
        raise DataValidationError(f"Invalid model results format: {path}")
    return result


def load_model_performance_data(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load only saved training-CV metrics for baseline and tuned pipelines."""
    settings = validate_config(config if config is not None else load_config())
    paths = get_project_paths(settings)
    cv_results = _read_json(paths.cv_results)
    tuning_results = _read_json(paths.tuning_results)
    if cv_results.get("holdout_evaluated") is not False:
        raise DataValidationError("Model CV report does not mark holdout unused")
    if tuning_results.get("holdout_used") is not False:
        raise DataValidationError("Tuning report indicates holdout use")
    if tuning_results.get("model_version") != settings["explainability"]["model_version"]:
        raise DataValidationError("Tuned results do not match configured model version")
    models = pd.DataFrame(cv_results.get("models", []))
    required = {
        "model",
        "accuracy_mean",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "roc_auc_mean",
        "roc_auc_std",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "specificity_mean",
    }
    if models.empty or not required.issubset(models.columns):
        raise DataValidationError("Saved CV results lack required performance metrics")
    selected_model = tuning_results.get("selected_model")
    selected_rows = models.loc[models["model"].eq(selected_model)]
    if len(selected_rows) != 1:
        raise DataValidationError(
            "Tuning report model does not resolve to one CV candidate"
        )
    baseline = selected_rows.iloc[0]
    for metric, cv_key in (
        ("balanced_accuracy", "balanced_accuracy_mean"),
        ("roc_auc", "roc_auc_mean"),
    ):
        baseline_key = f"baseline_{metric}_mean"
        if baseline_key not in tuning_results or not math.isclose(
            float(tuning_results[baseline_key]),
            float(baseline[cv_key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise DataValidationError(
                f"Tuning report baseline {metric} disagrees with candidate CV"
            )
    if (
        tuning_results.get("training_rows") != cv_results.get("training_rows")
        or tuning_results.get("cv_folds") != cv_results.get("cv_folds")
    ):
        raise DataValidationError(
            "Tuning and candidate CV reports disagree on the training protocol"
        )
    return {
        "candidate_cv": models,
        "tuning": tuning_results,
        "model_name": tuning_results["selected_model"],
        "model_version": tuning_results["model_version"],
        "cv_folds": tuning_results["cv_folds"],
        "training_rows": tuning_results["training_rows"],
        "cv_figure": (
            paths.figures_dir / "model_comparison_cv.png"
            if (paths.figures_dir / "model_comparison_cv.png").is_file()
            else None
        ),
    }


def load_holdout_evaluation_data(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse saved baseline holdout metrics, provenance, and actual figures."""
    settings = validate_config(config if config is not None else load_config())
    paths = get_project_paths(settings)
    report_path = paths.evaluation_report
    try:
        report = report_path.read_text(encoding="utf-8")
        report_timestamp = report_path.stat().st_mtime
    except OSError as exc:
        raise DataValidationError(
            f"Unable to read holdout evaluation report: {report_path}"
        ) from exc

    version_match = re.search(
        r"Selected candidate: `([^`]+)` version `([^`]+)`",
        report,
    )
    rows_match = re.search(r"\((\d+) rows\)", report)
    if not version_match or not rows_match:
        raise DataValidationError(
            "Evaluation report is missing model version or holdout row count"
        )
    metrics_section = re.search(
        r"## Selected model final holdout metrics\s+(.*?)(?=\n### |\Z)",
        report,
        flags=re.DOTALL,
    )
    if not metrics_section:
        raise DataValidationError("Evaluation report is missing holdout metrics")

    metrics: dict[str, float | None] = {}
    metric_pattern = re.compile(
        r"^\|\s*(accuracy|precision|recall|f1|roc_auc)\s*\|\s*"
        r"(Unavailable|[0-9]+(?:\.[0-9]+)?)\s*\|$",
        flags=re.MULTILINE,
    )
    for metric, value in metric_pattern.findall(metrics_section.group(1)):
        metrics[metric] = None if value == "Unavailable" else float(value)
    expected_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    if set(metrics) != expected_metrics:
        raise DataValidationError(
            "Evaluation report lacks one or more requested holdout metrics"
        )

    matrix_match = re.search(
        r"\| N \| (\d+) \| (\d+) \|\s*\n\| Y \| (\d+) \| (\d+) \|",
        report,
    )
    if not matrix_match:
        raise DataValidationError("Evaluation report is missing its confusion matrix")
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in matrix_match.groups()
    )
    holdout_rows = int(rows_match.group(1))
    if (
        true_negative + false_positive + false_negative + true_positive
        != holdout_rows
    ):
        raise DataValidationError(
            "Evaluation confusion matrix does not match reported holdout rows"
        )
    derived_metrics = {
        "accuracy": (true_negative + true_positive) / holdout_rows,
        "precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        ),
        "recall": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        ),
    }
    derived_metrics["f1"] = (
        2 * derived_metrics["precision"] * derived_metrics["recall"]
        / (derived_metrics["precision"] + derived_metrics["recall"])
        if derived_metrics["precision"] + derived_metrics["recall"]
        else 0.0
    )
    for metric, value in derived_metrics.items():
        if metrics[metric] is None or not math.isclose(
            metrics[metric],
            value,
            rel_tol=0.0,
            abs_tol=0.00006,
        ):
            raise DataValidationError(
                f"Evaluation report {metric} does not match its confusion matrix"
            )

    figures = {
        "confusion_matrix": paths.figures_dir / "confusion_matrix_holdout.png",
        "roc_curve": paths.figures_dir / "roc_curve_holdout.png",
        "precision_recall": paths.figures_dir / "precision_recall_holdout.png",
    }
    return {
        "model_name": version_match.group(1),
        "model_version": version_match.group(2),
        "positive_class": "Y",
        "negative_class": "N",
        "metrics": metrics,
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
        "holdout_rows": holdout_rows,
        "report_date": datetime.fromtimestamp(report_timestamp).astimezone().strftime(
            "%Y-%m-%d"
        ),
        "report_date_source": "evaluation report file modification timestamp",
        "report_path": report_path,
        "figures": {
            name: path if path.is_file() else None
            for name, path in figures.items()
        },
    }


def load_explainability_data(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load available saved reports and historical patterns for risk analysis."""
    settings = validate_config(config if config is not None else load_config())
    paths = get_project_paths(settings)
    warnings: list[str] = []
    importance = _load_optional_csv(
        paths.global_importance,
        {"feature", "importance_mean", "importance_fold_std"},
        warnings,
        "Global permutation importance",
    )
    groups = _load_optional_csv(
        paths.fairness_results,
        {
            "attribute",
            "group",
            "n",
            "eligible_for_comparison",
            "approval_rate",
            "observed_approval_rate",
            "true_positive_rate",
            "false_positive_rate",
        },
        warnings,
        "Fairness group metrics",
    )
    local = _load_optional_json(
        paths.local_explanations,
        warnings,
        "Local explanation output",
    )
    quality_report = _load_optional_text(
        paths.quality_report,
        warnings,
        "Data quality report",
    )
    explanation_report = _load_optional_text(
        paths.explainability_report,
        warnings,
        "Explainability report",
    )
    try:
        historical = load_historical_dashboard_data(settings)
        labeled_data = load_csv(paths.labeled_dataset)
        approval_patterns = {
            column: _historical_approval_pattern(
                labeled_data,
                column,
                settings["dataset"]["target_column"],
            )
            for column in ("Property_Area", "Credit_History", "Self_Employed")
            if column in labeled_data.columns
        }
    except (DataValidationError, OSError) as exc:
        historical = None
        approval_patterns = {}
        warnings.append(f"Historical approval patterns are unavailable: {exc}")

    shap_figures = sorted(
        path for path in paths.figures_dir.glob("shap*.png") if path.is_file()
    )
    fairness_figure = paths.figures_dir / "fairness_group_rates.png"
    importance_figure = paths.figures_dir / "global_permutation_importance.png"
    return {
        "importance": importance,
        "groups": groups,
        "local": local,
        "approval_patterns": approval_patterns,
        "historical_application_count": (
            historical["application_count"] if historical is not None else None
        ),
        "quality_report": quality_report,
        "explainability_report": explanation_report,
        "quality_limitations": _section_text(
            quality_report,
            "Interpretation and handling",
        ),
        "fairness_interpretation": _section_text(
            explanation_report,
            "Exploratory subgroup review",
        ),
        "shap_figures": shap_figures,
        "shap_status": _section_text(explanation_report, "SHAP"),
        "report": paths.explainability_report,
        "figure": fairness_figure if fairness_figure.is_file() else None,
        "importance_figure": importance_figure if importance_figure.is_file() else None,
        "warnings": warnings,
    }


def _load_optional_csv(
    path: Path,
    required_columns: set[str],
    warnings: list[str],
    label: str,
) -> pd.DataFrame | None:
    if not path.is_file():
        warnings.append(f"{label} report is missing: {path.name}")
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        warnings.append(f"{label} report could not be read: {exc}")
        return None
    if frame.empty or not required_columns.issubset(frame.columns):
        warnings.append(f"{label} report is empty or has an invalid schema.")
        return None
    return frame


def _load_optional_json(
    path: Path,
    warnings: list[str],
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        warnings.append(f"{label} is missing: {path.name}")
        return {}
    try:
        result = _read_json(path)
    except DataValidationError as exc:
        warnings.append(f"{label} could not be read: {exc}")
        return {}
    if isinstance(result, dict) and "predictions" in result:
        predictions = result["predictions"]
        if isinstance(predictions, dict):
            return predictions
    return result


def _load_optional_text(
    path: Path,
    warnings: list[str],
    label: str,
) -> str | None:
    if not path.is_file():
        warnings.append(f"{label} is missing: {path.name}")
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        warnings.append(f"{label} could not be read: {exc}")
        return None


def _section_text(report: str | None, section: str) -> str | None:
    if report is None:
        return None
    match = re.search(
        rf"^## {re.escape(section)}\s*\n(.*?)(?=^## |\Z)",
        report,
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _historical_approval_pattern(
    frame: pd.DataFrame,
    feature: str,
    target_column: str,
) -> pd.DataFrame:
    labels = frame[feature].astype("object").where(frame[feature].notna(), "Missing")
    grouped = (
        pd.DataFrame({"group": labels, "approved": frame[target_column].eq("Y")})
        .groupby("group", dropna=False, observed=False)["approved"]
        .agg(applications="size", approvals="sum", approval_rate="mean")
        .reset_index()
        .rename(columns={"group": feature})
    )
    grouped["approval_rate"] = grouped["approval_rate"].astype(float)
    return grouped


def load_report_text(
    report_path: Path,
    *,
    label: str,
) -> str:
    """Load a required Markdown source report with a clear data error."""
    try:
        return report_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"{label} is unavailable: {report_path}") from exc
