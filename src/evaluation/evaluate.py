"""CV-based candidate selection and final holdout evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config.paths import get_project_paths
from config.settings import load_config, validate_config
from data.splitting import DataSplit, split_labeled_data
from models.registry import (
    dataframe_sha256,
    dependency_versions,
    load_model_artifact,
)
from utils.exceptions import ArtifactError, DataValidationError

POSITIVE_LABEL = "Y"
NEGATIVE_LABEL = "N"
SELECTION_METRIC = "balanced_accuracy_mean"


@dataclass(frozen=True)
class EvaluationResult:
    """Selected candidate's holdout scores, CV comparison, and outputs."""

    selected_model: str
    selected_version: str
    metrics: dict[str, Any]
    classification_report: dict[str, Any]
    cv_comparison: pd.DataFrame
    comparison_path: Path
    report_path: Path
    figure_paths: dict[str, Path]
    selection_path: Path


def select_model_from_cv(
    cv_results: pd.DataFrame,
    model_names: list[str] | None = None,
) -> str:
    """Select by mean CV balanced accuracy, then mean CV ROC-AUC.

    The selector accepts CV results only. It has no holdout parameter or
    access to holdout labels/predictions.
    """
    required = {"model", "balanced_accuracy_mean", "roc_auc_mean"}
    missing_columns = required.difference(cv_results.columns)
    if missing_columns:
        raise DataValidationError(
            f"CV results are missing selection columns: {sorted(missing_columns)}"
        )
    candidates = cv_results.copy()
    if model_names is not None:
        candidates = candidates[candidates["model"].isin(model_names)]
    if candidates.empty:
        raise DataValidationError("No CV model candidates are available")
    scores = candidates[["balanced_accuracy_mean", "roc_auc_mean"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(scores).all():
        raise DataValidationError("CV selection metrics must be finite")
    candidates = candidates.sort_values(
        ["balanced_accuracy_mean", "roc_auc_mean", "model"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return str(candidates.iloc[0]["model"])


def _read_cv_results(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"Unable to read CV results: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise DataValidationError(f"Invalid CV results format: {path}")
    return payload


def _positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ArtifactError("Selected model does not expose fitted class ordering")
    class_labels = list(classes)
    if POSITIVE_LABEL not in class_labels:
        raise ArtifactError(
            f"Selected model has no positive class {POSITIVE_LABEL!r}: {class_labels}"
        )
    probabilities = model.predict_proba(X)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_labels):
        raise ArtifactError("Model probability output does not match class ordering")
    positive_index = class_labels.index(POSITIVE_LABEL)
    return np.asarray(probabilities[:, positive_index], dtype=float)


def _calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    positive_probabilities: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    labels = [NEGATIVE_LABEL, POSITIVE_LABEL]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    specificity_denominator = tn + fp
    specificity = (
        float(tn / specificity_denominator)
        if specificity_denominator
        else None
    )
    class_recall_values = [
        true_positive / actual_count
        for true_positive, actual_count in (
            (tn, tn + fp),
            (tp, tp + fn),
        )
        if actual_count
    ]
    positive_present = bool((y_true == POSITIVE_LABEL).any())
    negative_present = bool((y_true == NEGATIVE_LABEL).any())
    roc_auc: float | None = None
    if positive_present and negative_present:
        roc_auc = float(
            roc_auc_score(
                (y_true == POSITIVE_LABEL).astype(int),
                positive_probabilities,
            )
        )
        positive_count = int((y_true == POSITIVE_LABEL).sum())
        negative_count = int((y_true == NEGATIVE_LABEL).sum())
        ranks = pd.Series(positive_probabilities).rank(method="average").to_numpy()
        positive_rank_sum = float(
            ranks[(y_true.to_numpy() == POSITIVE_LABEL)].sum()
        )
        independent_roc_auc = (
            positive_rank_sum - positive_count * (positive_count + 1) / 2
        ) / (positive_count * negative_count)
        if not np.isclose(roc_auc, independent_roc_auc, rtol=0.0, atol=1e-12):
            raise AssertionError("Independent ROC-AUC verification disagreed")

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        ),
        "roc_auc": roc_auc,
        "specificity": specificity,
        "balanced_accuracy": float(np.mean(class_recall_values)),
        "confusion_matrix": {
            "labels": labels,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "positive_class": POSITIVE_LABEL,
        "negative_class": NEGATIVE_LABEL,
        "positive_probability_column_index": None,
    }

    independently_calculated = {
        "accuracy": (tn + tp) / (tn + fp + fn + tp),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "balanced_accuracy": (
            (
                (tp / (tp + fn) if tp + fn else 0.0)
                + (tn / (tn + fp) if tn + fp else 0.0)
            )
            / (int(bool(tp + fn)) + int(bool(tn + fp)))
        ),
    }
    for metric, independent_value in independently_calculated.items():
        measured = metrics[metric]
        if independent_value is None:
            if measured is not None:
                raise AssertionError(f"Independent {metric} verification disagreed")
        elif not np.isclose(measured, independent_value, rtol=0.0, atol=1e-12):
            raise AssertionError(f"Independent {metric} verification disagreed")
    return metrics, matrix


def _save_figures(
    cv_comparison: pd.DataFrame,
    confusion: np.ndarray,
    y_true: pd.Series,
    positive_probabilities: np.ndarray,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    figure_paths: dict[str, Path] = {}

    figure, axis = plt.subplots(figsize=(11, 6))
    plot_data = cv_comparison.set_index("model")
    x = np.arange(len(plot_data))
    width = 0.36
    for offset, metric, color, label in (
        (-width / 2, "balanced_accuracy", "#4C72B0", "Balanced accuracy"),
        (width / 2, "roc_auc", "#55A868", "ROC-AUC"),
    ):
        axis.bar(
            x + offset,
            plot_data[f"{metric}_mean"],
            width,
            yerr=plot_data[f"{metric}_std"],
            capsize=4,
            color=color,
            label=f"{label} (CV mean ± SD)",
        )
    axis.set_xticks(x, plot_data.index, rotation=15, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_xlabel("Candidate model")
    axis.set_title("Stratified cross-validation model comparison")
    axis.legend()
    figure.tight_layout()
    figure_paths["model_comparison"] = output_dir / "model_comparison_cv.png"
    figure.savefig(figure_paths["model_comparison"], dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[NEGATIVE_LABEL, POSITIVE_LABEL],
        yticklabels=[NEGATIVE_LABEL, POSITIVE_LABEL],
        ax=axis,
    )
    axis.set_xlabel("Predicted loan status")
    axis.set_ylabel("Actual loan status")
    axis.set_title("Selected model — holdout confusion matrix")
    figure.tight_layout()
    figure_paths["confusion_matrix"] = output_dir / "confusion_matrix_holdout.png"
    figure.savefig(figure_paths["confusion_matrix"], dpi=160)
    plt.close(figure)

    y_binary = (y_true == POSITIVE_LABEL).astype(int)
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_binary,
        positive_probabilities,
    )
    auc = roc_auc_score(y_binary, positive_probabilities)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"Selected model (AUC = {auc:.3f})",
        color="#4C72B0",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="No-skill")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate")
    axis.set_title("Selected model — holdout ROC curve")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure_paths["roc_curve"] = output_dir / "roc_curve_holdout.png"
    figure.savefig(figure_paths["roc_curve"], dpi=160)
    plt.close(figure)

    precision, recall, _ = precision_recall_curve(y_binary, positive_probabilities)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, color="#C44E52", label="Selected model")
    axis.axhline(
        y=float(y_binary.mean()),
        linestyle="--",
        color="gray",
        label="Positive-class prevalence",
    )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Recall",
        ylabel="Precision",
    )
    axis.set_title("Selected model — holdout precision-recall curve")
    axis.legend(loc="best")
    figure.tight_layout()
    figure_paths["precision_recall"] = output_dir / "precision_recall_holdout.png"
    figure.savefig(figure_paths["precision_recall"], dpi=160)
    plt.close(figure)
    return figure_paths


def _render_report(
    *,
    selected_model: str,
    model_version: str,
    selection_scores: Mapping[str, Any],
    cv_comparison: pd.DataFrame,
    metrics: Mapping[str, Any],
    class_report: Mapping[str, Any],
    probability_classes: list[str],
    positive_index: int,
    holdout_rows: int,
) -> str:
    lines = [
        "# Phase 5.1 — Model evaluation",
        "",
        "## Protocol",
        "",
        f"- Positive class: `{POSITIVE_LABEL}` (approved); negative class: "
        f"`{NEGATIVE_LABEL}` (rejected).",
        f"- Selection used training-partition CV only: highest mean "
        f"stratified CV balanced accuracy, with mean ROC-AUC as the tie-breaker.",
        f"- Selected candidate: `{selected_model}` version `{model_version}`.",
        f"- Selection scores: balanced accuracy "
        f"{selection_scores['balanced_accuracy_mean']:.6f}; ROC-AUC "
        f"{selection_scores['roc_auc_mean']:.6f}.",
        f"- The final stratified holdout was evaluated after selection "
        f"({holdout_rows} rows); it was not used for preprocessing, CV, "
        f"selection, or threshold tuning.",
        f"- Fitted class ordering: `{probability_classes}`; probability column "
        f"{positive_index} was used for positive class `{POSITIVE_LABEL}`.",
        "- Undefined precision/recall/F1 use `zero_division=0`; ROC-AUC is "
        "reported as unavailable if the holdout contains only one class; "
        "specificity is unavailable if no negatives are present.",
        "- Independent confusion-matrix calculations were compared against "
        "the metric-library results.",
        "",
        "## Cross-validation comparison",
        "",
        "Values are mean ± standard deviation across identical stratified "
        "training folds; standard deviation summarizes fold variation. "
        "The holdout is not included in this comparison.",
        "",
        "| Model | Accuracy | Balanced accuracy | ROC-AUC | Specificity | Precision (Y) | Recall (Y) | F1 (Y) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cv_metrics = (
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "specificity",
        "precision",
        "recall",
        "f1",
    )
    for _, row in cv_comparison.iterrows():
        cells = [
            f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}"
            for metric in cv_metrics
        ]
        lines.append(f"| `{row['model']}` | " + " | ".join(cells) + " |")

    holdout_metrics = (
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "specificity",
        "precision",
        "recall",
        "f1",
    )
    lines.extend(
        [
            "",
            "## Selected model final holdout metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for metric in holdout_metrics:
        value = metrics[metric]
        lines.append(f"| {metric} | {'Unavailable' if value is None else f'{value:.4f}'} |")
    lines.extend(
        [
            "",
            "### Confusion matrix",
            "",
            "Rows are actual classes `[N, Y]`; columns are predicted classes `[N, Y]`.",
            "",
            "| Actual / Predicted | N | Y |",
            "|---|---:|---:|",
            f"| N | {metrics['confusion_matrix']['tn']} | {metrics['confusion_matrix']['fp']} |",
            f"| Y | {metrics['confusion_matrix']['fn']} | {metrics['confusion_matrix']['tp']} |",
            "",
            "### Classification report",
            "",
            "| Class | Precision | Recall | F1-score | Support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in (NEGATIVE_LABEL, POSITIVE_LABEL):
        row = class_report[label]
        lines.append(
            f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1-score']:.4f} | {int(row['support'])} |"
        )
    lines.extend(
        [
            "",
            f"Macro average: precision {class_report['macro avg']['precision']:.4f}, "
            f"recall {class_report['macro avg']['recall']:.4f}, "
            f"F1 {class_report['macro avg']['f1-score']:.4f}.",
            "",
            "## Figures and artifacts",
            "",
                        "- [CV model comparison with fold variation](plots/model_comparison_cv.png)",
            "- [Selected model holdout confusion matrix](plots/confusion_matrix_holdout.png)",
            "- [Selected model holdout ROC curve](plots/roc_curve_holdout.png)",
            "- [Selected model holdout precision-recall curve](plots/precision_recall_holdout.png)",
            "- Machine-readable model comparison: `model_comparison.csv`.",
            "",
            "These evaluation results describe this dataset and fixed modeling "
            "protocol. They do not establish causal relationships, fairness, "
            "or suitability for real lending decisions.",
        ]
    )
    return "\n".join(lines) + "\n"


def evaluate_selected_model(
    labeled_frame: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
    cv_results_path: Path | None = None,
    model_dir: Path | None = None,
    output_dir: Path | None = None,
) -> EvaluationResult:
    """Select by CV metrics, then evaluate the selected pipeline on holdout.

    Holdout labels are not accessed until the CV-only selection has completed.
    Only the selected, registered, training-fitted pipeline is applied to the
    holdout.
    """
    settings = validate_config(config if config is not None else load_config())
    project_paths = get_project_paths(settings)
    reports_dir = Path(output_dir) if output_dir is not None else project_paths.root / "reports"
    artifacts_dir = (
        Path(model_dir) if model_dir is not None else project_paths.models_dir
    )
    cv_path = (
        Path(cv_results_path)
        if cv_results_path is not None
        else project_paths.reports_dir / "model_cv_results.json"
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

    candidates = pd.DataFrame(cv_payload["models"])
    selected_model = select_model_from_cv(
        candidates,
        model_names=settings["models"]["candidates"],
    )
    selection_row = candidates.loc[candidates["model"] == selected_model].iloc[0]
    selected_version = str(selection_row.get("model_version", settings["training"]["model_version"]))

    split: DataSplit = split_labeled_data(labeled_frame, config=settings)
    expected_schema = {
        column: str(labeled_frame[column].dtype)
        for column in split.X_train.columns
    }
    expected_target_mapping = {
        target: index
        for index, target in enumerate(
            sorted(settings["dataset"]["target_classes"])
        )
    }
    model, metadata = load_model_artifact(
        model_name=selected_model,
        model_version=selected_version,
        model_dir=artifacts_dir,
        expected_feature_schema=expected_schema,
        expected_target_mapping=expected_target_mapping,
        expected_dependency_versions=dependency_versions(),
    )
    if metadata.get("dataset_hash") != dataset_hash:
        raise ArtifactError("Selected model was trained on a different dataset hash")

    probability_classes = list(model.classes_)
    if POSITIVE_LABEL not in probability_classes:
        raise ArtifactError(
            f"Selected model class ordering does not contain {POSITIVE_LABEL!r}"
        )
    positive_probability_index = probability_classes.index(POSITIVE_LABEL)
    positive_probabilities = _positive_probability(model, split.X_test)
    y_pred = np.asarray(model.predict(split.X_test))
    metrics, matrix = _calculate_metrics(
        split.y_test,
        y_pred,
        positive_probabilities,
    )
    metrics["positive_probability_column_index"] = positive_probability_index
    report_dict = classification_report(
        split.y_test,
        y_pred,
        labels=[NEGATIVE_LABEL, POSITIVE_LABEL],
        output_dict=True,
        zero_division=0,
    )

    cv_comparison = candidates.copy()
    cv_comparison["selected_by_cv"] = cv_comparison["model"].eq(selected_model)
    cv_comparison["selection_metric"] = SELECTION_METRIC
    cv_comparison["selection_score"] = cv_comparison[SELECTION_METRIC]
    for metric in (
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "specificity",
        "precision",
        "recall",
        "f1",
    ):
        if f"{metric}_mean" not in cv_comparison:
            cv_comparison[f"{metric}_mean"] = np.nan
            cv_comparison[f"{metric}_std"] = np.nan
    for metric in (
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "specificity",
        "precision",
        "recall",
        "f1",
    ):
        cv_comparison[f"holdout_{metric}"] = np.where(
            cv_comparison["model"] == selected_model,
            metrics[metric] if metrics[metric] is not None else np.nan,
            np.nan,
        )

        reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = reports_dir / "model_comparison.csv"
    report_path = reports_dir / "evaluation_report.md"
    plots_dir = project_paths.root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    selection_path = reports_dir / "model_selection.json"
    cv_comparison.to_csv(comparison_path, index=False)
    figure_paths = _save_figures(
        cv_comparison,
        matrix,
        split.y_test,
        positive_probabilities,
        plots_dir,
    )
    report_path.write_text(
        _render_report(
            selected_model=selected_model,
            model_version=selected_version,
            selection_scores=selection_row.to_dict(),
            cv_comparison=cv_comparison,
            metrics=metrics,
            class_report=report_dict,
            probability_classes=probability_classes,
            positive_index=positive_probability_index,
            holdout_rows=len(split.X_test),
        ),
        encoding="utf-8",
    )
    selection_path.write_text(
        json.dumps(
            {
                "selected_model": selected_model,
                "selected_version": selected_version,
                "selection_metric": SELECTION_METRIC,
                "selection_tiebreaker": "roc_auc_mean",
                "selection_cv_scores": {
                    "balanced_accuracy_mean": float(
                        selection_row["balanced_accuracy_mean"]
                    ),
                    "roc_auc_mean": float(selection_row["roc_auc_mean"]),
                },
                "selection_dataset_hash": dataset_hash,
                "holdout_evaluated_after_selection": True,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return EvaluationResult(
        selected_model=selected_model,
        selected_version=selected_version,
        metrics=metrics,
        classification_report=report_dict,
        cv_comparison=cv_comparison,
        comparison_path=comparison_path,
        report_path=report_path,
        figure_paths=figure_paths,
        selection_path=selection_path,
    )
