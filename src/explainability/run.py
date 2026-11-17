"""Generate global, local, and subgroup explanations for the tuned pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.paths import get_project_paths
from config.settings import load_config, validate_config
from data.loader import load_csv
from data.splitting import get_model_feature_columns, split_labeled_data
from explainability.fairness import (
    calculate_group_metrics,
    save_fairness_chart,
    summarize_group_gaps,
)
from explainability.global_explanation import (
    GlobalExplanationResult,
    compute_shap_values,
    cross_validated_permutation_importance,
    save_global_importance_chart,
)
from explainability.local_explanation import (
    explain_local_prediction,
    save_local_explanation_chart,
)
from models.registry import dataframe_sha256, load_model_artifact
from prediction.scope import HISTORICAL_LOAN_STATUS_DISCLOSURE
from utils.exceptions import DataValidationError


def _display_rate(value: float) -> str:
    return f"{value:.4f}" if np.isfinite(value) else "not estimable"


@dataclass(frozen=True)
class ExplainabilityResult:
    """Paths and measured outputs from Phase 6.1 explanation generation."""

    global_importance: pd.DataFrame
    fairness_metrics: pd.DataFrame
    fairness_gaps: pd.DataFrame
    local_explanations: dict[str, dict[str, Any]]
    shap_available: bool
    report_path: Path
    figure_paths: dict[str, Path]


def _render_report(
    *,
    model_name: str,
    model_version: str,
    training_rows: int,
    holdout_rows: int,
    global_result: GlobalExplanationResult,
    fairness_metrics: pd.DataFrame,
    fairness_gaps: pd.DataFrame,
    local_explanations: dict[str, dict[str, Any]],
    shap_available: bool,
    figure_paths: dict[str, Path],
) -> str:
    top_features = global_result.importance.head(10)
    reviewed_attributes = ", ".join(
        f"`{attribute}`"
        for attribute in fairness_metrics["attribute"].drop_duplicates()
    )
    lines = [
        "# Phase 6.1 — Explainable AI and exploratory fairness review",
        "",
        HISTORICAL_LOAN_STATUS_DISCLOSURE,
        "",
        "## Scope and method",
        "",
        "The model estimates the dataset's historical `Loan_Status` class. "
        "Feature importance indicates predictive contribution within this "
        "dataset, not causation.",
        "",
        f"- Model: `{model_name}` version `{model_version}`.",
        f"- Explanations use only the {training_rows}-row training partition. "
        f"The {holdout_rows}-row holdout was not passed to these analyses.",
        "- Global importance is permutation importance over original input "
        "columns, measured on each stratified CV validation fold. The complete "
        "pipeline regenerates engineered and one-hot encoded features after "
        "each raw input column is permuted.",
        "- Local cases use out-of-fold predictions. For each, the model fitted "
        "without that case's fold is used for one-feature-at-a-time sensitivity "
        "replacements against that fold's training-data medians/modes.",
        "- Fairness metrics use the same out-of-fold class predictions and "
        "report observed approval rates alongside model selection and error rates.",
        "- The optimized model's prior training/holdout split is respected here. "
        "The holdout had been evaluated for the baseline model during Phase 5.1; "
        "that historical exposure is not reversed by this analysis.",
        "",
        "## Global permutation importance",
        "",
        "| Original feature | Mean balanced-accuracy decrease | SD across folds |",
        "|---|---:|---:|",
    ]
    for row in top_features.itertuples(index=False):
        fold_sd = (
            f"{row.importance_fold_std:.4f}"
            if np.isfinite(row.importance_fold_std)
            else "not available"
        )
        lines.append(
            f"| `{row.feature}` | {row.importance_mean:.4f} | {fold_sd} |"
        )
    lines.extend(
        [
            "",
            "Negative permutation importance can occur when shuffling improves "
            "a fold score; correlated inputs can share or mask importance. "
            "Derived-feature effects are reflected through their original "
            "input columns rather than ranked as separate, independently "
            "permutable fields.",
            "",
                        "![Cross-validated permutation importance](plots/"
            f"{figure_paths['global'].name})",
            "",
            "## Local prediction sensitivity",
            "",
            "Each example changes one raw input at a time to the training-fold "
            "reference median (numeric) or mode (categorical) and measures the "
            "resulting change in its raw, uncalibrated score for the historical "
            "approval class. It is not an estimated outcome probability or "
            "additive attribution. Independent replacements can create implausible "
            "combinations, and these changes are not causal explanations.",
            "",
        ]
    )
    for case_name, explanation in local_explanations.items():
        lines.extend(
            [
                f"### {case_name.replace('_', ' ').title()}",
                "",
                f"- Predicted historical `Loan_Status`: "
                f"`{explanation['predicted_loan_status']}`.",
                f"- Raw, uncalibrated historical approval-class score: "
                f"{explanation['uncalibrated_approval_class_score']:.4f}.",
                "- Largest one-at-a-time score sensitivities:",
                "",
                "| Feature | Applicant value | Reference value | Raw score change |",
                "|---|---|---|---:|",
            ]
        )
        for feature in explanation["features"][:5]:
            lines.append(
                f"| `{feature['feature']}` | `{feature['applicant_value']}` | "
                f"`{feature['reference_value']}` | "
                f"{feature['uncalibrated_approval_score_delta']:+.4f} |"
            )
        lines.append("")

    lines.extend(
        [
                        "![Local prediction sensitivities](plots/"
            f"{figure_paths['local'].name})",
            "",
            "## Exploratory subgroup review",
            "",
            f"The reviewed attributes are {reviewed_attributes}, as available "
            "in the source data. These are limited observed "
            "attributes, not a complete demographic or legal fairness audit.",
            "",
            "| Attribute | Eligible groups | Predicted historical approval-label "
            "rate range | TPR range | FPR range | Min/max selection-rate ratio |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fairness_gaps.itertuples(index=False):
        lines.append(
            f"| `{row.attribute}` | {row.eligible_groups} | "
            f"{_display_rate(row.approval_rate_gap)} | "
            f"{_display_rate(row.tpr_gap)} | "
            f"{_display_rate(row.fpr_gap)} | "
            f"{_display_rate(row.selection_rate_ratio)} |"
        )
    lines.extend(
        [
            "",
            "Group sizes and the underlying selection, observed approval, TPR, "
            "and FPR rates are in `artifacts/reports/fairness_group_metrics.csv`. "
            "Groups below the configured minimum are retained in the data but "
            "excluded from range comparisons. Missing subgroup membership is "
            "shown explicitly.",
            "",
            "| Attribute | Group | n | Predicted historical approval-label rate | "
            "Observed historical approval rate | "
            "TPR | FPR |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fairness_metrics.itertuples(index=False):
        lines.append(
            f"| `{row.attribute}` | `{row.group}` | {row.n} | "
            f"{_display_rate(row.approval_rate)} | "
            f"{_display_rate(row.observed_approval_rate)} | "
            f"{_display_rate(row.true_positive_rate)} | "
            f"{_display_rate(row.false_positive_rate)} |"
        )
    lines.extend(
        [
            "",
                        "![Exploratory subgroup rates](plots/"
            f"{figure_paths['fairness'].name})",
            "",
            "Feature importance indicates predictive contribution within this "
            "dataset, not causation. Differences are descriptive and may reflect sampling variation, "
            "historical decisions, label quality, or other omitted factors. "
            "They do not establish discrimination or its absence, satisfy a "
            "legal standard, or support causal claims. The dataset has only "
            f"{training_rows} training rows and lacks important demographic "
            "and socioeconomic context, so small-group rates are especially "
            "uncertain.",
            "",
            "## SHAP",
            "",
        ]
    )
    if shap_available:
        lines.append(
            "SHAP was available and was calculated for the local examples using "
            "the complete raw-input pipeline. Values are included in "
            "`artifacts/reports/local_explanations.json`; they describe this "
            "model's output and are not causal."
        )
    else:
        lines.append(
            "SHAP was not run because the optional `shap` package is not "
            "installed in the current environment. No SHAP values were "
            "fabricated. The implemented SHAP path accepts original named "
            "features through the complete pipeline when the package is "
            "installed and compatible."
        )
    lines.extend(
        [
            "",
            "## Decision-support limitation",
            "",
            "These explanations describe model behavior, not applicant "
            "worthiness. No explanation, feature-importance ranking, or "
            "subgroup metric should be presented as the sole reason for a "
            "lending decision. Human review, appropriate validation, and "
            "domain/legal assessment remain necessary.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_explanations(
    *,
    config: dict[str, Any] | None = None,
) -> ExplainabilityResult:
    """Generate CV-based explanations and subgroup metrics for the tuned model."""
    settings = validate_config(config if config is not None else load_config())
    paths = get_project_paths(settings)
    paths.ensure_directories()
    explain_settings = settings["explainability"]

    labeled_frame = load_csv(paths.labeled_dataset)
    split = split_labeled_data(labeled_frame, config=settings)
    expected_schema = {
        column: str(labeled_frame[column].dtype)
        for column in get_model_feature_columns(settings)
    }
    model, metadata = load_model_artifact(
        model_name=explain_settings["model_name"],
        model_version=explain_settings["model_version"],
        model_dir=paths.models_dir,
        expected_feature_schema=expected_schema,
        expected_target_mapping={"N": 0, "Y": 1},
    )
    if metadata.get("dataset_hash") != dataframe_sha256(labeled_frame):
        raise DataValidationError(
            "Explanation dataset does not match the model training dataset"
        )
    if metadata.get("training_configuration", {}).get("holdout_used") is not False:
        raise DataValidationError(
            "Explanations require an artifact trained without holdout data"
        )
    if not hasattr(model, "predict_proba"):
        raise DataValidationError(
            "Registered artifact does not expose the required class-score interface"
        )
    missing_fairness_attributes = sorted(
        set(explain_settings["fairness_attributes"]).difference(split.X_train.columns)
    )
    if missing_fairness_attributes:
        raise DataValidationError(
            "Configured fairness attributes are missing from model inputs: "
            f"{missing_fairness_attributes}"
        )

    global_result = cross_validated_permutation_importance(
        model,
        split.X_train,
        split.y_train,
        n_splits=settings["training"]["cv_folds"],
        random_state=settings["project"]["random_seed"],
        n_repeats=explain_settings["permutation_repeats"],
    )
    attributes = explain_settings["fairness_attributes"]
    fairness_metrics = calculate_group_metrics(
        split.X_train,
        split.y_train,
        global_result.oof_predictions,
        attributes=attributes,
        minimum_group_size=explain_settings["fairness_min_group_size"],
    )
    fairness_gaps = summarize_group_gaps(fairness_metrics)

    sorted_approval_scores = global_result.oof_approval_score.sort_values()
    local_case_count = min(
        explain_settings["local_cases"],
        len(sorted_approval_scores),
    )
    quantile_positions = (
        [len(sorted_approval_scores) // 2]
        if local_case_count == 1
        else np.rint(
            np.linspace(0, len(sorted_approval_scores) - 1, local_case_count)
        )
        .astype(int)
        .tolist()
    )
    selected_cases: dict[str, Any] = {}
    for case_number, position in enumerate(quantile_positions, start=1):
        if local_case_count == 1:
            case_name = "median_out_of_fold_historical_approval_score"
        elif case_number == 1:
            case_name = "lower_out_of_fold_historical_approval_score"
        elif case_number == local_case_count:
            case_name = "higher_out_of_fold_historical_approval_score"
        else:
            case_name = f"intermediate_out_of_fold_historical_case_{case_number}"
        selected_cases[case_name] = sorted_approval_scores.index[position]
    local_explanations: dict[str, dict[str, Any]] = {}
    for case_name, row_index in selected_cases.items():
        fold_number = int(global_result.oof_fold.loc[row_index])
        fold_position = fold_number - 1
        fold_model = global_result.fold_models[fold_position]
        training_indices = global_result.fold_train_indices[fold_position]
        reference = split.X_train.iloc[training_indices]
        applicant = split.X_train.loc[[row_index]]
        explanation = explain_local_prediction(fold_model, applicant, reference)
        explanation["out_of_fold_fold"] = fold_number
        explanation["case_id"] = case_name

        shap_result = compute_shap_values(
            fold_model,
            reference,
            applicant,
            background_rows=explain_settings["shap_background_rows"],
        )
        if shap_result is not None:
            baseline, shap_values = shap_result
            explanation["shap"] = {
                "uncalibrated_approval_class_score": baseline,
                "values": {
                    feature: float(value)
                    for feature, value in shap_values.iloc[0].items()
                },
            }
        local_explanations[case_name] = explanation

    shap_available = all("shap" in item for item in local_explanations.values())
    paths.global_importance.parent.mkdir(parents=True, exist_ok=True)
    paths.fairness_results.parent.mkdir(parents=True, exist_ok=True)
    paths.local_explanations.parent.mkdir(parents=True, exist_ok=True)

    global_result.importance.to_csv(paths.global_importance, index=False)
    fairness_metrics.to_csv(paths.fairness_results, index=False)
    paths.local_explanations.write_text(
        json.dumps(
            {
                "prediction_type": "historical_approval_prediction",
                "disclosure": HISTORICAL_LOAN_STATUS_DISCLOSURE,
                "predictions": local_explanations,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    plots_dir = paths.root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "global": plots_dir / "global_permutation_importance.png",
        "fairness": plots_dir / "fairness_group_rates.png",
        "local": plots_dir / "local_prediction_sensitivities.png",
    }
    save_global_importance_chart(
        global_result.importance,
        figure_paths["global"],
    )
    save_fairness_chart(fairness_metrics, figure_paths["fairness"])
    save_local_explanation_chart(local_explanations, figure_paths["local"])
    paths.explainability_report.write_text(
        _render_report(
            model_name=explain_settings["model_name"],
            model_version=explain_settings["model_version"],
            training_rows=len(split.X_train),
            holdout_rows=len(split.X_test),
            global_result=global_result,
            fairness_metrics=fairness_metrics,
            fairness_gaps=fairness_gaps,
            local_explanations=local_explanations,
            shap_available=shap_available,
            figure_paths=figure_paths,
        ),
        encoding="utf-8",
    )
    return ExplainabilityResult(
        global_importance=global_result.importance,
        fairness_metrics=fairness_metrics,
        fairness_gaps=fairness_gaps,
        local_explanations=local_explanations,
        shap_available=shap_available,
        report_path=paths.explainability_report,
        figure_paths=figure_paths,
    )


if __name__ == "__main__":
    result = generate_explanations()
    print(f"Explainability report saved to {result.report_path}")
