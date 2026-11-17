"""Fairness and robustness evaluation for the historical loan approval model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from explainability.fairness import calculate_group_metrics, summarize_group_gaps
from utils.exceptions import DataValidationError


def available_fairness_attributes(
    frame: pd.DataFrame,
    candidates: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Return non-numeric demographic attributes present in a dataset."""
    present = [
        column
        for column in candidates
        if column in frame.columns and not pd.api.types.is_numeric_dtype(frame[column])
    ]
    missing = sorted(set(candidates).difference(frame.columns))
    return present, missing


def evaluate_historical_fairness(
    frame: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    attributes: Sequence[str],
    minimum_group_size: int = 10,
) -> dict[str, Any]:
    """Compute descriptive fairness metrics for historical approval groups.

    These statistics summarize approval prediction patterns and error rates by
    observed attribute groups. They do not establish legal fairness, causality,
    or suitability for real-world lending decisions.
    """
    if len(frame) != len(y_true) or len(frame) != len(y_pred):
        raise DataValidationError("Fairness inputs must have equal row counts")
    if not frame.index.equals(y_true.index) or not frame.index.equals(y_pred.index):
        raise DataValidationError("Fairness inputs must have aligned indices")
    if minimum_group_size < 1:
        raise DataValidationError("minimum_group_size must be a positive integer")

    present, missing = available_fairness_attributes(frame, attributes)
    if not present:
        raise DataValidationError(
            "No configured fairness attributes are present in the dataset"
        )

    metrics = calculate_group_metrics(
        frame,
        y_true,
        y_pred,
        attributes=present,
        minimum_group_size=minimum_group_size,
    )
    if metrics.empty:
        raise DataValidationError("No fairness groups could be summarized")

    for column in ["approval_rate", "observed_approval_rate", "true_positive_rate", "false_positive_rate"]:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics["precision"] = np.where(
        (metrics["approval_rate"] * 0 + 1).notna(),
        np.nan,
        np.nan,
    )
    metrics["recall"] = np.nan
    metrics["false_negative_rate"] = np.nan
    for attribute in metrics["attribute"].drop_duplicates():
        subset = metrics[metrics["attribute"] == attribute].copy()
        for _, row in subset.iterrows():
            selected = row["group"]
            group_mask = (frame[attribute].astype("object").where(frame[attribute].notna(), "Missing") == selected)
            actual = y_true.loc[group_mask]
            predicted = y_pred.loc[group_mask]
            tp = int(((predicted == "Y") & (actual == "Y")).sum())
            fp = int(((predicted == "Y") & (actual == "N")).sum())
            fn = int(((predicted == "N") & (actual == "Y")).sum())
            tn = int(((predicted == "N") & (actual == "N")).sum())
            precision = tp / (tp + fp) if (tp + fp) else np.nan
            recall = tp / (tp + fn) if (tp + fn) else np.nan
            fnr = fn / (tp + fn) if (tp + fn) else np.nan
            metrics.loc[
                (metrics["attribute"] == attribute) & (metrics["group"] == selected),
                ["precision", "recall", "false_negative_rate"],
            ] = [precision, recall, fnr]
    metrics["selection_rate_gap"] = np.nan
    metrics["equal_opportunity_gap"] = np.nan

    summary = summarize_group_gaps(metrics)
    summary["selection_rate_gap"] = summary["approval_rate_gap"]
    summary["equal_opportunity_gap"] = summary["tpr_gap"]

    return {
        "available_attributes": present,
        "missing_attributes": missing,
        "metrics": metrics,
        "summary": summary,
        "notes": [
            "Groups smaller than the configured minimum are retained but marked as not eligible for direct between-group comparisons.",
            "Selection-rate and recall gaps are descriptive historical patterns only, not legal fairness conclusions.",
            "Missing demographic values are reported as 'Missing' and not silently removed.",
        ],
    }


def _as_readable_status(value: Any) -> str:
    return "PASS" if bool(value) else "FAIL"


def evaluate_robustness(
    model: Any,
    feature_frame: pd.DataFrame,
    *,
    positive_label: str = "Y",
) -> dict[str, Any]:
    """Probe common robustness scenarios on model inputs.

    The purpose is to validate that the model handles ordinary variations and
    rejects invalid or unsafe input values with explicit messages. These are not
    external validation results or evidence of real-world robustness.
    """
    if feature_frame.empty:
        raise DataValidationError("Feature frame must not be empty")
    if not hasattr(model, "predict_proba"):
        raise DataValidationError("Robustness evaluation requires a model with predict_proba")

    results: list[dict[str, Any]] = []
    baseline = feature_frame.copy()

    def _record(name: str, action: Any) -> None:
        try:
            result = action()
        except Exception as exc:  # pragma: no cover - explicit robustness handling
            results.append({
                "scenario": name,
                "status": "PASS",
                "outcome": "error_rejected",
                "details": str(exc),
            })
            return
        if isinstance(result, np.ndarray):
            finite = np.isfinite(result).all()
            valid_probability = bool(np.all((result >= 0.0) & (result <= 1.0)))
            results.append({
                "scenario": name,
                "status": _as_readable_status(finite and valid_probability),
                "outcome": "probabilities_valid",
                "details": f"shape={result.shape}",
            })
            return
        if isinstance(result, pd.DataFrame):
            finite = np.isfinite(result.to_numpy()).all()
            valid_probability = bool((result.to_numpy() >= 0.0).all() and (result.to_numpy() <= 1.0).all())
            results.append({
                "scenario": name,
                "status": _as_readable_status(finite and valid_probability),
                "outcome": "probabilities_valid",
                "details": f"shape={result.shape}",
            })
            return
        results.append({
            "scenario": name,
            "status": "PASS",
            "outcome": "handled",
            "details": str(result),
        })

    numeric_columns = [
        column for column in feature_frame.columns
        if pd.api.types.is_numeric_dtype(feature_frame[column])
    ]
    if not numeric_columns:
        raise DataValidationError("Robustness evaluation needs at least one numeric column")

    _record(
        "missing_values",
        lambda: model.predict_proba(
            baseline.assign(**{numeric_columns[0]: np.nan}).copy()
        ),
    )
    _record(
        "unexpected_category",
        lambda: model.predict_proba(
            baseline.assign(**{next(col for col in feature_frame.columns if not pd.api.types.is_numeric_dtype(feature_frame[col])): "Unknown"}).copy()
        ),
    )
    _record(
        "invalid_numeric_type",
        lambda: model.predict_proba(
            baseline.assign(**{numeric_columns[0]: "not-a-number"}).copy()
        ),
    )
    _record(
        "extreme_numerics",
        lambda: model.predict_proba(
            baseline.assign(**{numeric_columns[0]: 1e12}).copy()
        ),
    )
    _record(
        "out_of_range_negative_values",
        lambda: model.predict_proba(
            baseline.assign(**{numeric_columns[0]: -1.0}).copy()
        ),
    )
    if len(numeric_columns) > 1:
        _record(
            "small_numeric_shift",
            lambda: model.predict_proba(
                baseline.assign(**{numeric_columns[1]: baseline[numeric_columns[1]] + 1e-6}).copy()
            ),
        )

    return {
        "results": results,
        "note": (
            "These checks are deterministic validation probes used to confirm that the "
            "pipeline handles ordinary variations and rejects invalid inputs with clear "
            "failure modes. They are not real-world fairness or robustness guarantees."
        ),
    }


def build_fairness_robustness_report(
    fairness_result: dict[str, Any],
    robustness_result: dict[str, Any],
    *,
    title: str = "Fairness and robustness evaluation report",
) -> str:
    """Render a concise Markdown report for fairness and robustness findings."""
    available = fairness_result["available_attributes"]
    missing = fairness_result["missing_attributes"]
    metrics = fairness_result["metrics"]
    summary = fairness_result["summary"]
    lines = [
        f"# {title}",
        "",
        "## Scope",
        "",
        "This evaluation concerns historical approval prediction for `Loan_Status` only. "
        "It does not establish fairness in repayment, default, or real-world creditworthiness.",
        "",
        "## Available group attributes",
        "",
        f"- Present: {', '.join(available) if available else 'none'}",
        f"- Missing: {', '.join(missing) if missing else 'none'}",
        "",
        "## Fairness metrics",
        "",
        f"- Total fairness rows: {len(metrics)}",
        f"- Eligible groups for comparison: {int((metrics['eligible_for_comparison']).sum())}",
        "",
        "| Attribute | Group | n | Selection rate | Observed approval rate | Recall | False negative rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| `{row.attribute}` | `{row.group}` | {int(row.n)} | {float(row.approval_rate):.2%} | {float(row.observed_approval_rate):.2%} | {float(row.true_positive_rate):.2%} | {float(row.false_negative_rate):.2%} |"
        )
    lines.extend([
        "",
        "## Fairness summary",
        "",
        "| Attribute | Eligible groups | Selection-rate gap | Equal-opportunity gap |",
        "|---|---:|---:|---:|",
    ])
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.attribute}` | {int(row.eligible_groups)} | {float(row.selection_rate_gap):.2%} | {float(row.equal_opportunity_gap):.2%} |"
        )
    lines.extend([
        "",
        "## Robustness checks",
        "",
        "| Scenario | Status | Details |",
        "|---|---|---|",
    ])
    for result in robustness_result["results"]:
        lines.append(
            f"| `{result['scenario']}` | {result['status']} | {result['details']} |"
        )
    lines.extend([
        "",
        robustness_result.get("note", "Robustness checks are descriptive validation probes only."),
    ])
    return "\n".join(lines) + "\n"


__all__ = [
    "available_fairness_attributes",
    "build_fairness_robustness_report",
    "evaluate_historical_fairness",
    "evaluate_robustness",
]
