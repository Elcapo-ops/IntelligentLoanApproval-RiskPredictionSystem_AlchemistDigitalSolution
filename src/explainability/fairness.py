"""Exploratory subgroup performance and selection-rate measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.exceptions import DataValidationError


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def calculate_group_metrics(
    frame: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    attributes: Sequence[str],
    minimum_group_size: int = 10,
) -> pd.DataFrame:
    """Calculate observed outcomes and OOF model rates for configured groups.

    Rates are descriptive and do not determine legal compliance or fairness.
    Small groups are reported but marked ineligible for between-group
    comparisons due to high sampling uncertainty.
    """
    if len(frame) != len(y_true) or len(frame) != len(y_pred):
        raise DataValidationError("Fairness inputs must have equal row counts")
    if not frame.index.equals(y_true.index) or not frame.index.equals(y_pred.index):
        raise DataValidationError("Fairness inputs must have aligned indices")
    if minimum_group_size < 1:
        raise DataValidationError("Minimum group size must be positive")
    missing = sorted(set(attributes).difference(frame.columns))
    if missing:
        raise DataValidationError(f"Fairness attributes are missing: {missing}")

    rows: list[dict[str, object]] = []
    for attribute in attributes:
        groups = frame[attribute].astype("object").where(
            frame[attribute].notna(),
            "Missing",
        )
        for group_value in groups.drop_duplicates():
            mask = groups.eq(group_value)
            actual = y_true.loc[mask]
            predicted = y_pred.loc[mask]
            positive_actual = actual.eq("Y")
            negative_actual = actual.eq("N")
            predicted_positive = predicted.eq("Y")
            n = int(mask.sum())
            rows.append(
                {
                    "attribute": attribute,
                    "group": str(group_value),
                    "n": n,
                    "eligible_for_comparison": n >= minimum_group_size,
                    "approval_rate": _rate(
                        int(predicted_positive.sum()),
                        n,
                    ),
                    "observed_approval_rate": _rate(
                        int(positive_actual.sum()),
                        n,
                    ),
                    "true_positive_rate": _rate(
                        int((predicted_positive & positive_actual).sum()),
                        int(positive_actual.sum()),
                    ),
                    "false_positive_rate": _rate(
                        int((predicted_positive & negative_actual).sum()),
                        int(negative_actual.sum()),
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_group_gaps(group_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize eligible-group ranges and selection-rate ratios per attribute."""
    summaries: list[dict[str, object]] = []
    for attribute, attribute_metrics in group_metrics.groupby("attribute", sort=False):
        eligible = attribute_metrics[
            attribute_metrics["eligible_for_comparison"]
        ]
        if len(eligible) < 2:
            summaries.append(
                {
                    "attribute": attribute,
                    "eligible_groups": int(len(eligible)),
                    "approval_rate_gap": np.nan,
                    "tpr_gap": np.nan,
                    "fpr_gap": np.nan,
                    "selection_rate_ratio": np.nan,
                }
            )
            continue
        approval_rates = eligible["approval_rate"].dropna()
        minimum_rate = (
            float(approval_rates.min()) if not approval_rates.empty else np.nan
        )
        maximum_rate = (
            float(approval_rates.max()) if not approval_rates.empty else np.nan
        )
        summaries.append(
            {
                "attribute": attribute,
                "eligible_groups": int(len(eligible)),
                "approval_rate_gap": (
                    maximum_rate - minimum_rate
                    if np.isfinite(minimum_rate) and np.isfinite(maximum_rate)
                    else np.nan
                ),
                "tpr_gap": eligible["true_positive_rate"].max()
                - eligible["true_positive_rate"].min()
                if eligible["true_positive_rate"].notna().all()
                else np.nan,
                "fpr_gap": eligible["false_positive_rate"].max()
                - eligible["false_positive_rate"].min()
                if eligible["false_positive_rate"].notna().all()
                else np.nan,
                "selection_rate_ratio": (
                    minimum_rate / maximum_rate
                    if np.isfinite(minimum_rate)
                    and np.isfinite(maximum_rate)
                    and maximum_rate > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(summaries)


def save_fairness_chart(group_metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot subgroup selection and error rates for eligible groups."""
    eligible = group_metrics[group_metrics["eligible_for_comparison"]].copy()
    if eligible.empty:
        raise DataValidationError("No groups meet the minimum size for plotting")
    eligible["label"] = eligible["attribute"] + " = " + eligible["group"]
    figure, axes = plt.subplots(1, 3, figsize=(16, max(5, 0.38 * len(eligible))))
    measures = (
        ("approval_rate", "Predicted historical approval-label rate"),
        ("true_positive_rate", "Recall for historical approvals"),
        ("false_positive_rate", "False positive rate for historical rejections"),
    )
    for axis, (column, title) in zip(axes, measures):
        plotted = eligible.sort_values(column, ascending=True)
        axis.barh(plotted["label"], plotted[column], color="#5a82a8")
        axis.set_xlim(0, 1)
        axis.set_title(title)
        axis.set_xlabel("Rate")
        axis.set_ylabel("")
    figure.suptitle("Exploratory subgroup rates from out-of-fold predictions")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
