"""Reusable charts for the Streamlit pages."""

import pandas as pd
import streamlit as st


def render_status_chart(counts: pd.Series, *, height: int = 300) -> None:
    """Show actual historical application counts for approved and rejected."""
    chart_data = (
        counts.rename_axis("Historical label")
        .rename("Applications")
        .to_frame()
        .reset_index()
    )
    st.bar_chart(
        chart_data,
        x="Historical label",
        y="Applications",
        color=["#2874b2"],
        height=height,
    )


def render_group_approval_chart(approval_by_area: pd.DataFrame) -> None:
    """Show observed historical approvals and rejections by property area."""
    st.bar_chart(
        approval_by_area,
        x="Property_Area",
        y=["Y", "N"],
        color=["#23845b", "#b84d55"],
        stack=False,
        height=300,
    )


def render_metric_comparison_chart(
    results: pd.DataFrame,
    metrics: list[str],
    *,
    height: int = 360,
) -> None:
    """Compare measured cross-validation means for each saved candidate."""
    chart_data = results.set_index("model")[metrics].rename(
        columns={
            "accuracy_mean": "Accuracy",
            "precision_mean": "Precision (Y)",
            "recall_mean": "Recall (Y)",
            "f1_mean": "F1 (Y)",
            "roc_auc_mean": "ROC-AUC",
        }
    )
    st.bar_chart(chart_data, y=list(chart_data.columns), height=height)


def render_historical_approval_pattern(
    pattern: pd.DataFrame,
    group_column: str,
) -> None:
    """Chart observed historical approval rates and group support."""
    chart_data = pattern.rename(
        columns={
            group_column: "Group",
            "approval_rate": "Historical approval rate",
        }
    )
    st.bar_chart(
        chart_data,
        x="Group",
        y="Historical approval rate",
        height=280,
    )
