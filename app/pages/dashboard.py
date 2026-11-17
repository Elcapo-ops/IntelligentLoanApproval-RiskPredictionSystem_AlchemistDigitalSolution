"""Historical portfolio snapshot and in-session live prediction counts."""

from pathlib import Path

import streamlit as st

from app.components.charts import render_group_approval_chart, render_status_chart
from app.components.metrics import render_metrics
from app.components.theme import (
    render_historical_approval_disclosure,
    render_page_header,
)
from app.services.dashboard_service import (
    load_historical_dashboard_data,
    load_model_performance_data,
)
from utils.exceptions import LoanPredictionError


def render() -> None:
    """Render the dashboard using actual labeled-dataset and CV outputs."""
    render_page_header(
        "Portfolio overview",
        "Loan Approval Dashboard",
        "A view of observed historical labels and independently reported "
        "training-CV performance.",
    )
    render_historical_approval_disclosure()
    try:
        historical = load_historical_dashboard_data()
        performance = load_model_performance_data()
    except LoanPredictionError as exc:
        st.error(f"Dashboard data could not be loaded: {exc}")
        return

    st.markdown("### Historical labeled applications")
    st.caption(
        f"Source file: `{Path(historical['source']).name}`. These are dataset records, not "
        "current session predictions or current portfolio accounts."
    )
    render_metrics(
        [
            {
                "label": "Applications",
                "value": f"{historical['application_count']:,}",
                "help": "Rows in the labeled dataset.",
            },
            {
                "label": "Approval rate",
                "value": f"{historical['approval_rate']:.1%}",
                "help": (
                    f"{historical['approval_count']:,} records labeled "
                    "Loan_Status = Y."
                ),
            },
            {
                "label": "Rejection rate",
                "value": f"{historical['rejection_rate']:.1%}",
                "help": (
                    f"{historical['rejection_count']:,} records labeled "
                    "Loan_Status = N."
                ),
            },
        ]
    )

    counts_column, area_column = st.columns(2)
    with counts_column:
        st.subheader("Approval and rejection")
        render_status_chart(historical["status_counts"])
    with area_column:
        st.subheader("Historical labels by property area")
        render_group_approval_chart(historical["approval_by_area"])

    st.markdown("### Model performance")
    st.caption(
        f"Training-only, {performance['cv_folds']}-fold stratified CV for "
        f"{performance['model_name']} v{performance['model_version']}. "
        "These are not live-prediction outcomes."
    )
    tuning = performance["tuning"]
    render_metrics(
        [
            {
                "label": "CV balanced accuracy",
                "value": f"{tuning['tuned_balanced_accuracy_mean']:.1%}",
                "help": "Mean across stratified training CV folds.",
            },
            {
                "label": "CV ROC-AUC",
                "value": f"{tuning['tuned_roc_auc_mean']:.3f}",
                "help": "Mean ROC-AUC across training CV folds.",
            },
            {
                "label": "CV change vs baseline",
                "value": f"{tuning['balanced_accuracy_delta']:+.3f}",
                "help": "Tuned minus untuned mean balanced accuracy.",
            },
        ]
    )

    st.divider()
    st.subheader("Live predictions in this browser session")
    live_count = int(st.session_state.get("live_prediction_count", 0))
    live_approved = int(st.session_state.get("live_prediction_approved", 0))
    live_rejected = int(st.session_state.get("live_prediction_rejected", 0))
    render_metrics(
        [
            {"label": "Predictions run", "value": f"{live_count:,}"},
            {
                "label": "Predicted historical approvals",
                "value": f"{live_approved:,}",
            },
            {
                "label": "Predicted historical rejections",
                "value": f"{live_rejected:,}",
            },
        ]
    )
    st.caption(
        "Session counts are in-memory UI activity only. They reset when the "
        "Streamlit session restarts and are never added to historical dataset "
        "counts or model evaluation metrics."
    )
