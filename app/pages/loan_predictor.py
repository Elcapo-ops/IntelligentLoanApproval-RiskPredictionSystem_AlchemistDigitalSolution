"""Interactive applicant form backed by the registered full ML pipeline."""

import pandas as pd
import streamlit as st

from app.components.metrics import render_metrics
from app.components.theme import (
    render_historical_approval_disclosure,
    render_page_header,
)
from app.services.prediction_service import (
    build_historical_prediction_exports,
    explain_historical_approval_prediction,
    get_form_defaults,
    load_prediction_model,
    predict_historical_approval,
)
from config.settings import load_config
from utils.exceptions import LoanPredictionError


@st.cache_resource(show_spinner="Loading the registered model...")
def _cached_model():
    return load_prediction_model()


def _input_value(
    column: str,
    default,
    allowed,
    *,
    numeric: bool,
    positive_numeric: bool = False,
):
    """Render a form field with an explicit optional-missing choice."""
    with st.container():
        not_provided = st.checkbox(
            "Not provided",
            key=f"missing_{column}",
            help="The pipeline will handle this using its fitted training-fold imputer.",
        )
        if numeric:
            value = st.number_input(
                column.replace("_", " "),
                min_value=1.0 if positive_numeric else 0.0,
                value=float(default),
                step=100.0 if "Income" in column else 1.0,
                disabled=not_provided,
                key=f"value_{column}",
            )
        else:
            options = list(allowed)
            default_index = next(
                (
                    index
                    for index, value in enumerate(options)
                    if value == default
                ),
                0,
            )
            value = st.selectbox(
                column.replace("_", " "),
                options=options,
                index=default_index,
                format_func=lambda item: (
                    "Good credit history" if column == "Credit_History" and item == 1
                    else "No credit history" if column == "Credit_History" and item == 0
                    else str(item)
                ),
                disabled=not_provided,
                key=f"value_{column}",
            )
    return None if not_provided else value


def render() -> None:
    """Render the model's historical approval-label estimate form."""
    render_page_header(
        "Decision-support prototype",
        "Loan Eligibility Predictor",
        "Explore how the registered model classifies one entered application "
        "against historical approval labels.",
    )
    render_historical_approval_disclosure()
    config = load_config()
    try:
        defaults = get_form_defaults(config)
    except LoanPredictionError as exc:
        st.error(f"Prediction form could not load training-data examples: {exc}")
        return

    st.info(
        "Editable numeric examples use training-partition medians and "
        "categorical examples use training-partition modes. These are form "
        "examples, not recommended applicant values."
    )
    dataset_settings = config["dataset"]
    numeric_columns = dataset_settings["numeric_columns"]
    category_domains = dataset_settings["categorical_domains"]
    applicant: dict = {}

    with st.form("loan_prediction_form"):
        st.subheader("Applicant information")
        left, right = st.columns(2)
        with left:
            for column in numeric_columns[:2]:
                applicant[column] = _input_value(
                    column,
                    defaults[column],
                    [],
                    numeric=True,
                    positive_numeric=(
                        column in dataset_settings["positive_numeric_columns"]
                    ),
                )
            for column in list(category_domains)[:4]:
                applicant[column] = _input_value(
                    column,
                    defaults[column],
                    category_domains[column],
                    numeric=False,
                )
        with right:
            for column in numeric_columns[2:]:
                applicant[column] = _input_value(
                    column,
                    defaults[column],
                    [],
                    numeric=True,
                    positive_numeric=(
                        column in dataset_settings["positive_numeric_columns"]
                    ),
                )
            for column in list(category_domains)[4:]:
                applicant[column] = _input_value(
                    column,
                    defaults[column],
                    category_domains[column],
                    numeric=False,
                )
        submitted = st.form_submit_button(
            "Estimate Historical Approval Status",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return
    try:
        model, metadata = _cached_model()
        result = predict_historical_approval(model, applicant, config)
    except LoanPredictionError as exc:
        st.error(f"Prediction could not be completed: {exc}")
        return

    st.session_state.live_prediction_count = (
        int(st.session_state.get("live_prediction_count", 0)) + 1
    )
    outcome_key = (
        "live_prediction_approved"
        if result["predicted_loan_status"] == "Y"
        else "live_prediction_rejected"
    )
    st.session_state[outcome_key] = int(st.session_state.get(outcome_key, 0)) + 1

    st.divider()
    st.subheader("Estimated Historical Approval Status")
    render_historical_approval_disclosure()
    if result["predicted_loan_status"] == "Y":
        st.success("Predicted Historical Loan_Status: APPROVED")
    else:
        st.error("Predicted Historical Loan_Status: REJECTED")
    st.info(
        "A probability is not displayed because this model's probability "
        "calibration has not been validated."
    )
    render_metrics(
        [
            {
                "label": "Estimated Historical Approval Status",
                "value": (
                    "Approved"
                    if result["predicted_loan_status"] == "Y"
                    else "Rejected"
                ),
                "help": "The model classifies the historical Loan_Status target.",
            },
            {
                "label": "Model version",
                "value": f"v{result['model_version']}",
                "help": (
                    f"Registered model: {result['model_name']}. "
                    f"Artifact trained {metadata.get('training_date', 'date unavailable')}."
                ),
            },
        ]
    )
    exports = build_historical_prediction_exports(result, metadata)
    st.download_button(
        "Download prediction summary (JSON)",
        data=exports["json"],
        file_name="historical_loan_status_prediction.json",
        mime="application/json",
    )
    st.download_button(
        "Download prediction summary (CSV)",
        data=exports["csv"],
        file_name="historical_loan_status_prediction.csv",
        mime="text/csv",
    )
    try:
        explanation = explain_historical_approval_prediction(
            model,
            applicant,
            config,
        )
    except LoanPredictionError as exc:
        st.error(f"Prediction sensitivity checks could not be calculated: {exc}")
    else:
        st.subheader("One-feature sensitivity checks")
        st.caption(
            "Each row shows the model class after replacing only that input "
            "with its median or mode from the training partition."
        )
        explanation_frame = pd.DataFrame(explanation["features"]).rename(
            columns={
                "feature": "Input",
                "applicant_value": "Entered value",
                "training_reference": "Training reference",
                "predicted_loan_status_after_replacement": "Class after replacement",
                "changed": "Class changed",
            }
        )
        explanation_frame["Class after replacement"] = (
            explanation_frame["Class after replacement"]
            .map({"Y": "Approved", "N": "Rejected"})
        )
        changed_rows = explanation_frame[explanation_frame["Class changed"]]
        if changed_rows.empty:
            st.write(
                "No one-feature replacement with a training reference value "
                "changed the predicted class."
            )
        else:
            st.write(
                "The following single-input replacements changed the predicted class:"
            )
            for row in changed_rows.to_dict(orient="records"):
                st.markdown(
                    f"- **{row['Input']}**: `{row['Entered value']}` replaced by "
                    f"`{row['Training reference']}` changed the result to "
                    f"**{row['Class after replacement']}**."
                )
        with st.expander("Inspect all one-feature checks"):
            st.dataframe(
                explanation_frame,
                hide_index=True,
                use_container_width=True,
            )
        st.caption(explanation["warning"])
