"""Separate Streamlit page for LendingClub historical default-risk estimates."""

from pathlib import Path

import pandas as pd
import streamlit as st

from default_risk.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from default_risk.prediction import (
    DEFAULT_RISK_DISCLAIMER,
    load_default_risk_model,
    local_probability_sensitivity,
    predict_default_risk,
)
from utils.exceptions import LoanPredictionError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports" / "default_risk"

_CATEGORY_OPTIONS = {
    "term": ["36 months", "60 months"],
    "emp_length": [
        "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
        "6 years", "7 years", "8 years", "9 years", "10+ years",
    ],
    "home_ownership": ["MORTGAGE", "RENT", "OWN", "OTHER", "ANY", "NONE"],
    "verification_status": ["Not Verified", "Source Verified", "Verified"],
    "purpose": [
        "debt_consolidation", "credit_card", "home_improvement", "other",
        "major_purchase", "small_business", "car", "medical", "moving",
        "vacation", "house", "wedding", "renewable_energy", "educational",
    ],
    "addr_state": [
        "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
        "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME",
        "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM",
        "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
        "UT", "VA", "VT", "WA", "WI", "WV", "WY",
    ],
    "application_type": ["Individual", "Joint App"],
}


@st.cache_resource(show_spinner="Loading the registered default-risk model...")
def _cached_model():
    return load_default_risk_model(PROJECT_ROOT)


def _field_default(feature: str, defaults: dict) -> float | str:
    value = defaults.get(feature)
    if value is None:
        return 0.0 if feature in NUMERIC_FEATURES else ""
    if feature in NUMERIC_FEATURES:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    return str(value)


def _render_numeric_field(feature: str, defaults: dict) -> float | None:
    default = _field_default(feature, defaults)
    min_value = 300.0 if feature in {"fico_range_low", "fico_range_high"} else 0.0
    max_value = 850.0 if feature in {"fico_range_low", "fico_range_high"} else None
    if feature == "revol_util":
        max_value = 200.0
    if feature == "dti":
        max_value = 200.0
    missing = st.checkbox(
        "Not provided",
        key=f"default_risk_missing_{feature}",
        help="The fitted training pipeline will impute this value.",
    )
    kwargs = {
        "label": feature.replace("_", " ").title(),
        "min_value": min_value,
        "value": max(min_value, float(default)),
        "step": 100.0 if feature in {"loan_amnt", "annual_inc", "revol_bal"} else 1.0,
        "disabled": missing,
        "key": f"default_risk_value_{feature}",
    }
    if max_value is not None:
        kwargs["max_value"] = max_value
    value = st.number_input(**kwargs)
    return None if missing else float(value)


def _render_categorical_field(feature: str, defaults: dict) -> str | None:
    options = _CATEGORY_OPTIONS[feature]
    reference = _field_default(feature, defaults)
    index = options.index(reference) if reference in options else 0
    missing = st.checkbox(
        "Not provided",
        key=f"default_risk_missing_{feature}",
        help="The fitted training pipeline will impute this value.",
    )
    value = st.selectbox(
        feature.replace("_", " ").title(),
        options,
        index=index,
        disabled=missing,
        key=f"default_risk_value_{feature}",
    )
    return None if missing else value


def render() -> None:
    """Render the independent loan repayment/default estimate interface."""
    st.title("Loan Default Risk Prediction")
    st.caption(
        "Separate module using historical LendingClub repayment outcomes; it "
        "does not use the historical `Loan_Status` approval model."
    )
    st.warning(DEFAULT_RISK_DISCLAIMER)

    try:
        model, metadata = _cached_model()
    except LoanPredictionError:
        st.info(
            "The LendingClub default-risk model is not registered yet. "
            "Run the documented training command after dataset validation."
        )
        return

    defaults = metadata.get("application_time_reference_values", {})
    applicant: dict[str, float | str | None] = {}
    with st.form("default_risk_prediction_form"):
        st.subheader("Loan and applicant information")
        left, right = st.columns(2)
        with left:
            for feature in NUMERIC_FEATURES[:8]:
                applicant[feature] = _render_numeric_field(feature, defaults)
            for feature in CATEGORICAL_FEATURES[:4]:
                applicant[feature] = _render_categorical_field(feature, defaults)
        with right:
            for feature in NUMERIC_FEATURES[8:]:
                applicant[feature] = _render_numeric_field(feature, defaults)
            for feature in CATEGORICAL_FEATURES[4:]:
                applicant[feature] = _render_categorical_field(feature, defaults)
        submitted = st.form_submit_button(
            "Estimate historical default risk",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return
    try:
        result = predict_default_risk(model, applicant, metadata)
        sensitivity = local_probability_sensitivity(model, applicant, metadata)
    except LoanPredictionError as exc:
        st.error(f"Default-risk estimate could not be calculated: {exc}")
        return

    st.divider()
    st.subheader("Estimated default risk")
    st.warning(DEFAULT_RISK_DISCLAIMER)
    probability, non_default, category = st.columns(3)
    probability.metric("Estimated default probability", f"{result['default_probability']:.1%}")
    non_default.metric(
        "Estimated non-default probability",
        f"{result['non_default_probability']:.1%}",
    )
    category.metric("Illustrative risk category", result["risk_category"])
    st.caption(
        f"Model `{result['model_name']}` version `{result['model_version']}`. "
        f"Dataset: `{metadata.get('dataset_file', 'unavailable')}`; target: "
        "`Charged Off` vs. `Fully Paid`."
    )
    st.caption(
        "Risk bands are configurable analytical categories for this academic "
        "prototype, not universal banking standards."
    )

    if sensitivity:
        st.subheader("Individual model sensitivity")
        st.caption(
            "One-feature-at-a-time replacement with a training-sample reference. "
            "These differences are descriptive, may form unrealistic profiles, "
            "and do not establish causal factors."
        )
        sensitivity_frame = pd.DataFrame(sensitivity)
        sensitivity_frame["change_vs_input"] = sensitivity_frame[
            "change_vs_input"
        ].map(lambda value: f"{value:+.1%}")
        st.dataframe(sensitivity_frame, use_container_width=True, hide_index=True)

    importance_path = REPORTS_DIR / "permutation_importance.csv"
    if importance_path.is_file():
        st.subheader("Global predictive contribution")
        st.caption(
            "Permutation importance is dataset-specific predictive contribution, "
            "not causation or a personal lending rationale."
        )
        importance = pd.read_csv(importance_path).head(10)
        st.bar_chart(importance.set_index("feature")["importance_mean"])


__all__ = ["render"]
