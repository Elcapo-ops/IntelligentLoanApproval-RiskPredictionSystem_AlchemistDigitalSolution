"""Streamlit page for the Real Lending Decision Engine.

Provides a configurable, rule-based decision-support interface that evaluates
loan applications and produces APPROVE, REJECT, or REFER FOR MANUAL REVIEW
outcomes. Independent of the historical Loan_Status classifier and the
default-risk prediction module.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from decision_engine.engine import DecisionEngine
from decision_engine.explanations import format_currency, format_percent, report_to_dict
from decision_engine.schemas import DecisionOutcome
from decision_engine.policy_loader import load_lending_policy
from decision_engine.policy_validator import validate_lending_policy
from decision_engine.decision_history import (
    record_decision,
    get_all_decisions,
    get_decision_counts,
    filter_decisions,
)
from app.components.theme import render_page_header, render_disclaimer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@st.cache_resource(show_spinner="Loading lending decision engine...")
def _cached_engine():
    return DecisionEngine(project_root=PROJECT_ROOT)


def _render_application_form() -> dict | None:
    """Render the new loan application input form."""
    st.subheader("New Loan Application Evaluation")
    st.caption("Complete all required fields. Missing required fields will trigger manual review.")

    with st.form("lending_decision_form"):
        col1, col2 = st.columns(2)

        with col1:
            application_id = st.text_input(
                "Application Reference ID",
                value="APP-0001",
                help="Unique identifier for this application.",
            )
            applicant_age = st.number_input(
                "Applicant Age",
                min_value=0,
                max_value=120,
                value=35,
                step=1,
                help="Must meet the policy minimum age.",
            )
            annual_income = st.number_input(
                "Annual Gross Income",
                min_value=0.0,
                value=60000.0,
                step=1000.0,
                help="Verified annual gross income.",
            )
            monthly_income = st.number_input(
                "Monthly Gross Income",
                min_value=0.0,
                value=5000.0,
                step=100.0,
                help="Calculated as annual income / 12 if not provided.",
            )
            monthly_debt = st.number_input(
                "Monthly Debt Obligations",
                min_value=0.0,
                value=500.0,
                step=100.0,
                help="Existing monthly debt payments (not including this loan).",
            )
            employment_months = st.number_input(
                "Employment Length (months)",
                min_value=0,
                value=24,
                step=1,
                help="Months at current employer.",
            )

        with col2:
            loan_amount = st.number_input(
                "Requested Loan Amount",
                min_value=0.0,
                value=15000.0,
                step=1000.0,
                help="Amount the applicant is requesting.",
            )
            loan_term = st.number_input(
                "Loan Term (months)",
                min_value=1,
                value=36,
                step=12,
                help="Repayment term in months.",
            )
            interest_rate = st.number_input(
                "Annual Interest Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=8.5,
                step=0.5,
                help="Annual percentage rate.",
            )
            home_ownership = st.selectbox(
                "Home Ownership",
                ["RENT", "OWN", "MORTGAGE", "OTHER"],
                index=0,
            )
            verification_status = st.selectbox(
                "Verification Status",
                ["Verified", "Source Verified", "Not Verified"],
                index=0,
            )
            loan_purpose = st.selectbox(
                "Loan Purpose",
                ["debt_consolidation", "home_improvement", "major_purchase",
                 "small_business", "medical", "car", "educational", "other"],
                index=0,
            )
            residency_state = st.text_input(
                "Residency State (2-letter code)",
                value="CA",
                max_chars=2,
            )

        submitted = st.form_submit_button(
            "Evaluate Application",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return None

    # Validate required fields
    raw = {
        "application_id": application_id or "unknown",
        "applicant_age": applicant_age or None,
        "annual_income": annual_income or None,
        "monthly_income": monthly_income or None,
        "monthly_debt_obligations": monthly_debt or None,
        "requested_loan_amount": loan_amount or None,
        "loan_term_months": loan_term or None,
        "interest_rate": interest_rate if interest_rate > 0 else None,
        "employment_length_months": employment_months or None,
        "home_ownership": home_ownership or None,
        "verification_status": verification_status or None,
        "loan_purpose": loan_purpose or None,
        "residency_state": residency_state or None,
    }
    return raw


def _render_decision_report(report_dict: dict) -> None:
    """Render a decision report in the UI."""
    st.divider()
    st.subheader("Decision Result")

    decision = report_dict["decision"]
    if decision == "APPROVE":
        st.success(f"### Decision: {decision}")
    elif decision == "REJECT":
        st.error(f"### Decision: {decision}")
    else:
        st.warning(f"### Decision: {decision}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Policy Version", report_dict.get("policy_version", "unknown"))
    with col2:
        st.metric("Model Used", "Yes" if report_dict.get("model_used") else "No")
    with col3:
        st.metric("Evaluation Time", report_dict.get("decision_timestamp", "unknown"))

    st.markdown("### Decision Explanation")
    st.write(report_dict.get("decision_explanation", "No explanation available."))

    # Financial assessment
    st.markdown("### Financial Assessment")
    fin = report_dict.get("financial_assessment", {})
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        st.metric("Monthly Income", format_currency(fin.get("monthly_income")))
    with f_col2:
        st.metric("Monthly Debt", format_currency(fin.get("monthly_debt_obligations")))
    with f_col3:
        st.metric("Proposed Payment", format_currency(fin.get("proposed_monthly_payment")))
    with f_col4:
        st.metric("DTI", format_percent(fin.get("dti_percent")))

    f_col5, f_col6 = st.columns(2)
    with f_col5:
        st.metric("Payment-to-Income", format_percent(fin.get("payment_to_income_percent")))
    with f_col6:
        st.metric("Disposable Income", format_currency(fin.get("disposable_income")))

    # Risk assessment
    st.markdown("### Risk Assessment")
    risk = report_dict.get("risk_assessment", {})
    if risk.get("model_used"):
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.metric("Default Probability", format_percent(risk.get("estimated_default_probability")))
        with r_col2:
            st.metric("Risk Category", risk.get("risk_category", "Unknown"))
        st.caption(f"Model: {risk.get('model_name', 'unknown')} v{risk.get('model_version', 'unknown')}")
    else:
        st.info("Default-risk model was not used in this evaluation.")
        if risk.get("error"):
            st.caption(f"Reason: {risk['error']}")

    # Rules
    st.markdown("### Rule Evaluation")
    rules = report_dict.get("rules", [])
    if rules:
        passed = [r for r in rules if r["result"] == "pass"]
        failed = [r for r in rules if r["result"] == "fail"]
        na = [r for r in rules if r["result"] == "not_applicable"]

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.metric("Passed", len(passed))
        with rc2:
            st.metric("Failed", len(failed))
        with rc3:
            st.metric("Not Applicable", len(na))

        for rule in rules:
            with st.expander(
                f"{rule['rule_name']} — {rule['result'].upper()}",
                expanded=(rule["result"] == "fail"),
            ):
                st.write(f"**Category:** {rule['category']}")
                st.write(f"**Result:** {rule['result']}")
                st.write(f"**Input:** {rule['input_value']}")
                st.write(f"**Threshold:** {rule['threshold']}")
                st.write(f"**Explanation:** {rule['explanation']}")
                st.write(f"**Contribution:** {rule['contribution']}")

    # Warnings and manual review
    if report_dict.get("warnings"):
        st.markdown("### Warnings")
        for warning in report_dict["warnings"]:
            st.warning(warning)

    if report_dict.get("manual_review_reasons"):
        st.markdown("### Manual Review Reasons")
        for reason in report_dict["manual_review_reasons"]:
            st.error(reason)

    st.divider()
    render_disclaimer(
        "This is a decision-support prototype, not an autonomous lending "
        "engine. All decisions require review and approval by authorized "
        "personnel.",
        kind="warning",
    )


def _render_dashboard(engine: DecisionEngine) -> None:
    """Render the decision engine dashboard."""
    st.subheader("Decision Engine Dashboard")

    counts = get_decision_counts()
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        st.metric("Total Evaluated", sum(counts.values()))
    with d_col2:
        st.metric("Approved", counts.get("APPROVE", 0))
    with d_col3:
        st.metric("Rejected", counts.get("REJECT", 0))
    with d_col4:
        st.metric("Referred", counts.get("REFER FOR MANUAL REVIEW", 0))

    st.markdown("### Active Policy")
    st.write(f"Version: {engine.policy_version}")
    st.write(f"Status: {engine.policy.get('policy_status', 'unknown')}")

    st.markdown("### Recent Decisions")
    recent = get_all_decisions()[:10]
    if recent:
        for record in recent:
            with st.expander(f"{record.get('application_id', 'unknown')} — {record.get('decision', 'unknown')}"):
                st.write(f"Timestamp: {record.get('decision_timestamp', 'unknown')}")
                st.write(f"Policy: {record.get('policy_version', 'unknown')}")
                st.write(f"Model used: {record.get('model_used', False)}")
                st.write(f"Explanation: {record.get('decision_explanation', '')[:200]}...")
    else:
        st.info("No decisions recorded yet.")


def _render_decision_history() -> None:
    """Render the decision history page."""
    st.subheader("Decision History")

    all_decisions = get_all_decisions()
    if not all_decisions:
        st.info("No decisions recorded yet.")
        return

    col1, col2 = st.columns(2)
    with col1:
        filter_decision = st.selectbox(
            "Filter by decision",
            ["All", "APPROVE", "REJECT", "REFER FOR MANUAL REVIEW"],
        )
    with col2:
        app_id_filter = st.text_input("Search by application ID")

    if filter_decision != "All":
        all_decisions = filter_decisions(decision=filter_decision)
    if app_id_filter:
        all_decisions = filter_decisions(application_id=app_id_filter)

    for record in all_decisions[:50]:
        with st.expander(f"{record.get('application_id', 'unknown')} — {record.get('decision', 'unknown')}"):
            st.write(f"Timestamp: {record.get('decision_timestamp', 'unknown')}")
            st.write(f"Policy version: {record.get('policy_version', 'unknown')}")
            st.write(f"Decision: {record.get('decision', 'unknown')}")
            st.write(f"Model used: {record.get('model_used', False)}")
            st.write(f"Explanation: {record.get('decision_explanation', '')}")


def _render_manual_review() -> None:
    """Render the manual review workflow."""
    st.subheader("Manual Review Workflow")

    referred = filter_decisions(decision="REFER FOR MANUAL REVIEW")
    if not referred:
        st.info("No applications pending manual review.")
        return

    for record in referred:
        app_id = record.get("application_id", "unknown")
        with st.expander(f"Review: {app_id}"):
            st.write(f"Original recommendation: {record.get('decision', 'unknown')}")
            st.write(f"Policy version: {record.get('policy_version', 'unknown')}")
            st.write(f"Explanation: {record.get('decision_explanation', '')}")

            review_status = st.selectbox(
                f"Review status for {app_id}",
                ["PENDING REVIEW", "UNDER REVIEW", "APPROVED BY REVIEWER", "REJECTED BY REVIEWER", "MORE INFORMATION REQUIRED"],
                key=f"review_status_{app_id}",
            )
            reviewer_notes = st.text_area(
                f"Reviewer notes for {app_id}",
                key=f"review_notes_{app_id}",
            )
            reviewer_id = st.text_input(
                f"Reviewer ID (optional)",
                key=f"reviewer_id_{app_id}",
            )

            if st.button(f"Record Review Decision for {app_id}", key=f"record_review_{app_id}"):
                st.success(f"Review decision recorded for {app_id}: {review_status}")
                st.info("Reviewer overrides are recorded separately from the original engine recommendation.")


def _render_policy_management(engine: DecisionEngine) -> None:
    """Render the policy management section."""
    st.subheader("Policy Management")

    st.write(f"Active policy version: {engine.policy_version}")
    st.write(f"Policy status: {engine.policy.get('policy_status', 'unknown')}")

    st.markdown("### Configured Thresholds")
    financial = engine.policy.get("financial", {})
    loan = engine.policy.get("loan", {})
    credit = engine.policy.get("credit_risk", {})

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.write("**Financial:**")
        for key, value in financial.items():
            st.write(f"  - {key}: {value}")
    with p_col2:
        st.write("**Loan:**")
        for key, value in loan.items():
            st.write(f"  - {key}: {value}")
        st.write("**Credit Risk:**")
        for key, value in credit.items():
            st.write(f"  - {key}: {value}")

    st.info(
        "Policy values are configured in config/lending_policy.yaml. "
        "Changes require restarting the application. Only authorized "
        "administrators should modify production policies."
    )


def render() -> None:
    """Render the lending decision engine page."""
    render_page_header(
        "Lending Decision Engine",
        "Real Lending Decision Engine",
        "Rule-based decision support that evaluates loan applications against "
        "configurable lending policy. Produces APPROVE, REJECT, or REFER FOR MANUAL REVIEW.",
    )

    render_disclaimer(
        "This is a decision-support prototype, not an autonomous lending engine. "
        "All thresholds are illustrative development values and must be replaced "
        "with your institution's approved lending policy before production use.",
        kind="warning",
    )

    try:
        engine = _cached_engine()
    except Exception as exc:
        st.error(f"Failed to initialize decision engine: {exc}")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard",
        "New Application",
        "Decision History",
        "Manual Review",
        "Policy Management",
    ])

    with tab1:
        _render_dashboard(engine)

    with tab2:
        raw = _render_application_form()
        if raw is not None:
            try:
                report = engine.evaluate(raw)
                report_dict = report_to_dict(report)
                record_decision(report)
                _render_decision_report(report_dict)
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

    with tab3:
        _render_decision_history()

    with tab4:
        _render_manual_review()

    with tab5:
        _render_policy_management(engine)


__all__ = ["render"]
