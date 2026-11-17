"""Decision report explanation and serialization utilities."""

from __future__ import annotations

from typing import Any

from decision_engine.schemas import DecisionReport


def report_to_dict(report: DecisionReport) -> dict[str, Any]:
    """Serialize a DecisionReport to a plain dictionary for display or export."""
    return {
        "application_id": report.application_id,
        "decision_timestamp": report.decision_timestamp,
        "policy_version": report.policy_version,
        "decision": report.decision,
        "application": {
            "application_id": report.application.application_id,
            "applicant_age": report.application.applicant_age,
            "annual_income": report.application.annual_income,
            "monthly_income": report.application.monthly_income,
            "monthly_debt_obligations": report.application.monthly_debt_obligations,
            "requested_loan_amount": report.application.requested_loan_amount,
            "loan_term_months": report.application.loan_term_months,
            "interest_rate": report.application.interest_rate,
            "employment_length_months": report.application.employment_length_months,
            "home_ownership": report.application.home_ownership,
            "verification_status": report.application.verification_status,
            "loan_purpose": report.application.loan_purpose,
            "residency_state": report.application.residency_state,
            "default_probability": report.application.default_probability,
            "risk_category": report.application.risk_category,
            "model_used": report.application.model_used,
            "missing_required_fields": report.application.missing_required_fields,
        },
        "financial_assessment": {
            "monthly_income": report.financial_assessment.monthly_income,
            "monthly_debt_obligations": report.financial_assessment.monthly_debt_obligations,
            "proposed_monthly_payment": report.financial_assessment.proposed_monthly_payment,
            "dti_percent": report.financial_assessment.dti_percent,
            "loan_to_income": report.financial_assessment.loan_to_income,
            "payment_to_income_percent": report.financial_assessment.payment_to_income_percent,
            "disposable_income": report.financial_assessment.disposable_income,
            "affordability_pass": report.financial_assessment.affordability_pass,
        },
        "risk_assessment": {
            "model_used": report.risk_assessment.model_used,
            "model_name": report.risk_assessment.model_name,
            "model_version": report.risk_assessment.model_version,
            "estimated_default_probability": report.risk_assessment.estimated_default_probability,
            "risk_category": report.risk_assessment.risk_category,
            "prediction_timestamp": report.risk_assessment.prediction_timestamp,
            "error": report.risk_assessment.error,
        },
        "rules": [
            {
                "rule_id": r.rule_id,
                "rule_name": r.rule_name,
                "category": r.category,
                "input_value": r.input_value,
                "threshold": r.threshold,
                "result": r.result,
                "explanation": r.explanation,
                "contribution": r.contribution,
            }
            for r in report.rules
        ],
        "passed_rules": report.passed_rules,
        "failed_rules": report.failed_rules,
        "warnings": report.warnings,
        "manual_review_reasons": report.manual_review_reasons,
        "decision_explanation": report.decision_explanation,
        "model_used": report.model_used,
    }


def format_currency(value: float | None) -> str:
    """Format a numeric value as currency, or return 'Not available'."""
    if value is None:
        return "Not available"
    return f"{value:,.2f}"


def format_percent(value: float | None) -> str:
    """Format a numeric value as a percentage, or return 'Not available'."""
    if value is None:
        return "Not available"
    return f"{value:.1f}%"


__all__ = ["report_to_dict", "format_currency", "format_percent"]
