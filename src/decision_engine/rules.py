"""Mandatory rejection and mandatory manual review rules."""

from __future__ import annotations

from typing import Any

from decision_engine.schemas import ApplicationInput, RuleResult


def evaluate_mandatory_rejection(
    application: ApplicationInput,
    policy: dict,
) -> list[RuleResult]:
    """Evaluate mandatory rejection rules.

    These represent explicit policy violations that require rejection.
    """
    results: list[RuleResult] = []
    financial_cfg = policy.get("financial", {})
    loan_cfg = policy.get("loan", {})

    # Requested loan exceeds absolute policy limit
    max_loan = financial_cfg.get("maximum_loan_amount")
    if max_loan is not None:
        if application.requested_loan_amount is None:
            results.append(
                RuleResult(
                    rule_id="REJ-001",
                    rule_name="Maximum loan amount (mandatory rejection)",
                    category="mandatory_rejection",
                    input_value=None,
                    threshold=max_loan,
                    result="not_applicable",
                    explanation="Requested loan amount is missing; cannot verify against maximum loan limit.",
                    contribution="none",
                )
            )
        elif application.requested_loan_amount > max_loan:
            results.append(
                RuleResult(
                    rule_id="REJ-001",
                    rule_name="Maximum loan amount (mandatory rejection)",
                    category="mandatory_rejection",
                    input_value=application.requested_loan_amount,
                    threshold=max_loan,
                    result="fail",
                    explanation=f"Requested loan amount {application.requested_loan_amount} exceeds absolute maximum {max_loan}.",
                    contribution="reject",
                )
            )
        else:
            results.append(
                RuleResult(
                    rule_id="REJ-001",
                    rule_name="Maximum loan amount (mandatory rejection)",
                    category="mandatory_rejection",
                    input_value=application.requested_loan_amount,
                    threshold=max_loan,
                    result="pass",
                    explanation=f"Requested loan amount {application.requested_loan_amount} is within maximum {max_loan}.",
                    contribution="none",
                )
            )

    # Minimum loan amount
    min_loan = loan_cfg.get("minimum_loan_amount")
    if min_loan is not None:
        if application.requested_loan_amount is None:
            results.append(
                RuleResult(
                    rule_id="REJ-002",
                    rule_name="Minimum loan amount (mandatory rejection)",
                    category="mandatory_rejection",
                    input_value=None,
                    threshold=min_loan,
                    result="not_applicable",
                    explanation="Requested loan amount is missing; cannot verify minimum loan amount.",
                    contribution="none",
                )
            )
        elif application.requested_loan_amount < min_loan:
            results.append(
                RuleResult(
                    rule_id="REJ-002",
                    rule_name="Minimum loan amount (mandatory rejection)",
                    category="mandatory_rejection",
                    input_value=application.requested_loan_amount,
                    threshold=min_loan,
                    result="fail",
                    explanation=f"Requested loan amount {application.requested_loan_amount} is below minimum {min_loan}.",
                    contribution="reject",
                )
            )
        else:
            results.append(
                RuleResult(
                    rule_id="REJ-002",
                    rule_name="Minimum loan amount (mandatory rejection)",
                    category="mandatory_rejection",
                    input_value=application.requested_loan_amount,
                    threshold=min_loan,
                    result="pass",
                    explanation=f"Requested loan amount {application.requested_loan_amount} meets minimum {min_loan}.",
                    contribution="none",
                )
            )

    # Loan term outside allowed range
    min_term = loan_cfg.get("minimum_term_months")
    max_term = loan_cfg.get("maximum_term_months")
    if min_term is not None or max_term is not None:
        term = application.loan_term_months
        if term is None:
            results.append(
                RuleResult(
                    rule_id="REJ-003",
                    rule_name="Loan term range (mandatory rejection)",
                    category="mandatory_rejection",
                    input_value=None,
                    threshold=f"{min_term} to {max_term} months",
                    result="not_applicable",
                    explanation="Loan term is missing; cannot verify term limits.",
                    contribution="none",
                )
            )
        else:
            term_fail = False
            reasons = []
            if min_term is not None and term < min_term:
                term_fail = True
                reasons.append(f"below minimum {min_term} months")
            if max_term is not None and term > max_term:
                term_fail = True
                reasons.append(f"exceeds maximum {max_term} months")
            if term_fail:
                results.append(
                    RuleResult(
                        rule_id="REJ-003",
                        rule_name="Loan term range (mandatory rejection)",
                        category="mandatory_rejection",
                        input_value=term,
                        threshold=f"{min_term} to {max_term} months",
                        result="fail",
                        explanation=f"Loan term {term} months is invalid: {'; '.join(reasons)}.",
                        contribution="reject",
                    )
                )
            else:
                results.append(
                    RuleResult(
                        rule_id="REJ-003",
                        rule_name="Loan term range (mandatory rejection)",
                        category="mandatory_rejection",
                        input_value=term,
                        threshold=f"{min_term} to {max_term} months",
                        result="pass",
                        explanation=f"Loan term {term} months is within allowed range.",
                        contribution="none",
                    )
                )

    # Interest rate outside allowed range
    min_rate = loan_cfg.get("minimum_interest_rate")
    max_rate = loan_cfg.get("maximum_interest_rate")
    if min_rate is not None or max_rate is not None:
        rate = application.interest_rate
        if rate is None:
            results.append(
                RuleResult(
                    rule_id="REJ-004",
                    rule_name="Interest rate range (mandatory rejection)",
                    category="mandatory_rejection",
                    input_value=None,
                    threshold=f"{min_rate}% to {max_rate}%",
                    result="not_applicable",
                    explanation="Interest rate is missing; cannot verify rate limits.",
                    contribution="none",
                )
            )
        else:
            rate_fail = False
            reasons = []
            if min_rate is not None and rate < min_rate:
                rate_fail = True
                reasons.append(f"below minimum {min_rate}%")
            if max_rate is not None and rate > max_rate:
                rate_fail = True
                reasons.append(f"exceeds maximum {max_rate}%")
            if rate_fail:
                results.append(
                    RuleResult(
                        rule_id="REJ-004",
                        rule_name="Interest rate range (mandatory rejection)",
                        category="mandatory_rejection",
                        input_value=rate,
                        threshold=f"{min_rate}% to {max_rate}%",
                        result="fail",
                        explanation=f"Interest rate {rate}% is invalid: {'; '.join(reasons)}.",
                        contribution="reject",
                    )
                )
            else:
                results.append(
                    RuleResult(
                        rule_id="REJ-004",
                        rule_name="Interest rate range (mandatory rejection)",
                        category="mandatory_rejection",
                        input_value=rate,
                        threshold=f"{min_rate}% to {max_rate}%",
                        result="pass",
                        explanation=f"Interest rate {rate}% is within allowed range.",
                        contribution="none",
                    )
                )

    return results


def evaluate_mandatory_manual_review(
    application: ApplicationInput,
    policy: dict,
    risk_assessment=None,
) -> list[RuleResult]:
    """Evaluate mandatory manual review rules.

    These require human review but not automatic rejection.
    """
    results: list[RuleResult] = []
    credit_risk_cfg = policy.get("credit_risk", {})

    # Missing required application fields
    required_fields = policy.get("eligibility", {}).get("required_application_fields", [])
    missing = application.missing_required_fields or []
    for field in required_fields:
        if field in missing:
            results.append(
                RuleResult(
                    rule_id=f"REV-001-{field}",
                    rule_name=f"Missing required field: {field}",
                    category="mandatory_review",
                    input_value=None,
                    threshold="required",
                    result="fail",
                    explanation=f"Required application field '{field}' is missing.",
                    contribution="refer",
                )
            )

    # Model unavailable but required for review
    if credit_risk_cfg.get("manual_review_if_model_unavailable", False):
        if risk_assessment is not None and not risk_assessment.model_used:
            results.append(
                RuleResult(
                    rule_id="REV-002",
                    rule_name="Default-risk model unavailable",
                    category="mandatory_review",
                    input_value=None,
                    threshold="model required",
                    result="fail",
                    explanation=f"Default-risk model is not available: {risk_assessment.error if risk_assessment.error else 'unknown error'}. Manual review required.",
                    contribution="refer",
                )
            )

    # Probability missing but required
    if credit_risk_cfg.get("manual_review_if_probability_missing", False):
        if (
            risk_assessment is not None
            and risk_assessment.model_used
            and risk_assessment.estimated_default_probability is None
        ):
            results.append(
                RuleResult(
                    rule_id="REV-003",
                    rule_name="Default probability missing",
                    category="mandatory_review",
                    input_value=None,
                    threshold="probability required",
                    result="fail",
                    explanation="Default-risk model ran but did not return a valid probability. Manual review required.",
                    contribution="refer",
                )
            )

    # Missing monthly income when verification required
    if policy.get("eligibility", {}).get("required_verification_status", False):
        if application.annual_income is None and application.monthly_income is None:
            results.append(
                RuleResult(
                    rule_id="REV-004",
                    rule_name="Income verification missing",
                    category="mandatory_review",
                    input_value=None,
                    threshold="income required",
                    result="fail",
                    explanation="Income information is missing but verification is required. Manual review required.",
                    contribution="refer",
                )
            )

    return results


__all__ = [
    "evaluate_mandatory_rejection",
    "evaluate_mandatory_manual_review",
]
