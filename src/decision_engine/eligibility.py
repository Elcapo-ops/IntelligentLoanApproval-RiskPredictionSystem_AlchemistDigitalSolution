"""Applicant eligibility rule evaluation."""

from __future__ import annotations

from typing import Any

from decision_engine.schemas import ApplicationInput, RuleResult


def evaluate_eligibility(
    application: ApplicationInput,
    policy: dict,
) -> list[RuleResult]:
    """Evaluate applicant eligibility rules against the lending policy.

    Returns a list of RuleResult objects, one per eligibility rule.
    Does not reject on missing fields alone — that is handled by mandatory
    review rules. Missing required fields are recorded as not_applicable
    with an explanation.
    """
    eligibility_cfg = policy.get("eligibility", {})
    results: list[RuleResult] = []

    # Minimum age
    min_age = eligibility_cfg.get("minimum_age", 18)
    if application.applicant_age is None:
        results.append(
            RuleResult(
                rule_id="ELIG-001",
                rule_name="Minimum applicant age",
                category="eligibility",
                input_value=None,
                threshold=min_age,
                result="not_applicable",
                explanation="Applicant age is not provided; cannot verify minimum age requirement.",
                contribution="none",
            )
        )
    elif application.applicant_age < min_age:
        results.append(
            RuleResult(
                rule_id="ELIG-001",
                rule_name="Minimum applicant age",
                category="eligibility",
                input_value=application.applicant_age,
                threshold=min_age,
                result="fail",
                explanation=f"Applicant age {application.applicant_age} is below minimum required age {min_age}.",
                contribution="reject",
            )
        )
    else:
        results.append(
            RuleResult(
                rule_id="ELIG-001",
                rule_name="Minimum applicant age",
                category="eligibility",
                input_value=application.applicant_age,
                threshold=min_age,
                result="pass",
                explanation=f"Applicant age {application.applicant_age} meets minimum required age {min_age}.",
                contribution="none",
            )
        )

    # Minimum employment length
    min_emp = eligibility_cfg.get("minimum_employment_months", 0)
    if application.employment_length_months is None:
        results.append(
            RuleResult(
                rule_id="ELIG-002",
                rule_name="Minimum employment length",
                category="eligibility",
                input_value=None,
                threshold=min_emp,
                result="not_applicable",
                explanation="Employment length is not provided; cannot verify minimum employment requirement.",
                contribution="none",
            )
        )
    elif application.employment_length_months < min_emp:
        results.append(
            RuleResult(
                rule_id="ELIG-002",
                rule_name="Minimum employment length",
                category="eligibility",
                input_value=application.employment_length_months,
                threshold=min_emp,
                result="fail",
                explanation=f"Employment length {application.employment_length_months} months is below minimum {min_emp} months.",
                contribution="reject",
            )
        )
    else:
        results.append(
            RuleResult(
                rule_id="ELIG-002",
                rule_name="Minimum employment length",
                category="eligibility",
                input_value=application.employment_length_months,
                threshold=min_emp,
                result="pass",
                explanation=f"Employment length {application.employment_length_months} months meets minimum {min_emp} months.",
                contribution="none",
            )
        )

    # Home ownership required
    if eligibility_cfg.get("required_home_ownership", False):
        if application.home_ownership is None:
            results.append(
                RuleResult(
                    rule_id="ELIG-003",
                    rule_name="Home ownership required",
                    category="eligibility",
                    input_value=None,
                    threshold="required",
                    result="not_applicable",
                    explanation="Home ownership status is not provided.",
                    contribution="none",
                )
            )
        elif not application.home_ownership.strip():
            results.append(
                RuleResult(
                    rule_id="ELIG-003",
                    rule_name="Home ownership required",
                    category="eligibility",
                    input_value=application.home_ownership,
                    threshold="required",
                    result="fail",
                    explanation="Home ownership status is required but was not provided.",
                    contribution="reject",
                )
            )
        else:
            results.append(
                RuleResult(
                    rule_id="ELIG-003",
                    rule_name="Home ownership required",
                    category="eligibility",
                    input_value=application.home_ownership,
                    threshold="required",
                    result="pass",
                    explanation=f"Home ownership provided: {application.home_ownership}.",
                    contribution="none",
                )
            )

    # Verification status required
    if eligibility_cfg.get("required_verification_status", False):
        if application.verification_status is None:
            results.append(
                RuleResult(
                    rule_id="ELIG-004",
                    rule_name="Verification status required",
                    category="eligibility",
                    input_value=None,
                    threshold="required",
                    result="not_applicable",
                    explanation="Income verification status is not provided.",
                    contribution="none",
                )
            )
        elif not application.verification_status.strip():
            results.append(
                RuleResult(
                    rule_id="ELIG-004",
                    rule_name="Verification status required",
                    category="eligibility",
                    input_value=application.verification_status,
                    threshold="required",
                    result="fail",
                    explanation="Verification status is required but was not provided.",
                    contribution="reject",
                )
            )
        else:
            results.append(
                RuleResult(
                    rule_id="ELIG-004",
                    rule_name="Verification status required",
                    category="eligibility",
                    input_value=application.verification_status,
                    threshold="required",
                    result="pass",
                    explanation=f"Verification status provided: {application.verification_status}.",
                    contribution="none",
                )
            )

    # Loan purpose eligibility
    eligible_purposes = eligibility_cfg.get("eligible_loan_purposes", [])
    if application.loan_purpose is None:
        results.append(
            RuleResult(
                rule_id="ELIG-005",
                rule_name="Eligible loan purpose",
                category="eligibility",
                input_value=None,
                threshold=eligible_purposes,
                result="not_applicable",
                explanation="Loan purpose is not provided; cannot verify eligibility.",
                contribution="none",
            )
        )
    elif application.loan_purpose not in eligible_purposes:
        results.append(
            RuleResult(
                rule_id="ELIG-005",
                rule_name="Eligible loan purpose",
                category="eligibility",
                input_value=application.loan_purpose,
                threshold=eligible_purposes,
                result="fail",
                explanation=f"Loan purpose '{application.loan_purpose}' is not in the eligible purposes list.",
                contribution="reject",
            )
        )
    else:
        results.append(
            RuleResult(
                rule_id="ELIG-005",
                rule_name="Eligible loan purpose",
                category="eligibility",
                input_value=application.loan_purpose,
                threshold=eligible_purposes,
                result="pass",
                explanation=f"Loan purpose '{application.loan_purpose}' is eligible.",
                contribution="none",
            )
        )

    # Residency state eligibility
    eligible_states = eligibility_cfg.get("eligible_residency_states")
    if eligible_states is not None:
        if application.residency_state is None:
            results.append(
                RuleResult(
                    rule_id="ELIG-006",
                    rule_name="Eligible residency state",
                    category="eligibility",
                    input_value=None,
                    threshold=eligible_states,
                    result="not_applicable",
                    explanation="Residency state is not provided; cannot verify jurisdiction eligibility.",
                    contribution="none",
                )
            )
        elif application.residency_state not in eligible_states:
            results.append(
                RuleResult(
                    rule_id="ELIG-006",
                    rule_name="Eligible residency state",
                    category="eligibility",
                    input_value=application.residency_state,
                    threshold=eligible_states,
                    result="fail",
                    explanation=f"Residency state '{application.residency_state}' is not in the eligible states list.",
                    contribution="reject",
                )
            )
        else:
            results.append(
                RuleResult(
                    rule_id="ELIG-006",
                    rule_name="Eligible residency state",
                    category="eligibility",
                    input_value=application.residency_state,
                    threshold=eligible_states,
                    result="pass",
                    explanation=f"Residency state '{application.residency_state}' is eligible.",
                    contribution="none",
                )
            )

    return results


__all__ = ["evaluate_eligibility"]
