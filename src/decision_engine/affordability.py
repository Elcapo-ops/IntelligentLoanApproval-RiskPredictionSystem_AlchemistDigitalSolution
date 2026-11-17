"""Affordability rule evaluation using financial calculations."""

from __future__ import annotations

from decision_engine.financial_calculations import (
    calculate_affordability,
    calculate_dti,
    calculate_loan_to_income,
    calculate_monthly_installment,
)
from decision_engine.schemas import ApplicationInput, RuleResult


def evaluate_affordability(
    application: ApplicationInput,
    policy: dict,
    installment: float | None = None,
) -> list[RuleResult]:
    """Evaluate financial and affordability rules against the lending policy.

    Returns a list of RuleResult objects.
    """
    financial_cfg = policy.get("financial", {})
    results: list[RuleResult] = []

    # Calculate monthly installment if not provided
    if installment is None and application.requested_loan_amount and application.loan_term_months and application.interest_rate is not None:
        try:
            installment_calc = calculate_monthly_installment(
                principal=application.requested_loan_amount,
                annual_interest_rate=application.interest_rate,
                term_months=application.loan_term_months,
            )
            installment = installment_calc.monthly_payment
        except (ValueError, TypeError):
            installment = None

    # DTI calculation
    dti_calc = calculate_dti(
        total_monthly_debt=application.monthly_debt_obligations,
        gross_monthly_income=application.monthly_income,
    )

    # Loan-to-income calculation
    lti_calc = calculate_loan_to_income(
        loan_amount=application.requested_loan_amount,
        annual_income=(application.annual_income or (application.monthly_income * 12 if application.monthly_income else None)),
    )

    # Affordability calculation
    afford_calc = calculate_affordability(
        gross_monthly_income=application.monthly_income,
        monthly_debt_obligations=application.monthly_debt_obligations,
        proposed_monthly_payment=installment,
        dti_percent=dti_calc.dti_percent,
    )

    # Maximum loan amount
    max_loan = financial_cfg.get("maximum_loan_amount", float("inf"))
    if application.requested_loan_amount is None:
        results.append(
            RuleResult(
                rule_id="FIN-001",
                rule_name="Maximum loan amount",
                category="financial",
                input_value=None,
                threshold=max_loan,
                result="not_applicable",
                explanation="Requested loan amount is not provided.",
                contribution="none",
            )
        )
    elif application.requested_loan_amount > max_loan:
        results.append(
            RuleResult(
                rule_id="FIN-001",
                rule_name="Maximum loan amount",
                category="financial",
                input_value=application.requested_loan_amount,
                threshold=max_loan,
                result="fail",
                explanation=f"Requested loan amount {application.requested_loan_amount} exceeds maximum {max_loan}.",
                contribution="reject",
            )
        )
    else:
        results.append(
            RuleResult(
                rule_id="FIN-001",
                rule_name="Maximum loan amount",
                category="financial",
                input_value=application.requested_loan_amount,
                threshold=max_loan,
                result="pass",
                explanation=f"Requested loan amount {application.requested_loan_amount} is within maximum {max_loan}.",
                contribution="none",
            )
        )

    # Minimum verified monthly income
    min_income = financial_cfg.get("minimum_verified_monthly_income", 0.0)
    if application.monthly_income is None:
        results.append(
            RuleResult(
                rule_id="FIN-002",
                rule_name="Minimum verified monthly income",
                category="financial",
                input_value=None,
                threshold=min_income,
                result="not_applicable",
                explanation="Monthly income is not provided; cannot verify minimum income requirement.",
                contribution="none",
            )
        )
    elif application.monthly_income < min_income:
        results.append(
            RuleResult(
                rule_id="FIN-002",
                rule_name="Minimum verified monthly income",
                category="financial",
                input_value=application.monthly_income,
                threshold=min_income,
                result="fail",
                explanation=f"Monthly income {application.monthly_income} is below minimum {min_income}.",
                contribution="reject",
            )
        )
    else:
        results.append(
            RuleResult(
                rule_id="FIN-002",
                rule_name="Minimum verified monthly income",
                category="financial",
                input_value=application.monthly_income,
                threshold=min_income,
                result="pass",
                explanation=f"Monthly income {application.monthly_income} meets minimum {min_income}.",
                contribution="none",
            )
        )

    # DTI check
    max_dti = financial_cfg.get("maximum_dti_percent", 100.0)
    if not dti_calc.available:
        results.append(
            RuleResult(
                rule_id="FIN-003",
                rule_name="Maximum debt-to-income ratio",
                category="financial",
                input_value=None,
                threshold=max_dti,
                result="not_applicable",
                explanation="DTI cannot be calculated because monthly debt obligations or gross monthly income is missing.",
                contribution="none",
            )
        )
    elif dti_calc.dti_percent is not None and dti_calc.dti_percent > max_dti:
        results.append(
            RuleResult(
                rule_id="FIN-003",
                rule_name="Maximum debt-to-income ratio",
                category="financial",
                input_value=dti_calc.dti_percent,
                threshold=max_dti,
                result="fail",
                explanation=f"DTI {dti_calc.dti_percent}% exceeds maximum {max_dti}%.",
                contribution="reject",
            )
        )
    else:
        results.append(
            RuleResult(
                rule_id="FIN-003",
                rule_name="Maximum debt-to-income ratio",
                category="financial",
                input_value=dti_calc.dti_percent,
                threshold=max_dti,
                result="pass",
                explanation=f"DTI {dti_calc.dti_percent}% is within maximum {max_dti}%."
                if dti_calc.dti_percent is not None
                else "DTI not available.",
                contribution="none",
            )
        )

    # Loan-to-income check
    max_lti = financial_cfg.get("maximum_loan_to_income", float("inf"))
    if not lti_calc.available:
        results.append(
            RuleResult(
                rule_id="FIN-004",
                rule_name="Maximum loan-to-income ratio",
                category="financial",
                input_value=None,
                threshold=max_lti,
                result="not_applicable",
                explanation="Loan-to-income cannot be calculated because loan amount or annual income is missing.",
                contribution="none",
            )
        )
    elif lti_calc.ratio is not None and lti_calc.ratio > max_lti:
        results.append(
            RuleResult(
                rule_id="FIN-004",
                rule_name="Maximum loan-to-income ratio",
                category="financial",
                input_value=lti_calc.ratio,
                threshold=max_lti,
                result="fail",
                explanation=f"Loan-to-income {lti_calc.ratio} exceeds maximum {max_lti}.",
                contribution="reject",
            )
        )
    else:
        results.append(
            RuleResult(
                rule_id="FIN-004",
                rule_name="Maximum loan-to-income ratio",
                category="financial",
                input_value=lti_calc.ratio,
                threshold=max_lti,
                result="pass",
                explanation=f"Loan-to-income {lti_calc.ratio} is within maximum {max_lti}."
                if lti_calc.ratio is not None
                else "Ratio not available.",
                contribution="none",
            )
        )

    # Payment-to-income check
    max_pti = financial_cfg.get("maximum_payment_to_income_percent", 100.0)
    if afford_calc.payment_to_income_percent is None:
        results.append(
            RuleResult(
                rule_id="FIN-005",
                rule_name="Maximum payment-to-income ratio",
                category="financial",
                input_value=None,
                threshold=max_pti,
                result="not_applicable",
                explanation="Payment-to-income cannot be calculated because monthly income or proposed payment is missing.",
                contribution="none",
            )
        )
    elif afford_calc.payment_to_income_percent is not None and afford_calc.payment_to_income_percent > max_pti:
        results.append(
            RuleResult(
                rule_id="FIN-005",
                rule_name="Maximum payment-to-income ratio",
                category="financial",
                input_value=afford_calc.payment_to_income_percent,
                threshold=max_pti,
                result="fail",
                explanation=f"Payment-to-income {afford_calc.payment_to_income_percent}% exceeds maximum {max_pti}%.",
                contribution="reject",
            )
        )
    else:
        results.append(
            RuleResult(
                rule_id="FIN-005",
                rule_name="Maximum payment-to-income ratio",
                category="financial",
                input_value=afford_calc.payment_to_income_percent,
                threshold=max_pti,
                result="pass",
                explanation=f"Payment-to-income {afford_calc.payment_to_income_percent}% is within maximum {max_pti}%."
                if afford_calc.payment_to_income_percent is not None
                else "Ratio not available.",
                contribution="none",
            )
        )

    # Minimum disposable income
    min_disposable = financial_cfg.get("minimum_disposable_income", 0.0)
    if afford_calc.disposable_income is None:
        results.append(
            RuleResult(
                rule_id="FIN-006",
                rule_name="Minimum disposable income",
                category="financial",
                input_value=None,
                threshold=min_disposable,
                result="not_applicable",
                explanation="Disposable income cannot be calculated because monthly debt obligations are missing.",
                contribution="none",
            )
        )
    elif afford_calc.disposable_income is not None and afford_calc.disposable_income < min_disposable:
        results.append(
            RuleResult(
                rule_id="FIN-006",
                rule_name="Minimum disposable income",
                category="financial",
                input_value=afford_calc.disposable_income,
                threshold=min_disposable,
                result="fail",
                explanation=f"Disposable income {afford_calc.disposable_income} is below minimum {min_disposable}.",
                contribution="reject",
            )
        )
    else:
        results.append(
            RuleResult(
                rule_id="FIN-006",
                rule_name="Minimum disposable income",
                category="financial",
                input_value=afford_calc.disposable_income,
                threshold=min_disposable,
                result="pass",
                explanation=f"Disposable income {afford_calc.disposable_income} meets minimum {min_disposable}."
                if afford_calc.disposable_income is not None
                else "Disposable income not available.",
                contribution="none",
            )
        )

    return results


__all__ = ["evaluate_affordability"]
