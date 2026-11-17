"""Data schemas for the lending decision engine.

Defines input, rule result, and decision report structures with validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class AffordabilityCalculation:
    """Financial calculations for an application (alias for FinancialAssessment)."""

    monthly_income: float | None
    monthly_debt_obligations: float | None
    proposed_monthly_payment: float | None
    dti_percent: float | None
    loan_to_income: float | None
    payment_to_income_percent: float | None
    disposable_income: float | None
    affordability_pass: bool | None


@dataclass(frozen=True)
class ApplicationInput:
    """Validated loan application input for decision evaluation."""

    application_id: str
    applicant_age: int | None
    annual_income: float | None
    monthly_income: float | None
    monthly_debt_obligations: float | None
    requested_loan_amount: float | None
    loan_term_months: int | None
    interest_rate: float | None
    employment_length_months: int | None
    home_ownership: str | None
    verification_status: str | None
    loan_purpose: str | None
    residency_state: str | None
    default_probability: float | None
    risk_category: str | None
    model_version: str | None
    model_used: bool
    missing_required_fields: list[str] = field(default_factory=list)

    def is_complete(self) -> bool:
        """Return True if no required fields are missing."""
        return len(self.missing_required_fields) == 0


@dataclass(frozen=True)
class RuleResult:
    """Result of evaluating a single policy rule."""

    rule_id: str
    rule_name: str
    category: str  # mandatory_rejection, mandatory_review, eligibility, financial, risk, warning
    input_value: Any
    threshold: Any
    result: str  # pass, fail, not_applicable
    explanation: str
    contribution: str  # reject, refer, approve, none


@dataclass(frozen=True)
class FinancialAssessment:
    """Financial calculations for an application."""

    monthly_income: float | None
    monthly_debt_obligations: float | None
    proposed_monthly_payment: float | None
    dti_percent: float | None
    loan_to_income: float | None
    payment_to_income_percent: float | None
    disposable_income: float | None
    affordability_pass: bool | None


@dataclass(frozen=True)
class RiskAssessment:
    """Default-risk model assessment, if available."""

    model_used: bool
    model_name: str | None
    model_version: str | None
    estimated_default_probability: float | None
    risk_category: str | None
    prediction_timestamp: str | None
    error: str | None


@dataclass(frozen=True)
class DecisionReport:
    """Complete structured decision report."""

    application_id: str
    decision_timestamp: str
    policy_version: str
    decision: str  # APPROVE, REJECT, REFER FOR MANUAL REVIEW
    application: ApplicationInput
    financial_assessment: FinancialAssessment
    risk_assessment: RiskAssessment
    rules: list[RuleResult]
    passed_rules: list[str]
    failed_rules: list[str]
    warnings: list[str]
    manual_review_reasons: list[str]
    decision_explanation: str
    model_used: bool


@dataclass(frozen=True)
class DecisionOutcome:
    """Primary decision outcome."""

    decision: str
    reasons: list[str]
    warnings: list[str]
    manual_review_reasons: list[str]

    def is_approved(self) -> bool:
        return self.decision == "APPROVE"

    def is_rejected(self) -> bool:
        return self.decision == "REJECT"

    def is_referral(self) -> bool:
        return self.decision == "REFER FOR MANUAL REVIEW"


__all__ = [
    "AffordabilityCalculation",
    "ApplicationInput",
    "DecisionOutcome",
    "DecisionReport",
    "FinancialAssessment",
    "RiskAssessment",
    "RuleResult",
]
