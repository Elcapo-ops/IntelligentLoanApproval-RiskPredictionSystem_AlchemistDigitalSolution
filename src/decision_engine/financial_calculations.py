"""Reusable financial calculation functions for the lending decision engine.

All functions are pure and independently testable. They do not access models,
databases, or external services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DTICalculation:
    """Debt-to-income ratio result."""

    dti_percent: float | None
    total_monthly_debt: float | None
    gross_monthly_income: float | None
    available: bool


@dataclass(frozen=True)
class LoanToIncomeCalculation:
    """Loan-to-income ratio result."""

    ratio: float | None
    loan_amount: float | None
    annual_income: float | None
    available: bool


@dataclass(frozen=True)
class InstallmentCalculation:
    """Monthly loan installment result."""

    monthly_payment: float | None
    principal: float
    annual_interest_rate: float
    term_months: int
    zero_interest: bool


@dataclass(frozen=True)
class AffordabilityCalculation:
    """Affordability assessment result."""

    gross_monthly_income: float | None
    monthly_debt_obligations: float | None
    proposed_monthly_payment: float | None
    payment_to_income_percent: float | None
    remaining_monthly_income: float | None
    disposable_income: float | None
    dti_percent: float | None
    available: bool


def calculate_dti(
    total_monthly_debt: float | None,
    gross_monthly_income: float | None,
) -> DTICalculation:
    """Calculate debt-to-income ratio.

    DTI = (Total Monthly Debt Obligations / Gross Monthly Income) × 100

    Only calculates when both values are available and valid. Does not treat
    the monthly loan payment alone as total debt obligations.
    """
    if (
        total_monthly_debt is None
        or gross_monthly_income is None
        or gross_monthly_income <= 0
        or total_monthly_debt < 0
    ):
        return DTICalculation(
            dti_percent=None,
            total_monthly_debt=total_monthly_debt,
            gross_monthly_income=gross_monthly_income,
            available=False,
        )
    dti = (total_monthly_debt / gross_monthly_income) * 100
    return DTICalculation(
        dti_percent=round(dti, 2),
        total_monthly_debt=total_monthly_debt,
        gross_monthly_income=gross_monthly_income,
        available=True,
    )


def calculate_loan_to_income(
    loan_amount: float | None,
    annual_income: float | None,
) -> LoanToIncomeCalculation:
    """Calculate loan-to-income ratio.

    Loan-to-Income = Requested Loan Amount / Annual Gross Income

    Units: both in same currency. Ratio is dimensionless.
    """
    if (
        loan_amount is None
        or annual_income is None
        or annual_income <= 0
        or loan_amount < 0
    ):
        return LoanToIncomeCalculation(
            ratio=None,
            loan_amount=loan_amount,
            annual_income=annual_income,
            available=False,
        )
    ratio = loan_amount / annual_income
    return LoanToIncomeCalculation(
        ratio=round(ratio, 4),
        loan_amount=loan_amount,
        annual_income=annual_income,
        available=True,
    )


def calculate_monthly_installment(
    principal: float,
    annual_interest_rate: float,
    term_months: int,
) -> InstallmentCalculation:
    """Calculate the standard amortizing monthly loan payment.

    Formula:
        M = P * (r * (1 + r)^n) / ((1 + r)^n - 1)

    Where:
        P = principal
        r = monthly interest rate (annual rate / 12 / 100)
        n = number of payments (term in months)

    Zero-interest loans use simple division: M = P / n.
    """
    if principal <= 0:
        raise ValueError("Principal must be a positive number")
    if term_months < 1:
        raise ValueError("Term must be at least 1 month")
    if annual_interest_rate < 0:
        raise ValueError("Interest rate cannot be negative")

    zero_interest = annual_interest_rate == 0

    if zero_interest:
        monthly_payment = principal / term_months
    else:
        monthly_rate = annual_interest_rate / 100.0 / 12.0
        factor = (1 + monthly_rate) ** term_months
        monthly_payment = principal * (monthly_rate * factor) / (factor - 1)

    return InstallmentCalculation(
        monthly_payment=round(monthly_payment, 2),
        principal=principal,
        annual_interest_rate=annual_interest_rate,
        term_months=term_months,
        zero_interest=zero_interest,
    )


def calculate_affordability(
    gross_monthly_income: float | None,
    monthly_debt_obligations: float | None,
    proposed_monthly_payment: float | None,
    dti_percent: float | None,
) -> AffordabilityCalculation:
    """Calculate affordability metrics.

    Computes payment-to-income ratio, remaining income, and disposable income.
    Does not assume missing financial information is zero.
    """
    available = (
        gross_monthly_income is not None
        and gross_monthly_income > 0
        and proposed_monthly_payment is not None
    )

    payment_to_income: float | None = None
    remaining_income: float | None = None
    disposable_income: float | None = None

    if available:
        payment_to_income = round(
            (proposed_monthly_payment / gross_monthly_income) * 100, 2
        )
        remaining_income = round(gross_monthly_income - proposed_monthly_payment, 2)

        if monthly_debt_obligations is not None:
            disposable_income = round(
                gross_monthly_income - monthly_debt_obligations - proposed_monthly_payment, 2
            )
        else:
            disposable_income = None

    return AffordabilityCalculation(
        gross_monthly_income=gross_monthly_income,
        monthly_debt_obligations=monthly_debt_obligations,
        proposed_monthly_payment=proposed_monthly_payment,
        payment_to_income_percent=payment_to_income,
        remaining_monthly_income=remaining_income,
        disposable_income=disposable_income,
        dti_percent=dti_percent,
        available=available,
    )


__all__ = [
    "DTICalculation",
    "LoanToIncomeCalculation",
    "InstallmentCalculation",
    "AffordabilityCalculation",
    "calculate_dti",
    "calculate_loan_to_income",
    "calculate_monthly_installment",
    "calculate_affordability",
]
