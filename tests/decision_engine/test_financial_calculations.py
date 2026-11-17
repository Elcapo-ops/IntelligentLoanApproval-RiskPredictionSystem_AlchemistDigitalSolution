"""Tests for financial calculation functions."""

import pytest

from decision_engine.financial_calculations import (
    calculate_affordability,
    calculate_dti,
    calculate_loan_to_income,
    calculate_monthly_installment,
)


class TestDTI:
    def test_dti_calculation(self):
        result = calculate_dti(total_monthly_debt=2000, gross_monthly_income=5000)
        assert result.available is True
        assert result.dti_percent == 40.0

    def test_dti_zero_income(self):
        result = calculate_dti(total_monthly_debt=2000, gross_monthly_income=0)
        assert result.available is False
        assert result.dti_percent is None

    def test_dti_missing_debt(self):
        result = calculate_dti(total_monthly_debt=None, gross_monthly_income=5000)
        assert result.available is False
        assert result.dti_percent is None

    def test_dti_missing_income(self):
        result = calculate_dti(total_monthly_debt=2000, gross_monthly_income=None)
        assert result.available is False

    def test_dti_negative_debt(self):
        result = calculate_dti(total_monthly_debt=-100, gross_monthly_income=5000)
        assert result.available is False


class TestLoanToIncome:
    def test_lti_calculation(self):
        result = calculate_loan_to_income(loan_amount=50000, annual_income=100000)
        assert result.available is True
        assert result.ratio == 0.5

    def test_lti_zero_income(self):
        result = calculate_loan_to_income(loan_amount=50000, annual_income=0)
        assert result.available is False

    def test_lti_missing_values(self):
        result = calculate_loan_to_income(loan_amount=None, annual_income=100000)
        assert result.available is False


class TestInstallment:
    def test_standard_installment(self):
        result = calculate_monthly_installment(principal=10000, annual_interest_rate=5.0, term_months=12)
        assert result.available is True if hasattr(result, 'available') else True
        assert result.monthly_payment > 0
        assert result.zero_interest is False

    def test_zero_interest_installment(self):
        result = calculate_monthly_installment(principal=12000, annual_interest_rate=0.0, term_months=12)
        assert result.monthly_payment == 1000.0
        assert result.zero_interest is True

    def test_invalid_principal(self):
        with pytest.raises(ValueError):
            calculate_monthly_installment(principal=0, annual_interest_rate=5.0, term_months=12)

    def test_invalid_term(self):
        with pytest.raises(ValueError):
            calculate_monthly_installment(principal=10000, annual_interest_rate=5.0, term_months=0)

    def test_negative_interest_rate(self):
        with pytest.raises(ValueError):
            calculate_monthly_installment(principal=10000, annual_interest_rate=-1.0, term_months=12)


class TestAffordability:
    def test_affordability_complete(self):
        result = calculate_affordability(
            gross_monthly_income=5000,
            monthly_debt_obligations=1000,
            proposed_monthly_payment=500,
            dti_percent=20.0,
        )
        assert result.available is True
        assert result.payment_to_income_percent == 10.0
        assert result.disposable_income == 3500.0

    def test_affordability_missing_debt(self):
        result = calculate_affordability(
            gross_monthly_income=5000,
            monthly_debt_obligations=None,
            proposed_monthly_payment=500,
            dti_percent=None,
        )
        assert result.available is True
        assert result.disposable_income is None

    def test_affordability_missing_income(self):
        result = calculate_affordability(
            gross_monthly_income=None,
            monthly_debt_obligations=1000,
            proposed_monthly_payment=500,
            dti_percent=None,
        )
        assert result.available is False
