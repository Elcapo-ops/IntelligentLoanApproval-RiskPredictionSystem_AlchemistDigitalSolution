"""Validate lending policy structure and values before use."""

from __future__ import annotations

from typing import Any

from utils.exceptions import ConfigurationError

REQUIRED_TOP_LEVEL_KEYS = {
    "policy_version",
    "policy_status",
    "eligibility",
    "financial",
    "credit_risk",
    "loan",
}

REQUIRED_ELIGIBILITY_KEYS = {
    "minimum_age",
    "required_application_fields",
    "minimum_employment_months",
    "required_home_ownership",
    "required_verification_status",
    "eligible_loan_purposes",
}

REQUIRED_FINANCIAL_KEYS = {
    "minimum_verified_monthly_income",
    "maximum_loan_amount",
    "maximum_loan_to_income",
    "maximum_payment_to_income_percent",
    "maximum_dti_percent",
    "minimum_disposable_income",
}

REQUIRED_CREDIT_RISK_KEYS = {
    "maximum_default_probability",
    "risk_categories_restricted",
    "manual_review_if_model_unavailable",
    "manual_review_if_probability_missing",
}

REQUIRED_LOAN_KEYS = {
    "minimum_term_months",
    "maximum_term_months",
    "minimum_loan_amount",
    "maximum_interest_rate",
    "minimum_interest_rate",
}

NON_NEGATIVE_FINANCIAL_KEYS = {
    "minimum_verified_monthly_income",
    "maximum_loan_amount",
    "maximum_loan_to_income",
    "maximum_payment_to_income_percent",
    "maximum_dti_percent",
    "minimum_disposable_income",
}


def validate_lending_policy(policy: dict[str, Any]) -> None:
    """Validate the structure and values of a lending policy.

    Args:
        policy: Parsed policy dictionary from load_lending_policy.

    Raises:
        ConfigurationError: If any required section, key, or value is invalid.
    """
    if not isinstance(policy, dict):
        raise ConfigurationError("Policy must be a mapping")

    missing_top = REQUIRED_TOP_LEVEL_KEYS.difference(policy)
    if missing_top:
        raise ConfigurationError(f"Policy missing required sections: {sorted(missing_top)}")

    version = policy.get("policy_version")
    if not isinstance(version, str) or not version.strip():
        raise ConfigurationError("policy_version must be a non-empty string")

    eligibility = policy.get("eligibility")
    if not isinstance(eligibility, dict):
        raise ConfigurationError("Policy section 'eligibility' must be a mapping")
    missing_elig = REQUIRED_ELIGIBILITY_KEYS.difference(eligibility)
    if missing_elig:
        raise ConfigurationError(f"Eligibility missing required keys: {sorted(missing_elig)}")

    min_age = eligibility.get("minimum_age")
    if not isinstance(min_age, int) or isinstance(min_age, bool) or min_age < 0:
        raise ConfigurationError("eligibility.minimum_age must be a non-negative integer")

    req_fields = eligibility.get("required_application_fields")
    if not isinstance(req_fields, list) or not req_fields:
        raise ConfigurationError("eligibility.required_application_fields must be a non-empty list")

    min_emp = eligibility.get("minimum_employment_months")
    if not isinstance(min_emp, int) or isinstance(min_emp, bool) or min_emp < 0:
        raise ConfigurationError("eligibility.minimum_employment_months must be a non-negative integer")

    for bool_key in ("required_home_ownership", "required_verification_status"):
        if not isinstance(eligibility.get(bool_key), bool):
            raise ConfigurationError(f"eligibility.{bool_key} must be a boolean")

    purposes = eligibility.get("eligible_loan_purposes")
    if not isinstance(purposes, list) or not purposes:
        raise ConfigurationError("eligibility.eligible_loan_purposes must be a non-empty list")

    financial = policy.get("financial")
    if not isinstance(financial, dict):
        raise ConfigurationError("Policy section 'financial' must be a mapping")
    missing_fin = REQUIRED_FINANCIAL_KEYS.difference(financial)
    if missing_fin:
        raise ConfigurationError(f"Financial missing required keys: {sorted(missing_fin)}")

    for key in NON_NEGATIVE_FINANCIAL_KEYS:
        value = financial.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ConfigurationError(f"financial.{key} must be a non-negative number")

    if financial["maximum_loan_to_income"] <= 0:
        raise ConfigurationError("financial.maximum_loan_to_income must be greater than 0")
    if financial["maximum_payment_to_income_percent"] <= 0:
        raise ConfigurationError("financial.maximum_payment_to_income_percent must be greater than 0")
    if financial["maximum_dti_percent"] <= 0:
        raise ConfigurationError("financial.maximum_dti_percent must be greater than 0")

    credit_risk = policy.get("credit_risk")
    if not isinstance(credit_risk, dict):
        raise ConfigurationError("Policy section 'credit_risk' must be a mapping")
    missing_cr = REQUIRED_CREDIT_RISK_KEYS.difference(credit_risk)
    if missing_cr:
        raise ConfigurationError(f"Credit risk missing required keys: {sorted(missing_cr)}")

    max_prob = credit_risk.get("maximum_default_probability")
    if not isinstance(max_prob, (int, float)) or isinstance(max_prob, bool) or not 0 <= max_prob <= 1:
        raise ConfigurationError("credit_risk.maximum_default_probability must be between 0 and 1")

    restricted = credit_risk.get("risk_categories_restricted")
    if not isinstance(restricted, list):
        raise ConfigurationError("credit_risk.risk_categories_restricted must be a list")

    for bool_key in ("manual_review_if_model_unavailable", "manual_review_if_probability_missing"):
        if not isinstance(credit_risk.get(bool_key), bool):
            raise ConfigurationError(f"credit_risk.{bool_key} must be a boolean")

    loan = policy.get("loan")
    if not isinstance(loan, dict):
        raise ConfigurationError("Policy section 'loan' must be a mapping")
    missing_loan = REQUIRED_LOAN_KEYS.difference(loan)
    if missing_loan:
        raise ConfigurationError(f"Loan missing required keys: {sorted(missing_loan)}")

    for int_key in ("minimum_term_months", "maximum_term_months"):
        value = loan.get(int_key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ConfigurationError(f"loan.{int_key} must be a positive integer")

    if loan["minimum_term_months"] > loan["maximum_term_months"]:
        raise ConfigurationError("loan.minimum_term_months must not exceed loan.maximum_term_months")

    for rate_key in ("minimum_interest_rate", "maximum_interest_rate"):
        value = loan.get(rate_key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ConfigurationError(f"loan.{rate_key} must be a non-negative number")

    if loan["minimum_interest_rate"] > loan["maximum_interest_rate"]:
        raise ConfigurationError("loan.minimum_interest_rate must not exceed loan.maximum_interest_rate")

    min_amount = loan.get("minimum_loan_amount")
    if not isinstance(min_amount, (int, float)) or isinstance(min_amount, bool) or min_amount < 0:
        raise ConfigurationError("loan.minimum_loan_amount must be a non-negative number")
