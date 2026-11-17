"""Tests for the lending decision engine."""

import pytest

from decision_engine.engine import DecisionEngine
from decision_engine.policy_loader import load_lending_policy


@pytest.fixture
def engine():
    policy = load_lending_policy()
    return DecisionEngine(policy=policy)


def test_engine_initialization(engine):
    assert engine is not None
    assert engine.policy_version is not None


def test_approve_scenario(engine):
    application = {
        "application_id": "APP-APPROVE-001",
        "applicant_age": 35,
        "annual_income": 60000,
        "monthly_income": 5000,
        "monthly_debt_obligations": 500,
        "requested_loan_amount": 15000,
        "loan_term_months": 36,
        "interest_rate": 8.5,
        "employment_length_months": 24,
        "home_ownership": "RENT",
        "verification_status": "Verified",
        "loan_purpose": "debt_consolidation",
        "residency_state": "CA",
    }
    report = engine.evaluate(application)
    assert report.decision in ("APPROVE", "REFER FOR MANUAL REVIEW")


def test_reject_scenario(engine):
    application = {
        "application_id": "APP-REJECT-001",
        "applicant_age": 17,
        "annual_income": 60000,
        "monthly_income": 5000,
        "monthly_debt_obligations": 500,
        "requested_loan_amount": 15000,
        "loan_term_months": 36,
        "interest_rate": 8.5,
        "employment_length_months": 24,
        "home_ownership": "RENT",
        "verification_status": "Verified",
        "loan_purpose": "debt_consolidation",
        "residency_state": "CA",
    }
    report = engine.evaluate(application)
    # Under 18 minimum age should be rejection
    assert report.decision == "REJECT" or any(
        r.contribution == "reject" for r in report.rules if r.result == "fail"
    )


def test_refer_for_manual_review(engine):
    application = {
        "application_id": "APP-REFER-001",
        "applicant_age": 35,
        "annual_income": None,
        "monthly_income": None,
        "monthly_debt_obligations": None,
        "requested_loan_amount": 15000,
        "loan_term_months": 36,
        "interest_rate": 8.5,
        "employment_length_months": 24,
        "home_ownership": "RENT",
        "verification_status": None,
        "loan_purpose": "debt_consolidation",
        "residency_state": "CA",
    }
    report = engine.evaluate(application)
    assert report.decision in ("REFER FOR MANUAL REVIEW", "REJECT")


def test_report_structure(engine):
    application = {
        "application_id": "APP-STRUCT-001",
        "applicant_age": 35,
        "annual_income": 60000,
        "monthly_income": 5000,
        "monthly_debt_obligations": 500,
        "requested_loan_amount": 15000,
        "loan_term_months": 36,
        "interest_rate": 8.5,
        "employment_length_months": 24,
        "home_ownership": "RENT",
        "verification_status": "Verified",
        "loan_purpose": "debt_consolidation",
        "residency_state": "CA",
    }
    report = engine.evaluate(application)
    assert report.application_id == "APP-STRUCT-001"
    assert report.policy_version == engine.policy_version
    assert report.decision in ("APPROVE", "REJECT", "REFER FOR MANUAL REVIEW")
    assert isinstance(report.rules, list)
    assert isinstance(report.passed_rules, list)
    assert isinstance(report.failed_rules, list)
    assert isinstance(report.warnings, list)
    assert report.decision_explanation is not None
