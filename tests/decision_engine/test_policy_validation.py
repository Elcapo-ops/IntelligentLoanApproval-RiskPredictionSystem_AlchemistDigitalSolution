"""Tests for lending policy loading and validation."""

import pytest

from decision_engine.policy_loader import load_lending_policy
from decision_engine.policy_validator import validate_lending_policy


def test_load_default_policy():
    policy = load_lending_policy()
    assert policy is not None
    assert "policy_version" in policy
    assert "eligibility" in policy
    assert "financial" in policy
    assert "credit_risk" in policy
    assert "loan" in policy


def test_validate_valid_policy():
    policy = load_lending_policy()
    validate_lending_policy(policy)  # should not raise


def test_validate_missing_section():
    policy = load_lending_policy()
    del policy["credit_risk"]
    with pytest.raises(Exception):
        validate_lending_policy(policy)


def test_validate_negative_threshold():
    policy = load_lending_policy()
    policy["financial"]["maximum_dti_percent"] = -1
    with pytest.raises(Exception):
        validate_lending_policy(policy)


def test_validate_probability_out_of_range():
    policy = load_lending_policy()
    policy["credit_risk"]["maximum_default_probability"] = 1.5
    with pytest.raises(Exception):
        validate_lending_policy(policy)


def test_validate_term_range_inverted():
    policy = load_lending_policy()
    policy["loan"]["minimum_term_months"] = 84
    policy["loan"]["maximum_term_months"] = 12
    with pytest.raises(Exception):
        validate_lending_policy(policy)
