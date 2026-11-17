"""Lending decision engine package.

This module implements a separate, explainable, rule-based lending decision-support
engine. It is intentionally isolated from the historical `Loan_Status` approval
classifier and the Loan Default / Credit Risk Prediction module.

The engine evaluates loan applications against a configurable lending policy and
produces one of three outcomes: APPROVE, REJECT, or REFER FOR MANUAL REVIEW.
"""

from decision_engine.engine import DecisionEngine, DecisionOutcome
from decision_engine.schemas import ApplicationInput, DecisionReport
from decision_engine.policy_loader import load_lending_policy
from decision_engine.policy_validator import validate_lending_policy

__all__ = [
    "DecisionEngine",
    "DecisionOutcome",
    "ApplicationInput",
    "DecisionReport",
    "load_lending_policy",
    "validate_lending_policy",
]
