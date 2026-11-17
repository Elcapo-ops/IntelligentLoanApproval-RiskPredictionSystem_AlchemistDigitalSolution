# Lending Decision Engine

A separate, explainable, rule-based lending decision-support engine that evaluates
loan applications against a configurable lending policy and produces one of three
outcomes: **APPROVE**, **REJECT**, or **REFER FOR MANUAL REVIEW**.

This is an independent feature — it does not use the historical `Loan_Status`
approval classifier or the Loan Default / Credit Risk Prediction module as an
autonomous decision source.

## Architecture

```
src/decision_engine/
  __init__.py        - Package entry point
  engine.py          - Core decision engine with rule evaluation
  rules.py           - Mandatory rejection and manual review rules
  eligibility.py     - Applicant eligibility rules
  affordability.py   - Financial and affordability rules
  risk_integration.py - Optional default-risk model adapter
  explanations.py    - Decision report serialization
  decision_history.py - Audit trail persistence
  schemas.py         - Input/output data schemas
  policy_loader.py   - Load lending policy YAML
  policy_validator.py - Validate policy structure and values

config/
  lending_policy.yaml - Configurable lending policy

app/pages/
  lending_decision_engine.py - Streamlit UI page

tests/decision_engine/
  Unit and integration tests

docs/lending_decision_engine/
  Documentation
```

## Decision Outcomes

| Outcome | Meaning |
|---------|---------|
| APPROVE | All required conditions satisfied |
| REJECT | Mandatory rejection rule failed |
| REFER FOR MANUAL REVIEW | Mandatory review triggered, or information missing |

## Rule Precedence

1. Mandatory rejection — any fail = REJECT
2. Mandatory manual review — any fail = REFER FOR MANUAL REVIEW
3. Eligibility — any fail = REJECT
4. Financial — any fail = REJECT
5. Credit risk — any fail = REJECT
6. Warnings — recorded but do not change the decision
7. APPROVE — only when all required conditions pass

## Default-Risk Model Integration

The engine optionally integrates with the Loan Default / Credit Risk Prediction
module. If the default-risk model is available, the estimated default probability
is evaluated against `credit_risk.maximum_default_probability` in the policy.
If the model is unavailable, the engine still operates using rule-based policy
and indicates that model-based risk assessment was not performed.

## Disclaimer

This is a **decision-support prototype**, not an autonomous lending engine.
All policy thresholds are illustrative development values and must be replaced
with your institution's approved lending policy before production use. The system
must not claim regulatory compliance without a jurisdiction-specific legal and
compliance review.

## Running

```powershell
# Ensure virtual environment is active
.\venv\Scripts\Activate.ps1

# Run the Streamlit application
streamlit run app.py

# Navigate to "Lending Decision Engine" in the sidebar
```

## Running Tests

```powershell
pytest tests/decision_engine/ -v
```
