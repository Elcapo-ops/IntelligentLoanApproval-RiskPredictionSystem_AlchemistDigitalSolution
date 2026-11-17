# Decision Logic

The lending decision engine uses a deterministic, explainable rule engine that
evaluates applications against a configurable lending policy.

## Decision Outcomes

| Outcome | When It Occurs |
|---------|---------------|
| **APPROVE** | All required conditions pass; no rejection or review triggers |
| **REJECT** | Any mandatory rejection rule fails; or eligibility/financial/risk rule fails |
| **REFER FOR MANUAL REVIEW** | Any mandatory manual review rule fails; or required information is missing |

## Rule Precedence (Deterministic)

1. **Mandatory rejection** — any fail → REJECT immediately
2. **Mandatory manual review** — any fail → REFER FOR MANUAL REVIEW immediately
3. **Eligibility rules** — any fail → REJECT
4. **Financial rules** — any fail → REJECT
5. **Credit risk rules** — any fail → REJECT
6. **Warning rules** — recorded but do not change the decision
7. **APPROVE** — only if all required conditions pass

## Rule Categories

| Category | Purpose | Effect on Decision |
|----------|---------|-------------------|
| mandatory_rejection | Explicit policy violations | REJECT on fail |
| mandatory_review | Missing info or model unavailable | REFER on fail |
| eligibility | Applicant eligibility | REJECT on fail |
| financial | Income, DTI, affordability | REJECT on fail |
| risk | Default-risk model thresholds | REJECT on fail |
| warning | Concerns, not blocking | Recorded only |

## Default-Risk Model Integration

When available, the engine calls the default-risk model and:
- Compares estimated default probability against `credit_risk.maximum_default_probability`
- Checks if the risk category is in `credit_risk.risk_categories_restricted`
- If the model is unavailable and `manual_review_if_model_unavailable` is true → REFER

If the model is unavailable and review is not required, the decision proceeds
using rule-based policy only. Missing risk predictions are never treated as low risk.

## Rule Result Structure

Every rule produces:
- Rule ID
- Rule name
- Category
- Input value
- Configured threshold
- Result (pass/fail/not_applicable)
- Explanation
- Contribution (reject/refer/approve/none)

## Example Decision Flow

1. Applicant age 17 (< minimum 18) → ELIG-001 fail → **REJECT**
2. Missing monthly income → REV-004 fail → **REFER FOR MANUAL REVIEW**
3. DTI 45% (> maximum 40%) → FIN-003 fail → **REJECT**
4. All rules pass → **APPROVE**
