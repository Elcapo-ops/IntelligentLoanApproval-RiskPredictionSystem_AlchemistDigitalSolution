# Lending Policy Reference

All policy values are configured in `config/lending_policy.yaml` and can be
changed without editing any Python source code.

## Policy Sections

### eligibility — Applicant Eligibility Rules

| Key | Type | Description | Illustrative Default |
|-----|------|-------------|---------------------|
| minimum_age | int | Minimum applicant age | 18 |
| required_application_fields | list[str] | Fields that must be present | ["annual_income", "requested_loan_amount", "loan_term_months"] |
| minimum_employment_months | int | Minimum months at current employer | 6 |
| required_home_ownership | bool | Whether home ownership status is required | true |
| required_verification_status | bool | Whether income verification is required | true |
| eligible_loan_purposes | list[str] | Allowed loan purposes | ["debt_consolidation", "home_improvement", ...] |
| eligible_residency_states | list[str]\|null | Allowed states; null means all | null |

### financial — Financial Rules

| Key | Type | Description | Illustrative Default |
|-----|------|-------------|---------------------|
| minimum_verified_monthly_income | float | Minimum monthly income | 2000.0 |
| maximum_loan_amount | float | Absolute maximum loan amount | 100000.0 |
| maximum_loan_to_income | float | Max loan amount / annual income | 5.0 |
| maximum_payment_to_income_percent | float | Max payment / monthly income × 100 | 28.0 |
| maximum_dti_percent | float | Max debt-to-income × 100 | 40.0 |
| minimum_disposable_income | float | Minimum income after debts and payment | 500.0 |

### credit_risk — Credit Risk Rules

| Key | Type | Description | Illustrative Default |
|-----|------|-------------|---------------------|
| maximum_default_probability | float | Max estimated default probability (0–1) | 0.25 |
| risk_categories_restricted | list[str] | Risk categories requiring rejection | ["Very High Risk"] |
| manual_review_if_model_unavailable | bool | Refer if model can't load | true |
| manual_review_if_probability_missing | bool | Refer if probability is missing | true |

### loan — Loan-Specific Rules

| Key | Type | Description | Illustrative Default |
|-----|------|-------------|---------------------|
| minimum_term_months | int | Minimum loan term | 12 |
| maximum_term_months | int | Maximum loan term | 84 |
| minimum_loan_amount | float | Minimum loan amount | 1000.0 |
| maximum_interest_rate | float | Max annual interest rate (%) | 36.0 |
| minimum_interest_rate | float | Min annual interest rate (%) | 0.0 |

## Important Notice

**All values above are illustrative development defaults.** They are NOT
banking standards, regulatory requirements, or production values. Replace every
threshold with your institution's approved lending policy before production use.

## Policy Versioning

- `policy_version` must be updated whenever any value changes
- `policy_status` indicates development/production state
- `last_updated` and `change_summary` track modifications
- Every decision record stores the `policy_version` used
