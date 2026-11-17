# Limitations

This lending decision engine is an academic decision-support prototype. It is
not a production lending system and must not be used for actual lending
decisions without thorough review.

## Regulatory

- The system does not claim compliance with any lending regulation (ECOA, FCRA,
  fair lending laws, etc.).
- All policy thresholds are illustrative development values, not regulatory
  requirements.
- A jurisdiction-specific legal and compliance review is required before
  production use.

## Fairness

- Protected characteristics are not used as automatic rejection criteria.
- However, proxy variables may indirectly correlate with protected classes.
- No formal fairness testing or disparate impact analysis has been performed.
- Historical data limitations may introduce bias.

## Data

- No real applicant data is used in development.
- Synthetic test data is labeled as such and is not performance evidence.
- Missing information is never treated as zero — it is flagged as unavailable.

## Model

- Default-risk model integration is optional and uses historical LendingClub
  data with censored outcomes.
- Model predictions are statistical estimates, not guarantees.
- Missing or unavailable model predictions are never treated as low risk.

## Security

- No authentication or authorization is implemented for policy modification.
- Policy modification is development-only; do not expose unrestricted in
  production.
- Decision history stores limited application data but no raw sensitive
  information.

## Statement

The decision engine is a **decision-support prototype** until its policies,
data, validation, security, and legal requirements have been reviewed and
approved by the relevant institution.
