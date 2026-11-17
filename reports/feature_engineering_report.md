# Phase 3.2 — Feature engineering

## Features and formulas

`LoanFeatureEngineer` in `src/features/engineering.py` runs as the first step
of `build_model_pipeline`, before the fold-fitted `ColumnTransformer`. It does
not inspect or calculate from `Loan_Status`; the target is kept separate as
the estimator's `y`. Configured identifiers and the target are explicitly
dropped if present in input. The deterministic transformer learns no
statistics from rows, so training, cross-validation, holdout inference, and
application inference use the same formulas.

| Output feature | Formula / transformation |
|---|---|
| `Total_Income` | `ApplicantIncome + CoapplicantIncome` |
| `Loan_to_Income_Ratio` | `LoanAmount / Total_Income`, only when total income is positive and loan amount is nonnegative |
| `*_log1p` | `ln(1 + x)` for configured, nonnegative input values |

`Loan_to_Income_Ratio` is a loan-amount-to-total-income ratio, **not debt-to-
income (DTI)**. The source data does not provide other debt obligations, so
this feature must not be described as actual DTI.

The configured log1p features are `ApplicantIncome`, `CoapplicantIncome`,
`LoanAmount`, and `Total_Income`. Set `feature_engineering.log1p_columns` to
an empty list to disable optional log features. The original features remain
available to the estimator alongside engineered features.

## Missing and extreme values

- Numeric strings are parsed as numeric; invalid strings, missing values, and
  positive/negative infinity in numeric inputs are represented as `NaN`.
- `Total_Income` remains missing if either source income is missing. Overflow
  or another non-finite sum is converted to `NaN`.
- A zero, negative, or missing total income yields a missing ratio; negative
  loan amounts also yield a missing ratio. Division overflow/non-finite
  results become `NaN`.
- Negative values are outside the log1p domain and produce `NaN` for that log
  feature. Zero is valid and maps to zero. Non-finite log results become
  `NaN`.
- Finite extreme source values are preserved rather than clipped with an
  arbitrary bound. `log1p` provides a scale-compressed feature where
  configured. Overflow in derived arithmetic is made missing and then handled
  by the existing median imputer, whose statistics are fitted inside each
  training fold.
- The transformer does not impute values or learn bounds. Median imputation
  and scaling remain subsequent pipeline steps and therefore are fit only on
  the data provided to each pipeline fit.

## Verification

Added edge-case tests for formula values, zero denominators, missing inputs,
invalid numeric strings, infinities, extreme-value overflow, negative
logarithm inputs, consistent training/inference transforms, target
independence, and pipeline integration. The feature-engineering and
preprocessing test modules passed together: **14 passed**. The complete
project suite passed: **58 passed**; `pip check` reported no broken
requirements.
