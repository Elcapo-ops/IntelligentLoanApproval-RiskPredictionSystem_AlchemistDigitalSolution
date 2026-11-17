# Data dictionary

## Dataset context

Source: [Kaggle Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset).
Measured acquisition has 614 labeled rows and 367 unlabeled rows. The
unlabeled CSV has no target and is never concatenated with labeled data for
training. Both CSVs and their license terms should be handled according to
Kaggle's current conditions; the project does not redistribute the files.

Pandas dtypes below are observed when reading the acquired files, not forced
canonical types. `Loan_ID` is excluded from all model features. Percentages
and missing counts refer to the labeled 614-row file unless the table says
otherwise.

## Source fields

| Field | Labeled dtype | Unlabeled dtype | Missing (labeled) | Model use / meaning |
|---|---|---|---:|---|
| `Loan_ID` | string/object | string/object | 0 | Record identifier; excluded from modeling and applicant form |
| `Gender` | string/object | string/object | 13 (2.12%) | Categorical applicant attribute |
| `Married` | string/object | string/object | 3 (0.49%) | Categorical marital-status field |
| `Dependents` | string/object | string/object | 15 (2.44%) | Categorical count bucket; values include `0`, `1`, `2`, `3+` |
| `Education` | string/object | string/object | 0 | `Graduate` / `Not Graduate` category |
| `Self_Employed` | string/object | string/object | 32 (5.21%) | Categorical source indicator |
| `ApplicantIncome` | int64 | int64 | 0 | Applicant income; nonnegative numeric input |
| `CoapplicantIncome` | float64 | int64 | 0 | Co-applicant income; nonnegative numeric input |
| `LoanAmount` | float64 | float64 | 22 (3.58%) | Requested loan amount; configured as positive when provided |
| `Loan_Amount_Term` | float64 | float64 | 14 (2.28%) | Loan term in months; configured as positive when provided |
| `Credit_History` | float64 | float64 | 50 (8.14%) | Categorical model field, configured values 0 or 1 |
| `Property_Area` | string/object | string/object | 0 | `Rural`, `Semiurban`, or `Urban` |
| `Loan_Status` | string/object | Not present | 0 | Labeled target: `Y` approval, `N` rejection; never a model feature |

The unlabeled CSV has 11 missing `Gender`, 10 `Dependents`, 23
`Self_Employed`, 5 `LoanAmount`, 6 `Loan_Amount_Term`, and 29
`Credit_History` values. It has no target distribution. Full observed
profiles are saved in `artifacts/reports/dataset_profile.json`.

## Configured categories

| Field | Accepted source/form values |
|---|---|
| `Gender` | `Female`, `Male` |
| `Married` | `No`, `Yes` |
| `Dependents` | `0`, `1`, `2`, `3+` |
| `Education` | `Graduate`, `Not Graduate` |
| `Self_Employed` | `No`, `Yes` |
| `Credit_History` | `0`, `1` |
| `Property_Area` | `Rural`, `Semiurban`, `Urban` |

The application accepts missing optional fields and applies the fitted
pipeline's imputer. The prediction service validates provided categorical
values against this configured domain.

## Derived pipeline features

| Derived field | Definition | Notes |
|---|---|---|
| `Total_Income` | `ApplicantIncome + CoapplicantIncome` | Missing if an input is missing or the sum is non-finite |
| `Loan_to_Income_Ratio` | `LoanAmount / Total_Income` for positive total income | Not actual debt-to-income; source does not include other debts |
| `ApplicantIncome_log1p` | `ln(1 + ApplicantIncome)` when nonnegative | Configured optional log feature |
| `CoapplicantIncome_log1p` | `ln(1 + CoapplicantIncome)` when nonnegative | Configured optional log feature |
| `LoanAmount_log1p` | `ln(1 + LoanAmount)` when nonnegative | Configured optional log feature |
| `Total_Income_log1p` | `ln(1 + Total_Income)` when nonnegative | Configured optional log feature |

Invalid/non-finite derived results are represented as missing and then handled
by the numeric imputer fitted within training folds. No target-derived field
is engineered.

## Modeling partition

The configured 20% stratified holdout split with random seed 42 yields 491
training rows and 123 holdout rows from 614 labeled applications. It is
performed before learned imputation, scaling, or encoding. All 11 configured
prediction-time fields are present in both partitions; ID and target are not.

