# Data quality audit

- Source: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
- Target: `Loan_Status`
- Identifiers: `Loan_ID`
- Method: read-only audit; source CSVs were not modified.

## Dataset integrity

| Dataset | Rows | Columns | Duplicate rows | Duplicate identifier rows |
|---|---:|---:|---:|---:|
| Labeled | 614 | 13 | 0 | 0 |
| Unlabeled | 367 | 12 | 0 | 0 |

Cross-file identifier overlap: **0** across configured IDs.
Feature schemas match: **yes**.

## Missing values

Percentages use each dataset's own row count.

| Dataset | Column | Missing | Missing % |
|---|---|---:|---:|
| Labeled | `Loan_ID` | 0 | 0.0000% |
| Labeled | `Gender` | 13 | 2.1173% |
| Labeled | `Married` | 3 | 0.4886% |
| Labeled | `Dependents` | 15 | 2.4430% |
| Labeled | `Education` | 0 | 0.0000% |
| Labeled | `Self_Employed` | 32 | 5.2117% |
| Labeled | `ApplicantIncome` | 0 | 0.0000% |
| Labeled | `CoapplicantIncome` | 0 | 0.0000% |
| Labeled | `LoanAmount` | 22 | 3.5831% |
| Labeled | `Loan_Amount_Term` | 14 | 2.2801% |
| Labeled | `Credit_History` | 50 | 8.1433% |
| Labeled | `Property_Area` | 0 | 0.0000% |
| Labeled | `Loan_Status` | 0 | 0.0000% |
| Unlabeled | `Loan_ID` | 0 | 0.0000% |
| Unlabeled | `Gender` | 11 | 2.9973% |
| Unlabeled | `Married` | 0 | 0.0000% |
| Unlabeled | `Dependents` | 10 | 2.7248% |
| Unlabeled | `Education` | 0 | 0.0000% |
| Unlabeled | `Self_Employed` | 23 | 6.2670% |
| Unlabeled | `ApplicantIncome` | 0 | 0.0000% |
| Unlabeled | `CoapplicantIncome` | 0 | 0.0000% |
| Unlabeled | `LoanAmount` | 5 | 1.3624% |
| Unlabeled | `Loan_Amount_Term` | 6 | 1.6349% |
| Unlabeled | `Credit_History` | 29 | 7.9019% |
| Unlabeled | `Property_Area` | 0 | 0.0000% |

## Category and target validation

| Dataset | Field | Invalid values / result |
|---|---|---|
| Labeled | `Gender` | none |
| Labeled | `Married` | none |
| Labeled | `Dependents` | none |
| Labeled | `Education` | none |
| Labeled | `Self_Employed` | none |
| Labeled | `Credit_History` | none |
| Labeled | `Property_Area` | none |
| Labeled | `Loan_Status` | 0 invalid, 0 missing |
| Unlabeled | `Gender` | none |
| Unlabeled | `Married` | none |
| Unlabeled | `Dependents` | none |
| Unlabeled | `Education` | none |
| Unlabeled | `Self_Employed` | none |
| Unlabeled | `Credit_History` | none |
| Unlabeled | `Property_Area` | none |

### Labeled target distribution

| Label | Count | Proportion |
|---|---:|---:|
| `Y` | 422 | 68.7296% |
| `N` | 192 | 31.2704% |

Majority/minority ratio: **2.1979:1**. This indicates class imbalance, not invalid labels. No resampling was performed. The unlabeled dataset has no target.

## Numerical checks and extreme observations

Negative values are counted as domain violations for income, loan amount, and term. Loan amount and term must also be positive; zero income is retained and reported as unusual rather than automatically invalid. IQR fences are descriptive screening flags only. They are not error criteria.

| Dataset | Field | Min | Q1 | Median | Q3 | Max | Negative | Zero | Non-positive | Non-numeric | Non-finite | IQR fences | IQR flags |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Labeled | `ApplicantIncome` | 150.0 | 2877.5 | 3812.5 | 5795.0 | 81000.0 | 0 | 0 | 0 | 0 | 0 | -1498.75 to 10171.25 | 50 (8.14%) |
| Labeled | `CoapplicantIncome` | 0.0 | 0.0 | 1188.5 | 2297.25 | 41667.0 | 0 | 273 | 0 | 0 | 0 | -3445.88 to 5743.12 | 18 (2.93%) |
| Labeled | `LoanAmount` | 9.0 | 100.0 | 128.0 | 168.0 | 700.0 | 0 | 0 | 0 | 0 | 0 | -2.00 to 270.00 | 39 (6.59%) |
| Labeled | `Loan_Amount_Term` | 12.0 | 360.0 | 360.0 | 360.0 | 480.0 | 0 | 0 | 0 | 0 | 0 | 360.00 to 360.00 | 88 (14.67%) |
| Unlabeled | `ApplicantIncome` | 0.0 | 2864.0 | 3786.0 | 5060.0 | 72529.0 | 0 | 2 | 0 | 0 | 0 | -430.00 to 8354.00 | 32 (8.72%) |
| Unlabeled | `CoapplicantIncome` | 0.0 | 0.0 | 1025.0 | 2430.5 | 24000.0 | 0 | 156 | 0 | 0 | 0 | -3645.75 to 6076.25 | 8 (2.18%) |
| Unlabeled | `LoanAmount` | 28.0 | 100.25 | 125.0 | 158.0 | 550.0 | 0 | 0 | 0 | 0 | 0 | 13.62 to 244.62 | 18 (4.97%) |
| Unlabeled | `Loan_Amount_Term` | 6.0 | 360.0 | 360.0 | 360.0 | 480.0 | 0 | 0 | 0 | 0 | 0 | 360.00 to 360.00 | 50 (13.85%) |

## Interpretation and handling

- Missing values are present and should be handled in a later training-fold preprocessing pipeline; they have not been imputed.
- Any IQR-flagged high incomes or loan amounts may represent unusual but valid applicants. The IQR rule alone cannot establish a data-entry error. Review source context before any exclusion.
- The loan-term quartiles are all 360 months in both files, so the IQR fences collapse to 360. Other observed positive terms are unusual under this rule but are not automatically invalid.
- A zero co-applicant income is valid when no co-applicant income is reported. Zero applicant income is flagged for review, not rejected by this audit.
- A negative income, loan amount, or loan term is a domain violation. No negative values were observed in the acquired files.
- Target labels are checked against the configured `Y`/`N` domain. The unlabeled CSV is checked to ensure the target is absent.
- The two CSVs remain separate. This report does not train a model, impute values, delete records, or rewrite either source.
- No observations were removed, imputed, or changed.
