# Exploratory data analysis

**Dataset:** [Kaggle Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)  
**File analyzed:** `data\raw\train_u6lujuX_CVtuZ9i.csv` (all 614 labeled source rows; not the later model-training partition)  
**Scope:** 614 rows, 13 columns. The separate unlabeled Kaggle test CSV was not loaded. No holdout partition existed yet, and no preprocessing or model selection was performed.

**Retrospective holdout disclosure:** the notebook analyzed all 614 labeled
rows before the later seed-42 split was created. The 123 observations later
assigned to that holdout were therefore included in these descriptive
statistics and target-conditioned charts. No fitted transformations or model
selection were performed in EDA, but the holdout is not fully untouched or an
independent confirmation set. See [EDA business insights](eda_insights.md) and
the [final audit](final_audit_report.md).

## Target distribution

| Loan status | Applications | Share |
|---|---|---|
| N (rejected) | 192 | 31.27% |
| Y (approved) | 422 | 68.73% |

## Numerical descriptive statistics

Statistics use the observed values; missing entries are excluded per field. Income and loan values are strongly right-skewed in this sample, so means and maxima should be read alongside medians and quartiles. Distribution plots retain all observed values.

| field | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| ApplicantIncome | 614.0 | 5403.459 | 6109.042 | 150.0 | 2877.5 | 3812.5 | 5795.0 | 81000.0 |
| CoapplicantIncome | 614.0 | 1621.246 | 2926.248 | 0.0 | 0.0 | 1188.5 | 2297.25 | 41667.0 |
| LoanAmount | 592.0 | 146.412 | 85.587 | 9.0 | 100.0 | 128.0 | 168.0 | 700.0 |
| Loan_Amount_Term | 600.0 | 342.0 | 65.12 | 12.0 | 360.0 | 360.0 | 360.0 | 480.0 |
| Credit_History | 564.0 | 0.842 | 0.365 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |

## Categorical distributions

Missing values are represented as `(Missing)` in this table and in categorical plots; the source frame is unchanged.

| Field | Observed category | Applications | Share of rows |
|---|---|---|---|
| Gender | Male | 489 | 79.64% |
| Gender | Female | 112 | 18.24% |
| Gender | (Missing) | 13 | 2.12% |
| Married | Yes | 398 | 64.82% |
| Married | No | 213 | 34.69% |
| Married | (Missing) | 3 | 0.49% |
| Dependents | 0 | 345 | 56.19% |
| Dependents | 1 | 102 | 16.61% |
| Dependents | 2 | 101 | 16.45% |
| Dependents | 3+ | 51 | 8.31% |
| Dependents | (Missing) | 15 | 2.44% |
| Education | Graduate | 480 | 78.18% |
| Education | Not Graduate | 134 | 21.82% |
| Self_Employed | No | 500 | 81.43% |
| Self_Employed | Yes | 82 | 13.36% |
| Self_Employed | (Missing) | 32 | 5.21% |
| Credit_History | 1.0 | 475 | 77.36% |
| Credit_History | 0.0 | 89 | 14.50% |
| Credit_History | (Missing) | 50 | 8.14% |
| Property_Area | Semiurban | 233 | 37.95% |
| Property_Area | Urban | 202 | 32.90% |
| Property_Area | Rural | 179 | 29.15% |

## Missing value patterns

| Field | Missing | Missing share |
|---|---|---|
| Gender | 13 | 2.12% |
| Married | 3 | 0.49% |
| Dependents | 15 | 2.44% |
| Self_Employed | 32 | 5.21% |
| LoanAmount | 22 | 3.58% |
| Loan_Amount_Term | 14 | 2.28% |
| Credit_History | 50 | 8.14% |

Credit history is observed for 564 of 614 labeled applications. Employment status (`Self_Employed`) is missing for 32 rows. The missingness plot reports observed nulls; no missing values were filled.

## Approval patterns

Approval-rate figures show the observed share with `Loan_Status = Y` by category, including missing categories where applicable. These are descriptive associations, not causal effects. The correlation heatmap uses numeric columns only and pairwise available values; correlations are not predictive performance or causal evidence.

## Figures

- [Loan approval versus rejection](figures/01_approval_vs_rejection.png)
- [Approval rate by credit history](figures/02_approval_by_credit_history.png)
- [Income, loan amount, and term distributions](figures/03_income_loan_distributions.png)
- [Approval rate by employment status](figures/04_approval_by_employment.png)
- [Numerical correlation heatmap](figures/05_numerical_correlation.png)
- [Approval rate by property area](figures/06_approval_by_property_area.png)
- [Categorical distributions](figures/07_categorical_distributions.png)
- [Missing value counts](figures/08_missing_values.png)

## Interpretation and limitations

- EDA used the labeled Kaggle source CSV only; the separate unlabeled Kaggle CSV was excluded. All 614 labeled rows were analyzed before the later split, so the 123 rows subsequently assigned to holdout were included in EDA. See the retrospective holdout disclosure above.
- No imputation, filtering, outlier deletion, resampling, or source modification was performed.
- Long-tailed income and loan observations are shown as observed. Extreme values may be unusual but valid applicants; visual/statistical extremes alone do not establish data errors.
- Category-level approval differences are descriptive and may reflect confounding, missingness, or sample size. They are not lending rules.
- The dataset is small and contains missing values; findings may not generalize to other populations or current lending conditions.
