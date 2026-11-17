# Fairness and robustness evaluation

## Scope

This project models historical `Loan_Status` outcomes from a historical application dataset. The fairness and robustness analysis described here evaluates prediction patterns within that dataset; it does not establish fairness in real lending decisions, repayment, default, or creditworthiness.

## Fairness evaluation

The fairness analysis uses available observed attributes such as `Gender`, `Married`, `Education`, `Self_Employed`, and `Dependents` when present in the source data. It reports:

- selection rate by group
- observed approval rate by group
- recall / true positive rate by group
- false negative rate by group
- false positive rate by group
- eligible-group gaps for selection-rate and equal-opportunity comparisons

Small groups are retained in the output but flagged as ineligible for direct between-group comparisons when they fall below the configured threshold.

Important limitation: demographic parity and equal opportunity differences are descriptive historical patterns, not legal findings or proof of fairness.

## Robustness evaluation

Robustness checks cover realistic invalid or irregular input scenarios:

- missing values
- unexpected categorical values
- invalid numeric types
- extreme numerical values
- negative numeric values
- small numeric perturbations

The evaluation records whether the system handles these scenarios without crashing or raises a clear validation message. This is a software validation exercise, not a real-world generalization guarantee.

## Reporting status

- Fairness summary generation: implemented
- Robustness validation: implemented
- App integration: limited to the existing risk-analysis page, with explicit historical-only wording
- Documentation: included here and in the project reports
