# Phase 5.1 — Model evaluation

This output is an estimate of historical Loan_Status (approved/rejected), not default probability, repayment likelihood, a credit score, or a lending decision.

## Protocol

- Positive class: `Y` (approved); negative class: `N` (rejected).
- Selection used training-partition CV only: highest mean stratified CV balanced accuracy, with mean ROC-AUC as the tie-breaker.
- Selected candidate: `random_forest` version `1.0.0`.
- Selection scores: balanced accuracy 0.684495; ROC-AUC 0.778701.
- The final stratified holdout was evaluated after CV selection (123 rows). No learned preprocessing was fit on it, and it was not passed to CV, model selection, tuning, or threshold selection. However, all 614 labeled rows, including these later holdout rows, were included in earlier descriptive EDA and target-conditioned recommendations before the split. The holdout is therefore not fully untouched or independent; treat these metrics as exploratory rather than an unbiased external performance estimate.
- Fitted class ordering: `['N', 'Y']`; probability column 1 was used for positive class `Y`.
- Undefined precision/recall/F1 use `zero_division=0`; ROC-AUC is reported as unavailable if the holdout contains only one class; specificity is unavailable if no negatives are present.
- Independent confusion-matrix calculations were compared against the metric-library results.

## Cross-validation comparison

Values are mean ± standard deviation across identical stratified training folds; standard deviation summarizes fold variation. The holdout is not included in this comparison.

| Model | Accuracy | Balanced accuracy | ROC-AUC | Specificity | Precision (Y) | Recall (Y) | F1 (Y) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `logistic_regression` | 0.723 ± 0.068 | 0.679 ± 0.074 | 0.743 ± 0.092 | 0.558 ± 0.117 | 0.799 ± 0.044 | 0.799 ± 0.079 | 0.797 ± 0.054 |
| `decision_tree` | 0.656 ± 0.060 | 0.631 ± 0.064 | 0.648 ± 0.076 | 0.565 ± 0.147 | 0.781 ± 0.047 | 0.697 ± 0.099 | 0.733 ± 0.058 |
| `random_forest` | 0.745 ± 0.061 | 0.684 ± 0.082 | 0.779 ± 0.072 | 0.520 ± 0.168 | 0.798 ± 0.052 | 0.849 ± 0.068 | 0.821 ± 0.042 |
| `gradient_boosting` | 0.768 ± 0.068 | 0.681 ± 0.088 | 0.734 ± 0.110 | 0.448 ± 0.153 | 0.785 ± 0.049 | 0.914 ± 0.054 | 0.844 ± 0.044 |

## Selected model final holdout metrics

| Metric | Value |
|---|---:|
| accuracy | 0.8618 |
| balanced_accuracy | 0.8491 |
| roc_auc | 0.8672 |
| specificity | 0.8158 |
| precision | 0.9146 |
| recall | 0.8824 |
| f1 | 0.8982 |

### Confusion matrix

Rows are actual classes `[N, Y]`; columns are predicted classes `[N, Y]`.

| Actual / Predicted | N | Y |
|---|---:|---:|
| N | 31 | 7 |
| Y | 10 | 75 |

### Classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| N | 0.7561 | 0.8158 | 0.7848 | 38 |
| Y | 0.9146 | 0.8824 | 0.8982 | 85 |

Macro average: precision 0.8354, recall 0.8491, F1 0.8415.

## Figures and artifacts

- [CV model comparison with fold variation](figures/model_comparison_cv.png)
- [Selected model holdout confusion matrix](figures/confusion_matrix_holdout.png)
- [Selected model holdout ROC curve](figures/roc_curve_holdout.png)
- [Selected model holdout precision-recall curve](figures/precision_recall_holdout.png)
- Machine-readable model comparison: `model_comparison.csv`.

These evaluation results describe this dataset and fixed modeling protocol. They do not establish causal relationships, fairness, or suitability for real lending decisions.
