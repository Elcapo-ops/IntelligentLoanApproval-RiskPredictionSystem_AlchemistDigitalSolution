# Phase 4.1 — Model training and cross-validation

## Scope and leakage safeguards

- Labeled source: `data/raw/train_u6lujuX_CVtuZ9i.csv` only.
- Target: `Loan_Status`; classes: `Y, N`.
- Training rows: 491; holdout rows reserved: 123.
- Random seed: 42; stratified 5-fold CV on training rows only.
- Artifact version: `1.0.0`.
- All candidates used the same target labels and identical CV folds.
- The final holdout was not passed to cross-validation, fit, scoring, or model selection.
- No training-accuracy metric was calculated or used for selection.
- Preprocessing and feature engineering ran inside each estimator pipeline and were fit separately within CV folds.
- Class weighting was used instead of pre-split resampling; no SMOTE was applied.
- All four candidate pipelines were refit on the complete training partition for later holdout evaluation. No candidate is declared selected in this phase.

## Fixed model configurations

| Model | Parameters |
|---|---|
| `logistic_regression` | `{"class_weight": "balanced", "imbalance_handling": "class_weight='balanced'", "max_iter": 1000, "random_state": 42, "solver": "lbfgs"}` |
| `decision_tree` | `{"class_weight": "balanced", "imbalance_handling": "class_weight='balanced'", "max_depth": null, "min_samples_leaf": 2, "random_state": 42}` |
| `random_forest` | `{"class_weight": "balanced", "imbalance_handling": "class_weight='balanced'", "max_features": "sqrt", "min_samples_leaf": 2, "n_estimators": 300, "n_jobs": 1, "random_state": 42}` |
| `gradient_boosting` | `{"class_weight": "balanced_sample_weight", "imbalance_handling": "balanced sample weights computed inside each fit", "learning_rate": 0.05, "max_depth": 2, "n_estimators": 100, "random_state": 42, "subsample": 1.0}` |

## Training-partition cross-validation results

Values are mean ± standard deviation across the same stratified folds. These are CV estimates on training data, not final holdout performance.

| Model | accuracy (mean ± SD) | balanced_accuracy (mean ± SD) | roc_auc (mean ± SD) | specificity (mean ± SD) | precision (mean ± SD) | recall (mean ± SD) | f1 (mean ± SD) |
|---|---|---|---|---|---|---|---|
| `logistic_regression` | 0.723 ± 0.068 | 0.679 ± 0.074 | 0.743 ± 0.092 | 0.558 ± 0.117 | 0.799 ± 0.044 | 0.799 ± 0.079 | 0.797 ± 0.054 |
| `decision_tree` | 0.656 ± 0.060 | 0.631 ± 0.064 | 0.648 ± 0.076 | 0.565 ± 0.147 | 0.781 ± 0.047 | 0.697 ± 0.099 | 0.733 ± 0.058 |
| `random_forest` | 0.745 ± 0.061 | 0.684 ± 0.082 | 0.779 ± 0.072 | 0.520 ± 0.168 | 0.798 ± 0.052 | 0.849 ± 0.068 | 0.821 ± 0.042 |
| `gradient_boosting` | 0.768 ± 0.068 | 0.681 ± 0.088 | 0.734 ± 0.110 | 0.448 ± 0.153 | 0.785 ± 0.049 | 0.914 ± 0.054 | 0.844 ± 0.044 |

## Interpretation and limitations

- Models are compared using stratified cross-validation metrics, not training accuracy.
- Balanced accuracy and per-class precision/recall/F1 complement ROC-AUC because the labeled target is imbalanced.
- Logistic regression, decision tree, and random forest use `class_weight='balanced'`. Gradient boosting uses balanced sample weights computed from the target labels supplied to each fold's `fit` call.
- No threshold tuning, final model selection, or holdout evaluation was performed. Those decisions require a separate evaluation phase with the holdout used only as specified.
- Results describe this small Kaggle labeled sample and are not evidence of causal effects or lending suitability.
