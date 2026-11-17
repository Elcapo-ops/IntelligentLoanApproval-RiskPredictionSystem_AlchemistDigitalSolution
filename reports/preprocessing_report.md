# Phase 3.1 — Leakage-safe preprocessing pipeline

## Implementation

- `src/data/splitting.py` validates labeled input, separates `Loan_Status`,
  stratifies a train/holdout split using `training.test_size` and
  `project.random_seed`, and returns both partitions without fitting any
  transformation. The configured feature allowlist includes the declared
  numeric and categorical input columns only; `Loan_Status`, `Loan_ID`, and
  other undeclared columns (including any target-derived or unavailable
  prediction-time columns) are not passed as features.
- Before splitting, exact duplicate model-feature rows are rejected so
  duplicate or conflicting applications cannot cross the partitions silently.
  The Phase 1.2 audit found no duplicate source rows or duplicate IDs; the
  EDA training data also contains no duplicate model-feature rows.
- `src/preprocessing/transformers.py` defines numeric median imputation
  followed by `StandardScaler`, and categorical most-frequent imputation
  followed by `OneHotEncoder(handle_unknown="ignore")`. Empty columns are
  retained through imputation.
- `src/preprocessing/pipeline.py` builds an unfitted `ColumnTransformer` and
  a complete estimator `Pipeline`. The first step is the deterministic loan
  feature engineer; the fitted imputer, scaler, and encoder then learn only
  from the data passed to `fit`.

## Correct use

Use `split_labeled_data` once to reserve the final holdout. Use the complete
pipeline from `build_model_pipeline(estimator)` as the estimator passed to
cross-validation and hyperparameter search on **training data only**. Scikit-
learn clones and fits that pipeline independently inside each fold, ensuring
imputation, scaling, and category encoding are learned from that fold's
training rows. Do not call `build_preprocessor().fit_transform` on the full
labeled dataset before splitting, and do not use the holdout during fitting,
cross-validation, feature selection, or threshold selection.

No SMOTE or other resampling is applied. If class imbalance handling is later
evaluated, it must be placed inside the cross-validated training pipeline.

## Tests and verification

Unit tests cover stratification and reproducibility, target/identifier/
undeclared-feature exclusion, duplicate-feature rejection, training-only
imputer statistics, unseen-category handling, deterministic feature
engineering, and use of the complete pipeline in cross-validation. The
feature-engineering and preprocessing test modules passed together:
**14 passed**. The full project suite passed: **58 passed**.
