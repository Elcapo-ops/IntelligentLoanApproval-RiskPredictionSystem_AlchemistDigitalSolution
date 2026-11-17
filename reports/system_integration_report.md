# Phase 8.1 — System integration report

## Status

**Complete.** The data-to-prediction workflow was exercised with the acquired
labeled dataset. No confirmed production defect required changing runtime
behavior. The integration change adds an end-to-end regression test that writes
all generated models and reports to a temporary directory.

This output is an estimate of historical Loan_Status (approved/rejected), not default probability, repayment likelihood, a credit score, or a lending decision.

## Verified workflow

1. **Dataset loading and validation:** the configured CSV loader returned
   614 labeled records and 367 separate unlabeled records. `Loan_Status` was
   present only in the labeled data.
2. **Train/holdout split:** stratified splitting produced 491 training rows and
   123 reserved holdout rows. The model input schema contains the configured
   11 prediction-time features and excludes `Loan_ID` and `Loan_Status`.
3. **Preprocessing and feature engineering:** all four configured model
   candidates were cross-validated as complete pipelines, then fit on the
   training partition. The integration test verifies the serialized pipeline's
   imputer statistics equal the engineered training-partition medians; the
   ApplicantIncome holdout median differs from that fitted statistic.
4. **Evaluation and selection:** selection used candidate CV results; the
   selected registered pipeline was evaluated on the reserved holdout. The
   confusion-matrix total matched 123 rows, and the evaluation outputs were
   written successfully.
5. **Export and loading:** artifacts were saved to and loaded from a temporary
   registry with the configured feature schema and `N: 0`, `Y: 1` target
   mapping. The loaded estimator exposed classes in `["N", "Y"]` order, making
   the positive (`Y`) probability column index 1. These checks do not enable
   displaying probabilities in the application; calibration remains
   unvalidated.
6. **Streamlit prediction and explanation:** the loaded pipeline accepted the
   exact ordered raw feature schema; its direct class prediction matched the
   prediction service output. One-feature sensitivity returned explanations
   for all 11 inputs using training-partition references. A negative numeric
   input raised the expected validation error.
7. **Dashboard:** historical statistics were loaded from the labeled CSV:
   614 applications, 422 approvals, and 192 rejections. These are historical
   labels, not live predictions or repayment outcomes.

The end-to-end test uses a three-fold CV setting and reduced estimator sizes
to keep test runtime bounded. Its artifacts live only in pytest's temporary
directory; it does not overwrite the saved production-facing artifacts or
published reports.

## Current registered prediction artifact

A separate runtime smoke check loaded the configured `random_forest:1.1.0`
artifact and exercised prediction and explanation from a working directory
outside the project root. Project paths still resolved to the workspace root.
The artifact exposed `["N", "Y"]` classes, returned a valid two-column
probability array, and produced a class-only prediction plus 11 feature
sensitivity checks. The user-facing predictor intentionally does not display
uncalibrated probabilities.

## Integrity and error handling

- The feature allowlist, stored feature schema, and prediction form schema
  matched.
- Identifier, target, and unavailable columns were excluded from model inputs.
- Preprocessing remained inside the CV pipeline and fitted only on training
  rows; the holdout was used only after CV-based model selection.
- Prediction validation rejects missing/extra schema fields, unsupported
  categories, negative values, non-finite numeric values, and invalid positive
  numeric values. Expected data and artifact failures use the project's
  exception hierarchy and are surfaced by the application.
- Model artifacts are loaded only after local registry membership, digest, and
  metadata checks.
- Dashboard and performance views use actual dataset and saved report outputs;
  holdout results remain labeled as baseline `random_forest` v1.0.0, distinct
  from tuned v1.1.0 CV results.

## Verification results

- Full test suite: **108 passed**.
- End-to-end real-dataset test: passed.
- Project import smoke check: **48 modules imported successfully**.
- Pylance import scan: no unresolved module names.
- Pylance diagnostics for the new integration test: no diagnostics.
- Python compilation: passed for `app`, `src`, and `tests`.
- `pip check`: no broken requirements.
- Additional data-to-live-prediction smoke check from outside the project root:
  passed.

## Remaining limitations

This system predicts historical `Loan_Status`, not loan repayment or default.
Probability calibration is not validated, so probabilities remain hidden.
The eventual holdout rows were included in the pre-split 614-row EDA and the
holdout has since been used for the documented baseline evaluation. It is not
an untouched or independent dataset for future model selection. Historical
metrics and subgroup patterns do not establish causation, fairness, or
suitability for real lending decisions.
