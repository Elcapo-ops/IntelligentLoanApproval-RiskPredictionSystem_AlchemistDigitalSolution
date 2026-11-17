# Project architecture

## Purpose and current scope

This repository retains the academic title "Intelligent Loan Approval &
Credit Risk Prediction System", but its implemented model estimates only
historical `Loan_Status` labels. It does not predict default, repayment, or
real-world lending decisions. It is not an autonomous lending system. Its
explainability layer describes model behavior and performs limited exploratory
subgroup analysis; it does not establish causation, legal fairness, or a
sufficient rationale for an individual lending decision.

This output is an estimate of historical Loan_Status (approved/rejected), not default probability, repayment likelihood, a credit score, or a lending decision.

## Layers

- `app/` is the Streamlit presentation layer. `app.py` delegates to
  `app/main.py`; reusable UI elements belong in `components/` and `pages/`.
- `app/services/` is the application-service boundary. Dashboard and
  prediction data/model operations are kept out of Streamlit page code.
- `src/config/` loads and validates the root `config.yaml`.
- `src/data/` owns separate CSV loading, schema validation, profiling,
  read-only quality auditing, and split-first training/holdout separation.
- `src/features/` and `src/preprocessing/` build the training/inference
  feature and preprocessing pipeline. Data-dependent preprocessing is fitted
  only inside training folds after splitting labeled data.
- `src/models/` owns classifier training, CV-only optimization, and trusted
  versioned artifacts. `src/evaluation/` owns candidate selection and model
  metric calculation.
- `src/explainability/` owns fold-held-out permutation importance, optional
  SHAP over raw named inputs, local one-feature sensitivity analyses, and
  exploratory subgroup metrics. It keeps categorical features grouped under
  their original column names by explaining or permuting raw pipeline inputs.
- `src/prediction/` owns prediction-time application logic.
- `src/utils/` contains shared exceptions, logging, and random-seed helpers.
- `tests/` separates unit, integration, and project-validation checks.

## Configuration and reproducibility

`config.yaml` is the central runtime configuration for raw and processed
dataset paths, model/report/log locations, split size, cross-validation folds,
candidate and optimization settings, explainability settings, Streamlit, and
logging. `ProjectPaths` resolves
relative paths from the repository root, not the current working directory.
It can create configured directories without creating or processing data.
Missing or invalid required values raise `ConfigurationError`.

The configured project seed is applied to Python and NumPy by
`set_random_seeds`; model estimators must receive that same seed explicitly.
Seeds are validated against NumPy's supported range.
`PYTHONHASHSEED` must be set before the interpreter starts if hash-order
reproducibility is required. Determinism is not a guarantee of identical
results across different library versions or hardware.

## Errors and logging

Expected application failures use the `LoanPredictionError` hierarchy.
Configuration errors have their own subtype; future data and artifact
operations have dedicated subtypes. Unexpected errors should not be silently
converted to successful results. The named `loan_prediction` logger writes
structured JSON records to the console and configured log file by default;
generated logs are excluded from version control.

## Data and generated artifacts

Private or large datasets under `data/raw/`, `data/processed/`, and
`data/external/`, as well as generated models and reports, are ignored by
Git. Their `.gitkeep` placeholders remain tracked to preserve the folder
layout. Do not place credentials or personally identifying loan data in
tracked files.

## Integrated workflow (Phases 8.1-8.3 verified)

1. The loader reads labeled and unlabeled CSVs separately and validates their
   schemas and target contract.
2. The splitter separates labeled features and target, excludes identifiers
   and unavailable columns, and reserves a stratified holdout before any
   learned preprocessing.
3. Complete feature-engineering, preprocessing, and estimator pipelines run
   within training-only cross-validation; candidate pipelines are then fit on
   the training partition.
4. Evaluation selects from CV results before applying the selected registered
   pipeline to the holdout. The report records class ordering and the positive
   probability column.
5. Artifact loading checks registry membership, hashes, metadata, feature
   schema, and target mapping before deserialization.
6. Streamlit pages call application services, which validate the raw
   prediction schema and use the registered complete pipeline. Explanations
   use training-partition reference values and remain descriptive, not causal.

The configured feature allowlist has 11 fields: four numeric and seven
categorical. The fitted pipeline adds total income, loan-to-income ratio, and
configured log1p features before fold-fitted imputation, scaling, and
one-hot encoding.

Feature importance describes predictive contribution within this dataset,
not causation. The dataset contains no verified default or repayment outcomes.

The Phase 8.1 data-to-prediction integration test and Phase 8.2 QA suite
exercise the complete workflow, leakage safeguards, registered artifact
checks, application pages, and failure handling. See
[`reports/system_integration_report.md`](../reports/system_integration_report.md)
and [`testing_report.md`](testing_report.md).

The final audit found that the EDA notebook analyzed all 614 labeled source
rows before the later 491/123 split. Thus the observations later assigned to
the holdout were included in earlier descriptive and target-conditioned
analyses. Although training and CV code fit learned transformations only in
training folds and did not use holdout rows for model selection, that holdout
is not an independent confirmation set. See
[`../reports/final_audit_report.md`](../reports/final_audit_report.md) and
[`../reports/leakage_audit.md`](../reports/leakage_audit.md).

## Reproducibility and version semantics

The root `requirements.txt` pins runtime and test/coverage packages. Project
metadata requires Python 3.12 or newer; the delivery installation and suite
were checked in a fresh Python 3.14.5 environment. A new checkout must supply
the git-ignored Kaggle CSVs and locally registered model artifacts before
running data-dependent integration checks or the predictor.

Candidate CV results and tuned-model CV results are training-only. The
published holdout results are for baseline Random Forest v1.0.0 only; tuned
v1.1.0 has not been evaluated on that holdout. The baseline holdout was
previously used and is not pristine for future repeated selection. The
current UI intentionally hides probabilities because calibration has not
been validated.
