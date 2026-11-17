# Audited system architecture

## Scope and trust boundary

The application is an academic classifier for historical `Loan_Status`
labels (`Y`/`N`). It does not model repayment, delinquency, or default. The
actual labeled and unlabeled Kaggle CSVs and Joblib artifacts are local,
git-ignored resources. A registered Joblib artifact is executable
serialization and must only be loaded from a trusted local registry.

This output is an estimate of historical Loan_Status (approved/rejected), not default probability, repayment likelihood, a credit score, or a lending decision.

The audit found the executable training/inference code uses split-first
training, complete fold-local pipelines, and CV-only candidate selection and
tuning. A separate data-history caveat limits the evaluation: EDA analyzed
all 614 labeled rows before the later split, so the 123 rows later called the
holdout were not untouched. The existing holdout results are exploratory;
they are not an independent confirmation set.

## Layer responsibilities

| Layer | Responsibility | Main modules |
|---|---|---|
| Configuration and paths | Validated YAML settings, stable project-root paths, seed, domains, model options, report locations | `src/config/settings.py`, `src/config/paths.py`, `config.yaml` |
| Data contract | CSV loading, separate labeled/unlabeled validation, identifier and target checks, source quality audit | `src/data/loader.py`, `src/data/validation.py`, `src/data/quality_audit.py` |
| Split | Exclude target/IDs and non-allowlisted fields, reject duplicate feature records, create deterministic stratified split | `src/data/splitting.py` |
| Feature engineering | Deterministic total income, loan-to-income ratio, safe log1p features; no target statistics | `src/features/engineering.py` |
| Preprocessing | Median numeric imputation, scaling, categorical imputation, one-hot encoding with unknown-category handling | `src/preprocessing/transformers.py`, `src/preprocessing/pipeline.py` |
| Training and tuning | Four classifier factories, same stratified training CV, fold-local class/sample weights, CV-selected Random Forest search | `src/models/classifiers.py`, `src/models/train.py`, `src/models/optimize.py` |
| Evaluation | CV-based candidate selection, class-label probability lookup, baseline holdout metrics and charts | `src/evaluation/evaluate.py` |
| Artifact registry | Complete-pipeline Joblib serialization, metadata, registry allowlisting, SHA-256 checks and schema/target validation | `src/models/registry.py` |
| Explainability | Out-of-fold raw-feature permutation importance, local sensitivity, optional SHAP and limited subgroup metrics | `src/explainability/` |
| Application services | Historical dashboard metrics, report parsing, prediction input validation, complete-pipeline inference and explanations | `app/services/dashboard_service.py`, `app/services/prediction_service.py` |
| Presentation | Dashboard, predictor, risk review, model performance, about page, reusable charts/theme | `app/main.py`, `app/pages/`, `app/components/` |
| Quality | Unit, integration, and validation tests for contracts, leakage safeguards, model/artifact behavior, and application flows | `tests/unit/`, `tests/integration/`, `tests/validation/` |

## Data and model flow

1. Load and validate the 614-row labeled source separately from the 367-row
   unlabeled source. No source values are overwritten.
2. `split_labeled_data` validates `Loan_Status`, excludes `Loan_ID`, selects
   the configured 11 prediction-time fields, and creates the seed-42
   stratified 491/123 split. Exact duplicate feature rows are rejected.
3. Training and randomized search receive a complete
   `LoanFeatureEngineer` → `ColumnTransformer` → classifier pipeline. Numeric
   medians, scaling, categorical modes, and categories are fit within each
   CV training fold. Imbalance uses class weights or weights computed by each
   fold's fit.
4. Four candidate models are compared using the same five stratified folds.
   The selector uses mean balanced accuracy and ROC-AUC as a tie-breaker.
   Search and threshold decisions do not read holdout metrics.
5. The selected baseline v1.0.0 is evaluated once in the saved report.
   Tuned v1.1.0 has training-only CV results and no holdout evaluation.
6. The artifact registry verifies local membership, digest, metadata,
   dependency/schema information, and target mapping before deserializing.
7. Streamlit obtains the 11 raw fields, validates values, and calls the
   registered complete pipeline. The UI displays historical Loan_Status
   class predictions, model version, and a non-causal sensitivity check. It
   withholds probabilities because calibration has not been validated.

Feature importance indicates predictive contribution within this dataset,
not causation.

## Validation and release boundary

The current implementation passed 121 tests with 81% line coverage; actual
candidate CV means and baseline confusion matrix/metrics were independently
recomputed. The EDA/holdout chronology means the existing holdout is not
independent despite correct split, CV, and preprocessing code. The tree
models also show large in-sample/CV gaps. These facts block operational
readiness and independent generalization claims.

Before any future performance claim, acquire a new lawful labeled cohort and
lock the evaluation rows before EDA, transformations, feature decisions, or
tuning. A repayment/default system would require a separate outcome-specific
target, representative data, independent validation, calibration if
probabilities are exposed, fairness/legal review, and human governance.

See [final audit](../reports/final_audit_report.md),
[leakage audit](../reports/leakage_audit.md),
[model validation](../reports/model_validation_report.md), and
[test results](../reports/final_test_report.md).
