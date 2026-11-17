# Final delivery checklist

## Project and documentation

- [x] Project overview, problem statement, objectives, workflow, and folder
  structure documented in the README.
- [x] Dataset source, acquisition requirements, and licensing caveat linked.
- [x] Data dictionary includes source schema, dtypes, missing counts,
  categories, target, ID, and derived feature formulas.
- [x] Cleaning, EDA, feature engineering, algorithms, comparison results,
  evaluation metrics, selected model, and explanation limitations documented.
- [x] Architecture, model card, testing report, and final checklist provided.
- [x] Streamlit screenshots linked from the README.

## Code and reproducibility

- [x] Dependency versions are pinned to installed, compatible packages.
- [x] Project metadata Python floor aligned to the pinned dependency
  requirements (`>=3.12`).
- [x] Fresh virtual environment installation, editable project installation,
  imports, and `pip check` verified on Python 3.14.5.
- [x] Full clean-environment tests and coverage executed; see
  [`testing_report.md`](testing_report.md).
- [x] Dataset-dependent checks skip explicitly when git-ignored source data is
  not present.
- [x] Data and model/report outputs stay excluded from source control unless
  explicitly documented as placeholders/reports.
- [x] Automated scan found no embedded private values in application, source,
  test, or configuration files.

## Outputs and caveats

- [x] Model performance distinguishes baseline v1.0.0 holdout metrics from
  tuned v1.1.0 training-CV metrics.
- [x] Streamlit predictions are identified as historical approval-label
  estimates, not default or repayment predictions.
- [x] Probability output remains disabled because calibration is unvalidated.
- [x] Existing `data/raw/` CSVs and `artifacts/models/` model files are local,
  git-ignored inputs/outputs, not portable delivery artifacts.
- [x] The Kaggle dataset's license/redistribution terms are not verified; users
  must confirm applicable terms before redistribution.
- [x] SHAP was not run in the evaluated environment; no SHAP results are
  claimed.
- [x] No model performance is claimed for tuned v1.1.0 on the holdout.

## Before any real-world use

- [ ] Obtain lawful, representative repayment/default outcome data.
- [ ] Conduct independent temporal/external validation and validate
  calibration and decision thresholds.
- [ ] Complete domain, security, privacy, fairness, and legal reviews.
- [ ] Establish human oversight, monitoring, recourse, and governance.

This delivery is an academic prototype, not a production lending system.
