# Phase 8.2 — Automated QA report

## Result

**Complete.** The final automated run executed **121 tests: 121 passed, 0
failed, 0 skipped**. Line coverage for `app/` and `src/` was **81%**
(1,753 of 2,160 statements; 407 missed) using Python 3.14.5,
pytest-cov 7.0.0, and coverage 7.17.3.

Command:

```powershell
python -m pytest --cov=app --cov=src --cov-report=term-missing -q
```

## QA areas exercised

- **Dataset contracts:** configured CSV files exist; labeled and unlabeled
  schemas are validated independently; target labels are valid and complete;
  duplicates and identifier overlap are rejected; acquired data has 50
  missing `Credit_History` values and no missing target labels.
- **Leakage safeguards:** deterministic stratified split isolation; identifier,
  target, and unavailable/target-derived feature exclusion; fold-local
  preprocessing; training-only tuning; CV selection remains invariant to
  holdout and threshold-like columns; duplicate feature rows and duplicate IDs
  are checked.
- **Training and artifacts:** all candidate models train; evaluation uses the
  selected model and holdout after CV selection; registered artifact
  save/load, schema and target mapping checks, digest failures, missing or
  unregistered artifacts, and corrupt registered bytes are tested.
- **Inference:** raw feature schema validation, class mapping and probability
  dimensions/order, zero income, high finite values, missing optional values,
  empty/malformed input, unseen categories, and unsupported extreme values are
  exercised. Calibration remains unvalidated and prediction stays class-only.
- **Streamlit:** application startup/dashboard, configured route registration,
  page rendering, all predictor fields, successful prediction, all-optional
  inputs, live result counters, and user-visible artifact errors are tested.

## Defect found and fixed

An extremely large but finite numeric input passed service validation and
triggered a scikit-learn `ValueError` during model inference. The prediction
boundary now converts `ValueError` and numeric `RuntimeWarning` failures into
the project's user-visible `DataValidationError`. Realistic high values and
zero income continue to produce predictions; values outside the fitted
pipeline's representable range fail explicitly instead of crashing the page.

## Coverage interpretation

The 81% figure covers application and source modules together. Model training,
evaluation, prediction, preprocessing, and UI components have substantial
direct coverage. The explainability command-line runner (`src/explainability/
run.py`) is not exercised by this suite and has 0% coverage; optional and
less-used branches in explainability and fairness modules account for much of
the remaining uncovered source. Coverage is a measurement, not a pass/fail
quality threshold.

## Reproduction

Install the pinned dependencies with `python -m pip install -r requirements.txt`,
then run the command above from the project root. The test run did not skip
acquired-data checks because both configured Kaggle CSVs are present in this
workspace.
