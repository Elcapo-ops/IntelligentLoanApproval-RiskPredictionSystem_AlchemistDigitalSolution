# Testing and installation report

## Environment and clean installation

Final verification used Windows and Python **3.14.5** in a newly created
virtual environment outside the project. The following completed successfully:

1. `python -m pip install -r requirements.txt`
2. `python -m pip install -e .`
3. `python -m pip check` — **No broken requirements found**
4. Imports of the Streamlit entry point, configuration, loader, evaluation,
   feature engineering, model registry/training, and preprocessing modules.
5. `python -m pytest --cov=app --cov=src --cov-report=term-missing -q` from
   the clean environment.
6. Editable installation was repeated after aligning the project Python
   minimum with the pinned dependency metadata.

The final clean-environment run executed **121 tests: 121 passed, 0 failed,
0 skipped**. Line coverage was **81%** (1,753/2,160 statements) over `app/`
and `src/`. This run had the configured Kaggle CSV files available in the
workspace; the data-dependent contract and full-flow tests are marked to skip
when those ignored source files are absent in another checkout.

## Test areas

- Data paths, loading, schema, target values, missingness, duplicate rows and
  IDs, cross-file ID overlap, and labeled/unlabeled separation.
- Stratified split isolation, feature allowlist, target/identifier exclusion,
  fold-local preprocessing, unknown categories, and duplicate feature rows.
- Training and tuning on training folds only; selection remains driven by CV
  metrics, not holdout or threshold fields.
- Model export/load, registry membership, schema and target mapping,
  integrity checks, corrupt artifacts, and prediction class/probability
  ordering.
- Zero income, large finite values, missing optional values, empty/malformed
  input, unknown categories, and values beyond the estimator's numeric range.
- Streamlit startup, route registration, page rendering, predictor schema,
  successful and optional-value submissions, results, and surfaced artifact
  errors.
- Real acquired-data end-to-end train/evaluate/export/load/predict/explain
  flow, with output files isolated in temporary test directories.

## Coverage gaps

Coverage is reported, not used as a pass/fail threshold. The explainability
command-line runner is not exercised by the automated suite. Some optional
branches in fairness, explainability, dashboard artifact loading, and
initialization error handling remain uncovered. Coverage excludes notebooks
and third-party packages.

## Reproduce the verification

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
python -m pytest --cov=app --cov=src --cov-report=term-missing -q
```

The CSVs are excluded from source control. To execute data-dependent tests,
obtain the two dataset files from Kaggle and place them in `data/raw/` as
described in [`dataset_source.md`](dataset_source.md). Without them, the
dataset contract and real-data integration tests skip; unit tests still test
missing-file behavior.

For the earlier phase-specific QA summary and confirmed edge-case fix, see
[`../reports/phase_8_2_qa_report.md`](../reports/phase_8_2_qa_report.md).
