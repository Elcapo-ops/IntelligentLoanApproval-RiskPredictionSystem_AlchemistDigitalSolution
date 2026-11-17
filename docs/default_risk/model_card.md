# Model Card — LendingClub Default Risk

## Model Details

| Field | Value |
|-------|-------|
| Model name | `lendingclub_default_risk` |
| Model version | `0.1.0` |
| Model type | Calibrated scikit-learn `Pipeline` (ColumnTransformer + base estimator + `CalibratedClassifierCV`) |
| Task | Binary classification: loan default risk |
| Target | `loan_status` → `default_target` (`Charged Off` → 1, `Fully Paid` → 0) |
| Training data | LendingClub `accepted_2007_to_2018Q4.csv.gz` (CC0: Public Domain) |
| Artifact | `artifacts/models/default_risk/lendingclub_default_risk-0.1.0.joblib` |
| Registry | `artifacts/models/default_risk/registry.json` |
| Metadata | `artifacts/models/default_risk/lendingclub_default_risk-0.1.0.metadata.json` |

## Intended Use

### Primary use case

Estimate the historical probability of loan default (`Charged Off`) for individual
loan applications based on application-time borrower and bureau attributes. Outputs
are calibrated default probabilities in [0, 1] with an associated risk category
(Low / Moderate / High / Very High Risk).

### Intended users

- Credit risk analysts exploring historical default patterns
- Loan underwriting teams conducting comparative risk assessment
- Academic and research use on LendingClub historical data

### Out-of-scope uses

- **Not** a regulated credit rating or credit score
- **Not** an automatic lending decision (approve/deny)
- **Not** a guarantee of future repayment or default
- **Not** suitable for real-time production lending without independent validation
- **Not** suitable for populations or time periods materially different from the
  2007–2018 LendingClub training sample

## Training Data

### Source

LendingClub loan data, `accepted_2007_to_2018Q4.csv.gz`, sourced from Kaggle
(CC0: Public Domain as declared on Kaggle).

### Target construction

Only matured loans with resolved statuses are retained:

| Status | Mapped to | Action |
|--------|-----------|--------|
| `Charged Off` | 1 (default) | Retained |
| `Fully Paid` | 0 (non-default) | Retained |
| All other statuses (Current, Late, In Grace Period, Default, policy exceptions, missing) | excluded | Never mapped to either class |

### Train / holdout split

Temporal split by `issue_d` (loan issue date):

- **Training:** loans issued before `test_start_year` (configurable, default 2015)
- **Holdout:** loans issued in or after `test_start_year`

The holdout is never used for model selection, hyperparameter tuning, or calibration
fitting.

### Feature set

20 application-time features only (13 numeric + 7 categorical), as defined in
`docs/default_risk/data_dictionary.md`. No post-origination outcomes, underwriting
outputs, identifiers, or free-text fields are used.

## Model Development

### Candidate models

Four scikit-learn classifiers were compared using expanding time-series
cross-validation (TimeSeriesSplit) on training data only:

1. Logistic Regression (`saga`, `class_weight="balanced"`, `C=0.5`, `max_iter=120`)
2. Decision Tree (`class_weight="balanced"`, `max_depth=14`, `min_samples_leaf=50`)
3. Random Forest (`class_weight="balanced_subsample"`, `max_depth=18`,
   `min_samples_leaf=40`, `n_estimators=80`, `n_jobs=2`)
4. HistGradientBoosting (`learning_rate=0.08`, `l2_regularization=1.0`,
   `max_iter=50`, `max_leaf_nodes=15`, `min_samples_leaf=60`)

### Model selection

Selected by training-only temporal CV **F2 at threshold 0.50**, with PR-AUC as the
tie-breaker, then recall. F2 weights recall more heavily than precision, appropriate
for default-risk detection where missing defaults is costly.

### Preprocessing pipeline

- **Numeric:** `SimpleImputer(strategy="median")` → `StandardScaler()`
- **Categorical:** `SimpleImputer(strategy="most_frequent")` →
  `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
- All preprocessing is fit inside each CV fold to prevent data leakage.

### Calibration

`CalibratedClassifierCV` with the configured method (default: `sigmoid`), fit using
expanding TimeSeriesSplit folds within training data only. The final temporal holdout
is not used for calibration fitting.

## Evaluation

### Metrics reported

Accuracy, precision, recall, F1, F2, ROC-AUC, PR-AUC, specificity, balanced accuracy,
Brier score, log loss, confusion matrix at threshold 0.50.

### Holdout evaluation

Both uncalibrated and calibrated probabilities are evaluated on the temporal holdout
set. Calibration is assessed by Brier score and log loss improvement.

### Reports

- `reports/default_risk/model_comparison.json` — training-only temporal CV results
- `reports/default_risk/model_evaluation.md` — final holdout metrics
- `reports/default_risk/calibration_report.md` — calibration comparison
- `reports/default_risk/calibration_curve.png` — reliability diagram

## Risk Categories

Calibrated default probability is mapped to risk bands:

| Category | Probability Range |
|----------|-------------------|
| Low Risk | [0.00, 0.10) |
| Moderate Risk | [0.10, 0.25) |
| High Risk | [0.25, 0.50) |
| Very High Risk | [0.50, 1.00] |

These thresholds are illustrative and configurable; they are not a lending policy.

## Explainability

Global permutation importance is computed on a training-only sample (up to 1,000 rows)
using average precision scoring. Results are saved to:

- `reports/default_risk/permutation_importance.csv`
- `reports/default_risk/permutation_importance.png`
- `reports/default_risk/explainability_report.md`

Feature importance measures predictive contribution in this historical sample and is
**not** causal evidence or a lending rationale.

Local one-feature-at-a-time sensitivity analysis is available at inference time
through `default_risk.prediction.local_probability_sensitivity`.

## Ethical Considerations and Limitations

### Limitations

1. **Censored outcomes:** Unresolved loan statuses (Current, Late, etc.) are excluded,
   never mapped to non-default. The target is classification among resolved loans,
   not a fixed-horizon default-incidence estimate.
2. **Temporal coverage:** Training data covers loans issued 2007–Q4 2018. Performance
   on loans originated after this period is unknown.
3. **Single dataset:** Results are specific to the LendingClub platform and may not
   generalize to other lenders, geographies, or product types.
4. **Self-reported data:** Income and other applicant-reported fields may contain
   inaccuracies.
5. **No causal interpretation:** Feature importance and sensitivity results describe
   predictive association, not causation.

### Fairness considerations

- `addr_state` (state) is included as a feature. Users should assess whether regional
  patterns introduce unintended disparities.
- `emp_length`, `home_ownership`, `verification_status`, `purpose`, and
  `application_type` may correlate with protected attributes. Users should conduct
  independent fairness audits before any production use.
- The model was trained on historical data that may reflect past lending practices
  and biases.

### Required disclaimer

> "This output is a statistical estimate of loan default risk based on historical loan
> performance data. It is not a guarantee of future repayment or default, a regulated
> credit rating, or an automatic lending decision."

This disclaimer must be displayed with every prediction output.

## Artifact Integrity

The model artifact is registered with SHA-256 integrity hashes:

- **Artifact SHA-256:** stored in `artifacts/models/default_risk/registry.json`
- **Metadata SHA-256:** stored in `artifacts/models/default_risk/registry.json`
- **Registry key:** `lendingclub_default_risk:0.1.0`

Integrity is verified on every load via `models.registry.load_model_artifact`, which
checks artifact hash, metadata hash, feature schema, target mapping, and dependency
versions before deserialization.

## Dependencies

| Package | Purpose |
|---------|---------|
| scikit-learn | Pipelines, classifiers, calibration, metrics, preprocessing |
| pandas | Data manipulation |
| numpy | Numerical operations |
| joblib | Model serialization |
| PyYAML | Configuration loading |
| matplotlib | Visualization (reports only) |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | Initial release | LendingClub default risk model with chronological training, calibration, and risk categorization |

## Contact and Maintenance

Module owner: default-risk module maintainer
Last reviewed: see `docs/default_risk/implementation_audit.md`
