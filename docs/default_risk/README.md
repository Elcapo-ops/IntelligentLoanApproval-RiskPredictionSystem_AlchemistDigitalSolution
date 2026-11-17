# Loan Default / Credit Risk Prediction Module

## Overview

This module implements an independent Loan Default / Credit Risk Prediction pipeline using
LendingClub historical loan data. It is intentionally isolated from the historical
`Loan_Status` approval classifier and is a future-ready scaffold for genuine loan default
risk work using a repayment dataset such as LendingClub or Give Me Some Credit.

The module provides:

- **Dataset inspection and audit** — validates schema, target statuses, missing values,
  duplicate IDs, and temporal coverage of the LendingClub dataset.
- **Leakage-aware target construction** — maps matured outcomes (`Charged Off` → default,
  `Fully Paid` → non-default) and excludes unresolved statuses; constructs only
  application-time features.
- **Feature eligibility policy** — enforces application/origination-time feature selection;
  excludes identifiers, post-origination outcomes, and underwriting outputs.
- **Chronological model training** — compares four candidate models using expanding
  time-series cross-validation on training data only, with a temporal holdout set.
- **Probability calibration** — sigmoid calibration fit only on training folds; holdout
  never used for calibration or model selection.
- **Risk categorization** — assigns Low / Moderate / High / Very High Risk bands based on
  calibrated default probabilities.
- **Model explainability** — permutation importance on training-only data with reports and
  plots.
- **Inference** — single-applicant prediction with input validation, risk category assignment,
  disclaimer, and local sensitivity analysis.
- **Model artifact registry** — joblib serialization with SHA-256 integrity verification,
  metadata tracking, and schema validation.

## Project Structure

```
src/default_risk/
  __init__.py         Package exports
  data.py             Dataset inspection, target construction, feature eligibility
  modeling.py         Model comparison, training, calibration, evaluation, explainability
  prediction.py       Inference, input validation, risk categorization, sensitivity analysis
  dataset_status.py   Dataset availability and status gate
  train.py            CLI entry point for training
```

## Configuration

Module settings live in `config.yaml` under the `default_risk` key:

| Setting | Description | Default |
|---------|-------------|---------|
| `training_ready` | Whether the module is cleared for training/prediction | `true` |
| `test_start_year` | First year assigned to temporal holdout | `2015` |
| `max_train_rows` | Maximum training sample rows | `50000` |
| `max_test_rows` | Maximum holdout sample rows | `20000` |
| `cv_folds` | Number of temporal CV folds | `3` |
| `calibration_method` | Calibration method (`sigmoid` or `isotonic`) | `sigmoid` |
| `risk_category_thresholds` | Low/moderate/high/very-high probability bands | See below |

Risk category thresholds:

| Threshold | Value | Range |
|-----------|-------|-------|
| `low_risk_max` | `0.10` | [0.00, 0.10) |
| `moderate_risk_max` | `0.25` | [0.10, 0.25) |
| `high_risk_max` | `0.50` | [0.25, 0.50) |
| — | — | [0.50, 1.00] Very High Risk |

## Data

Source dataset: LendingClub `accepted_2007_to_2018Q4.csv.gz`
([Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club),
CC0: Public Domain).

Location: `data/default_risk/raw/accepted_2007_to_2018Q4.csv.gz`

Target construction:

| `loan_status` value | Mapped target | Meaning |
|---------------------|---------------|---------|
| `Charged Off` | `1` | Default |
| `Fully Paid` | `0` | Non-default |
| All other statuses | excluded | Unresolved; never mapped |

Feature set (20 features, application-time only):

- **Numeric (13):** `loan_amnt`, `annual_inc`, `dti`, `delinq_2yrs`,
  `fico_range_low`, `fico_range_high`, `inq_last_6mths`, `open_acc`, `pub_rec`,
  `revol_bal`, `revol_util`, `total_acc`, `credit_history_months`
- **Categorical (7):** `term`, `emp_length`, `home_ownership`,
  `verification_status`, `purpose`, `addr_state`, `application_type`

Derived: `credit_history_months` is computed from `issue_d` and `earliest_cr_line`.
Neither raw date is a model feature.

## Model

Registered artifact: `lendingclub_default_risk:0.1.0`

Location: `artifacts/models/default_risk/lendingclub_default_risk-0.1.0.joblib`

Registry: `artifacts/models/default_risk/registry.json`

The artifact is a calibrated scikit-learn `Pipeline` with:
- `ColumnTransformer` preprocessing (median imputation + `StandardScaler` for numeric;
  most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")` for categorical)
- Selected base estimator (chosen by training-only temporal CV F2 at threshold 0.50)
- `CalibratedClassifierCV` (sigmoid method, expanding TimeSeriesSplit folds)

## Reports

Generated under `reports/default_risk/`:

| Report | File | Description |
|--------|------|-------------|
| Dataset audit | `lendingclub_dataset_audit.json/.md` | Schema, target counts, missing values, duplicates, temporal coverage |
| Feature eligibility | `feature_eligibility.md` | Included/excluded fields with reasons |
| Model comparison | `model_comparison.json` | Training-only temporal CV results per candidate |
| Model evaluation | `model_evaluation.md` | Final holdout metrics (uncalibrated vs calibrated) |
| Calibration report | `calibration_report.md` | Brier score, log loss, calibration curve plot |
| Calibration plot | `calibration_curve.png` | Reliability diagram for holdout predictions |
| Explainability | `explainability_report.md` | Top permutation importance features |
| Permutation importance | `permutation_importance.csv/.png` | Feature importance scores and bar chart |

## Usage

### Training (CLI)

```bash
python -m src.default_risk.train --project-root .
```

Optional overrides:
```bash
python -m src.default_risk.train \
  --dataset data/default_risk/raw/accepted_2007_to_2018Q4.csv.gz \
  --test-start-year 2015 \
  --max-train-rows 50000 \
  --max-test-rows 20000 \
  --cv-folds 3 \
  --random-state 42
```

### Inference (Python)

```python
from pathlib import Path
from default_risk.prediction import load_default_risk_model, predict_default_risk

model, metadata = load_default_risk_model(Path("."))

applicant = {
    "loan_amnt": 10000,
    "term": "36 months",
    "emp_length": "5 years",
    "home_ownership": "RENT",
    "annual_inc": 60000,
    "verification_status": "Source Verified",
    "purpose": "debt_consolidation",
    "addr_state": "CA",
    "dti": 15.5,
    "delinq_2yrs": 0,
    "fico_range_low": 700,
    "fico_range_high": 749,
    "inq_last_6mths": 2,
    "open_acc": 10,
    "pub_rec": 0,
    "revol_bal": 5000,
    "revol_util": 45.0,
    "total_acc": 20,
    "application_type": "INDIVIDUAL",
    "credit_history_months": 84,
}

result = predict_default_risk(model, applicant, metadata)
print(result["default_probability"])
print(result["risk_category"])
print(result["disclaimer"])
```

### Inference (Streamlit)

The module is exposed through the Streamlit page at `app/pages/default_risk_predictor.py`,
which provides a single-prediction form, risk category display, sensitivity analysis,
and feature importance visualization.

## Model Card

See `docs/default_risk/model_card.md` for the full model card including intended use,
limitations, ethical considerations, and disclaimers.

## Data Dictionary

See `docs/default_risk/data_dictionary.md` for field-level definitions of all model
inputs, target construction, and excluded fields.

## Disclaimer

> "This output is a statistical estimate of loan default risk based on historical loan
> performance data. It is not a guarantee of future repayment or default, a regulated
> credit rating, or an automatic lending decision."

This disclaimer is displayed with every prediction output in both the Python API and
the Streamlit UI.

## Isolation from Historical Classifier

This module is separate from the project's historical `Loan_Status` approval classifier.
It does not share code, artifacts, or configuration with that system. Training or
inference in this module does not affect the historical approval pipeline.
