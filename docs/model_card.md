# Model card: historical loan approval label classifier

## Summary

- **System:** Intelligent Loan Approval & Credit Risk Prediction System
- **Task:** classify the dataset's historical `Loan_Status` value (`Y` or `N`)
- **Current prediction artifact:** tuned Random Forest v1.1.0
- **Selected family:** Random Forest; baseline candidate selection used
  training-only CV balanced accuracy with ROC-AUC as a tie-breaker
- **Intended context:** academic experiments and demonstration only
- **Not intended for:** real lending decisions, creditworthiness judgments,
  default/repayment prediction, or autonomous decision-making

This output is an estimate of historical Loan_Status (approved/rejected), not default probability, repayment likelihood, a credit score, or a lending decision.

## Data

The source is Kaggle's [Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset).
The measured labeled dataset contains 614 rows (422 `Y`, 192 `N`) and the
separate unlabeled file contains 367 rows. The modeling split has 491
training rows and a 123-row holdout. Fields include income, requested loan
amount, loan term, credit-history indicator, and selected applicant
demographic/categorical attributes. See [`data_dictionary.md`](data_dictionary.md)
for definitions and missingness.

The data is small, historical, and target labels represent previous decisions.
The dataset contains no verified default or repayment outcome labels; this
model is not a default or repayment model.
The data audit observed missing values, including 50 missing
`Credit_History` values in labeled data. No records are deleted or imputed in
the source files. A separate quality audit documented the observed values.
Kaggle license and reuse conditions have not been independently verified.

## Model and training

The full serialized pipeline includes deterministic feature engineering,
fold-fitted median numeric imputation, numeric scaling, categorical
most-frequent imputation, one-hot encoding, and a classifier. Original model
inputs consist of 11 configured fields; identifiers and target are excluded.
Cross-validation and hyperparameter search fit preprocessing only within each
training fold. The training/holdout split is stratified and performed before
fitting.

Four candidates were compared using the same five stratified CV folds and
seed 42. The selected Random Forest was tuned with training-only
RandomizedSearchCV using balanced accuracy as the refit criterion. No
threshold tuning was conducted. The v1.1.0 tuned artifact is a registered
local Joblib pipeline with metadata and SHA-256 integrity checks.

## Measured performance

### Training-only CV candidates

| Candidate | Balanced accuracy (mean ± SD) | ROC-AUC (mean ± SD) |
|---|---:|---:|
| Logistic Regression | 0.6785 ± 0.0737 | 0.7427 ± 0.0922 |
| Decision Tree | 0.6312 ± 0.0641 | 0.6478 ± 0.0756 |
| Random Forest v1.0.0 | 0.6845 ± 0.0824 | 0.7787 ± 0.0724 |
| Gradient Boosting | 0.6811 ± 0.0883 | 0.7342 ± 0.1102 |

Tuned Random Forest v1.1.0 achieved training-only CV balanced accuracy
**0.6928 ± 0.0635** and ROC-AUC **0.7828 ± 0.0601**. The baseline comparison
on the same folds was 0.6845 ± 0.0824 balanced accuracy and 0.7787 ± 0.0724
ROC-AUC. The change is modest and the search itself can make CV estimates
optimistic.

### Previously measured holdout — baseline v1.0.0 only

| Metric | Value |
|---|---:|
| Accuracy | 0.8618 |
| Balanced accuracy | 0.8491 |
| Precision for `Y` | 0.9146 |
| Recall for `Y` | 0.8824 |
| F1 for `Y` | 0.8982 |
| ROC-AUC | 0.8672 |
| Specificity | 0.8158 |

These holdout results are solely for **baseline Random Forest v1.0.0**. The
tuned v1.1.0 artifact has no holdout evaluation. The 123 rows later assigned
to the holdout were included in all-614-row EDA and target-conditioned
recommendations before the split, and the baseline holdout was subsequently
evaluated. It is therefore not an untouched or independent estimate and is
not suitable for future model selection or unbiased generalization claims.
Scores measure historical label agreement, not actual loan outcomes.

## Explainability and subgroup analysis

Training-fold permutation importance ranked `Credit_History` highest (mean
balanced-accuracy decrease 0.1823, SD 0.0670), then `ApplicantIncome`
(0.0346), `LoanAmount` (0.0342), and `CoapplicantIncome` (0.0309). The app
provides one-feature-at-a-time class sensitivity checks using training
references. These are descriptive model-behavior checks and not causal
attributions or sufficient individual decision rationales. Feature importance
indicates predictive contribution within this dataset, not causation.

Exploratory subgroup metrics are available for limited observed fields. They
are not a complete demographic, legal, or fairness audit and do not establish
discrimination or its absence. SHAP was not run because its optional package
was not installed in the evaluated environment.

## Known limitations and risks

- Historical approval labels are not repayment/default outcomes and may encode
  past policy and bias.
- The sample is small and may not generalize across time, institutions,
  regions, or populations.
- No probability calibration or operational threshold analysis has been
  validated; the UI therefore shows class predictions only.
- The v1.0.0 holdout was previously evaluated; v1.1.0 lacks independent
  holdout evaluation.
- Missingness, omitted variables, and limited subgroup size constrain
  interpretation and fairness analysis.
- Permutation importance and individual sensitivity do not imply causation.
- Joblib artifacts use pickle serialization. Only trusted registered files
  should be loaded; registry hashes are not a digital signature if an attacker
  can modify both the artifact and registry.

## Appropriate use

Use the system for education, reproducibility experiments, and analysis of
historical labels. Do not use outputs to accept, reject, price, or rank real
applicants. Any future operational research would require lawful data access,
larger representative data including repayment outcomes, independent
temporal/external validation, calibration, comprehensive fairness and legal
review, security review, monitoring, and meaningful human oversight.

## References

- [Dataset source and acquisition](dataset_source.md)
- [Evaluation report](../reports/evaluation_report.md)
- [Tuning report](../reports/hyperparameter_tuning_report.md)
- [Explainability report](../reports/explainability_report.md)
- [Business insights](../reports/business_insights.md)
- [Model artifact trust boundary](../reports/model_artifact_report.md)
