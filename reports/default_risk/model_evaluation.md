# LendingClub default-risk evaluation

## Protocol

- Source rows: 2,260,701; resolved target rows: 1,345,310.
- Training sample: 100,000; temporal holdout sample: 40,000.
- Training issue dates: 2007-07 to 2015-12.
- Holdout issue dates: 2016-01 to 2018-12.
- Unresolved statuses are excluded, never mapped to non-default.
- Candidate selection used expanding time-series CV on training rows only; the final holdout was not used for model selection, tuning, or calibration.
- Holdout metrics represent the sampled later-issue-date rows with resolved labels; because outcomes are censored and unresolved loans were excluded, these are not fixed-horizon default-incidence estimates.

## Temporal CV comparison (training data only)

| Model | PR-AUC | ROC-AUC | Default recall | Precision | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.3612 | 0.7003 | 0.6607 | 0.2983 | 0.4101 | 0.6433 |
| decision_tree | 0.2986 | 0.6434 | 0.6127 | 0.2686 | 0.3733 | 0.6063 |
| random_forest | 0.3593 | 0.7007 | 0.5873 | 0.3185 | 0.4121 | 0.6424 |
| hist_gradient_boosting | 0.3617 | 0.7009 | 0.0058 | 0.7184 | 0.0113 | 0.5024 |

Selected model: **hist_gradient_boosting**, based on training-only temporal CV PR-AUC.

## Final temporal holdout metrics

| Metric | Uncalibrated | Calibrated |
|---|---:|---:|
| Accuracy | 0.7763 | 0.7774 |
| Precision | 0.5492 | 0.5714 |
| Default recall | 0.0118 | 0.0268 |
| F1 | 0.0231 | 0.0511 |
| ROC-AUC | 0.6836 | 0.6821 |
| PR-AUC | 0.3730 | 0.3717 |
| Specificity | 0.9972 | 0.9942 |
| Balanced accuracy | 0.5045 | 0.5105 |
| Brier score | 0.1637 | 0.1635 |
| Log loss | 0.5022 | 0.5025 |

### Confusion matrices at threshold 0.50

- Uncalibrated: `{'true_negative': 30948, 'false_positive': 87, 'false_negative': 8859, 'true_positive': 106}`
- Calibrated: `{'true_negative': 30855, 'false_positive': 180, 'false_negative': 8725, 'true_positive': 240}`

Accuracy alone can conceal poor default detection when non-default loans are more common. Review default-class precision, recall, PR-AUC, calibration, and confusion matrices together. The 0.50 threshold is an analytical reference, not a lending policy.

Registered artifact: `C:\Projects\loan_predection_system\artifacts\models\default_risk\lendingclub_default_risk-0.1.0.joblib`

Feature importance is descriptive predictive contribution, not causation. See `permutation_importance.csv`.
