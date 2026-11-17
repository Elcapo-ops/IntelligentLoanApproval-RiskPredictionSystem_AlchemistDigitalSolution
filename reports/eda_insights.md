# EDA business insights and modeling considerations

## Scope and evidence

This report interprets all **614 labeled rows** in
`data/raw/train_u6lujuX_CVtuZ9i.csv`; the separate unlabeled Kaggle test CSV
was not used. At the time this EDA was performed, no holdout had yet been
created. The later seed-42 split assigned 123 of these previously inspected
rows to the holdout. Therefore those holdout observations were included in
descriptive plots, target-conditioned summaries, and feature/preprocessing
recommendations. No preprocessing was fitted and no model was trained during
EDA, but the eventual holdout is not fully untouched or an independent
confirmation set. Treat its later metrics as exploratory; use a new
prospective or external labeled cohort for independent validation. Counts and
rates below are descriptive proportions, not causal effects, fairness
determinations, or predictions for future applicants.

The supporting descriptive report is
[`eda_report.md`](eda_report.md); the figures cited below are under
[`figures/`](figures/).

## Approval and rejection patterns

- **Overall approval is the majority outcome:** 422 of 614 applications
  (68.73%) are labeled approved and 192 (31.27%) rejected. The majority/minority
  ratio is 2.20:1. This is a moderate class imbalance: it makes raw accuracy
  alone an incomplete evaluation metric, but does not by itself establish that
  resampling is needed. See
  [approval versus rejection](figures/01_approval_vs_rejection.png).
- **Credit history has the clearest observed association with status.** Among
  475 applicants with `Credit_History = 1`, 378 were approved (79.58%). Among
  89 with `Credit_History = 0`, 7 were approved (7.87%). Of 50 applicants with
  missing credit history, 37 were approved (74.00%). The 71.71 percentage-point
  gap between the two observed credit-history categories is substantial in
  this sample, but it does not demonstrate that credit history alone caused
  the decisions, or that missingness has a meaningful business interpretation.
  See [approval by credit history](figures/02_approval_by_credit_history.png).
- **Approval rates vary by property area:** 179 rural applications had 110
  approvals (61.45%); 233 semiurban applications had 179 (76.82%); and 202
  urban applications had 133 (65.84%). These differences are associations in
  this dataset and may reflect other applicant or operational differences.
  See [approval by property area](figures/06_approval_by_property_area.png).
- **Self-employment rates are very similar in the observed data:** 343 of 500
  non-self-employed applications (68.60%) and 56 of 82 self-employed
  applications (68.29%) were approved. The missing-status group had 23
  approvals out of 32 applications (71.88%); its small size and unknown
  missingness mechanism make that rate especially uncertain. See
  [approval by employment status](figures/04_approval_by_employment.png).
- **Other category rates also differ, but should not be over-interpreted.**
  Approval was 339/489 (69.33%) for male applicants and 75/112 (66.96%) for
  female applicants; 285/398 (71.61%) for married and 134/213 (62.91%) for
  unmarried applicants; and 340/480 (70.83%) for graduates versus 82/134
  (61.19%) for non-graduates. By dependents, observed rates were 238/345
  (68.99%) for `0`, 66/102 (64.71%) for `1`, 76/101 (75.25%) for `2`, and
  33/51 (64.71%) for `3+`. These unadjusted rates do not isolate the effect of
  any single attribute. The missing-group counts are small (for example,
  only 3 records have missing marital status), so rates for those groups are
  particularly unstable.

## Numerical features and target relationships

The income and loan histograms are strongly right-skewed and overlap across
approved and rejected cases. Group medians and interquartile ranges offer a
more representative comparison than means alone:

| Feature | Rejected: n; median (Q1–Q3) | Approved: n; median (Q1–Q3) |
|---|---:|---:|
| Applicant income | 192; 3,833.5 (2,885–5,861.25) | 422; 3,812.5 (2,877.5–5,771.5) |
| Co-applicant income | 192; 268 (0–2,273.75) | 422; 1,239.5 (0–2,297.25) |
| Loan amount | 181; 129 (100–176) | 411; 126 (100–161) |
| Loan term (months) | 186; 360 (360–360) | 414; 360 (360–360) |

Applicant-income medians and quartiles are nearly the same by outcome in this
sample. The co-applicant-income median differs, but both groups have a Q1 of
zero and overlapping upper quartiles; the observed distribution is not a
standalone decision rule. Loan-amount quartiles overlap substantially, and
the loan-term quartiles are all 360 months in both outcome groups. These
summaries do not imply that these variables are useless in combination with
other information. See
[income, loan amount, and term distributions](figures/03_income_loan_distributions.png).

The numerical correlation heatmap reports a Pearson correlation of **0.57**
between applicant income and loan amount and **0.19** between co-applicant
income and loan amount; the other off-diagonal numeric correlations are small
in magnitude (absolute value at most 0.12). The binary credit-history value
is strongly associated with approval in the group rates above. Pearson
correlation between numeric inputs is not a test of causal relationships,
conditional effects, predictive value, or fairness. See the
[numerical correlation heatmap](figures/05_numerical_correlation.png).

## Data quality and interpretation limitations

- The quality audit found **no duplicate rows or duplicate IDs**, no
  cross-file ID overlap, no invalid observed categories or target labels, and
  no negative or non-finite numeric values in the acquired files.
- The labeled data has missing values in `Credit_History` (50/614, 8.14%),
  `Self_Employed` (32/614, 5.21%), `LoanAmount` (22/614, 3.58%),
  `Dependents` (15/614, 2.44%), `Gender` (13/614, 2.12%),
  `Loan_Amount_Term` (14/614, 2.28%), and `Married` (3/614, 0.49%).
  Differences in missingness can affect group-rate comparisons; the data does
  not establish why values are missing.
- The sample is small and imbalanced, and the source documentation does not
  establish that it represents current applicants or lending practices.
  There is no information here about decision timing, policy changes, reasons
  for decisions, or sampling/selection processes.
- Income and loan amounts have long right tails. The audit flagged 50
  applicant-income, 18 co-applicant-income, and 39 observed loan-amount values
  using IQR fences; these are statistical screening flags, not proven errors.
  Loan-term IQR fences collapse at 360 months because its quartiles are all
  360, so its 88 IQR flags do not establish invalidity. Do not remove or cap
  these observations solely because of these flags.
- The group comparisons are unadjusted and no confidence intervals or
  significance tests are reported. Small observed gaps may be sampling
  variation; larger ones may still be confounded. Neither statistical
  association nor a future model's predictive performance would establish
  causality.

## Feature usefulness and redundancy considerations

- `Credit_History` is a strong candidate for evaluation because its observed
  outcome rates differ markedly (79.58% versus 7.87%); its 50 missing values
  also require careful treatment. Its observed association should not be
  interpreted as a complete explanation or a policy recommendation.
- `Property_Area`, the income fields, `LoanAmount`, and possibly other
  applicant attributes are reasonable candidates for a model comparison.
  Their descriptive associations are not proof of incremental predictive
  benefit; that must be established using training-fold validation only.
- **Exclude `Loan_ID` from predictors.** It is a record identifier, not an
  applicant characteristic, and can let a model exploit arbitrary row-level
  patterns. Keep it only for record tracking where needed.
- The observed applicant-income/loan-amount correlation (0.57) indicates some
  relationship, but is not by itself evidence that either feature is
  redundant. No strong pairwise numeric redundancy is demonstrated by the
  heatmap. A total-income feature formed from applicant and co-applicant
  income could be explored as an additional engineered feature, but should
  not replace its components without training-only validation.
- `Credit_History` is a binary category rather than a continuous measurement;
  treat it as categorical (including its missingness strategy) rather than
  assuming that scaling it gives it a meaningful continuous interpretation.
  `Dependents` is recorded as text categories including `3+`; do not treat it
  as an ordinary numeric count without an explicit decision about the open
  ended category.

## Preprocessing and evaluation recommendations

These are recommendations for a future modeling phase, not transformations
already applied:

1. Split the labeled data into stratified training and untouched holdout
   partitions before fitting any imputer, encoder, scaler, feature selector,
   or resampler. Use cross-validation only within the training partition for
   comparisons and tuning.
2. Use a scikit-learn `Pipeline` and `ColumnTransformer`. Fit numeric
   imputation and any scaling or transformation only on training folds; fit
   categorical imputation and encoding there as well. Consider preserving
   missingness indicators for fields with material missing rates, especially
   credit history, but retain them only if training-fold validation supports
   them.
3. One-hot encode nominal fields. Evaluate whether `Dependents` should be
   encoded categorically or with a documented ordinal mapping; the measured
   rates (64.71% to 75.25% among non-missing groups) are not monotonic, so an
   ordinal assumption is not established by this EDA.
4. Income and loan distributions are skewed; compare an untransformed baseline
   with a training-fold-fitted transformation or robust scaling where
   appropriate. Scaling is generally estimator-dependent. Do not use
   full-dataset statistics to define transformations, and do not drop IQR
   flags without source-based evidence.
5. Report class-sensitive metrics such as balanced accuracy, per-class
   precision/recall, confusion matrix, and ROC-AUC (and consider
   precision-recall analysis) alongside accuracy. If class weighting or
   resampling is evaluated, apply it only inside training folds. Do not select
   decision thresholds based on the holdout.

## Potential demographic bias and safeguards

The dataset contains a gender imbalance in representation: 489 male records
(79.64%), 112 female records (18.24%), and 13 records with missing gender
(2.12%). The observed approval rates are 69.33% for male and 66.96% for
female applicants, a **2.37 percentage-point** difference in this sample.
This comparison is unadjusted; the data is not enough to determine whether
the difference is statistically or practically meaningful, caused by gender,
or indicative of discrimination. Other group-rate differences (including by
property area, education, and marital status) likewise cannot establish
fairness or unfairness.

Before any real-world use, assess group representation, missingness, error
rates, calibration, and outcomes across relevant demographic and intersectional
groups on appropriately held-out data; review legal and domain requirements;
and investigate data provenance and decision processes. Do not use these
descriptive rates as automated rules. This project remains an academic
decision-support prototype, not an autonomous lending decision engine.
