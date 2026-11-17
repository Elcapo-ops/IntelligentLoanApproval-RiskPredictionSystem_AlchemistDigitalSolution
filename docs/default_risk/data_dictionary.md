# Default-Risk Data Dictionary

This document defines every field used by the Loan Default / Credit Risk Prediction
module, including model inputs, target construction, derived features, and excluded
fields with reasons.

## Model Input Features (20 total)

All features are restricted to information available at loan application/origination.
No post-origination outcomes, underwriting outputs, identifiers, or free-text fields
are used as model predictors.

### Numeric Features (13)

| Feature | Type | Description | Valid Range | Source |
|---------|------|-------------|-------------|--------|
| `loan_amnt` | float | Loan amount requested by the applicant | >= 0 | Application input |
| `annual_inc` | float | Self-reported annual income | >= 0 | Application input |
| `dti` | float | Debt-to-income ratio | 0 to 200 | Application input |
| `delinq_2yrs` | int | Number of delinquencies in past 2 years | >= 0 | Bureau attribute |
| `fico_range_low` | int | Lower bound of FICO score range | 300 to 850 | Bureau attribute |
| `fico_range_high` | int | Upper bound of FICO score range | 300 to 850 | Bureau attribute |
| `inq_last_6mths` | int | Credit inquiries in last 6 months | >= 0 | Bureau attribute |
| `open_acc` | int | Number of open credit lines | >= 0 | Bureau attribute |
| `pub_rec` | int | Number of public records | >= 0 | Bureau attribute |
| `revol_bal` | float | Revolving balance | >= 0 | Bureau attribute |
| `revol_util` | float | Revolving line utilization rate | 0 to 200 | Bureau attribute |
| `total_acc` | int | Total number of credit lines | >= 0 | Bureau attribute |
| `credit_history_months` | int | Months between earliest credit line and issue date | >= 0 | Derived |

### Categorical Features (7)

| Feature | Type | Description | Example Values | Source |
|---------|------|-------------|----------------|--------|
| `term` | categorical | Loan term | "36 months", "60 months" | Application input |
| `emp_length` | categorical | Employment length | "< 1 year", "5 years", "10+ years" | Application input |
| `home_ownership` | categorical | Home ownership status | "RENT", "OWN", "MORTGAGE", "OTHER" | Application input |
| `verification_status` | categorical | Income verification status | "Verified", "Source Verified", "Not Verified" | Application input |
| `purpose` | categorical | Loan purpose | "debt_consolidation", "credit_card", ... | Application input |
| `addr_state` | categorical | Applicant state (2-letter code) | "CA", "NY", "TX", ... | Application input |
| `application_type` | categorical | Application type | "INDIVIDUAL", "JOINT" | Application input |

### Derived Feature

| Feature | Derivation | Notes |
|---------|------------|-------|
| `credit_history_months` | `(issue_d year-month) - (earliest_cr_line year-month)` | Computed only when both dates are valid; neither raw date is a model feature |

## Target Variable

| Field | Type | Description |
|-------|------|-------------|
| `loan_status` | categorical (source) | LendingClub loan outcome status |
| `default_target` | int (derived) | Binary default target: 1 = default, 0 = non-default |

### Target Mapping

| `loan_status` value | `default_target` | Class |
|---------------------|------------------|-------|
| `Charged Off` | 1 | Default |
| `Fully Paid` | 0 | Non-default |

### Excluded Statuses (never mapped to either class)

- `Current`
- `Late (16-30 days)`
- `Late (31-120 days)`
- `In Grace Period`
- `Default`
- `Does not meet the credit policy. Status:Fully Paid`
- `Does not meet the credit policy. Status:Charged Off`
- Missing / blank / unknown

## Excluded Fields

### Identifiers and Free Text

Excluded to prevent record memorization, unnecessary personal data exposure, or
unbounded text features:

- `id`, `member_id`, `url`, `desc`, `emp_title`, `title`, `zip_code`

### Post-Origination Outcomes and Servicing Information

Excluded because these can reveal subsequent payment performance, delinquency,
hardship, recovery, collection, or settlement activity:

- `collection_recovery_fee`, `debt_settlement_flag`, `debt_settlement_flag_date`,
  `deferral_term`, `hardship_amount`, `hardship_dpd`, `hardship_end_date`,
  `hardship_flag`, `hardship_last_payment_amount`, `hardship_length`,
  `hardship_loan_status`, `hardship_payoff_balance_amount`, `hardship_reason`,
  `hardship_start_date`, `hardship_status`, `hardship_type`,
  `last_credit_pull_d`, `last_fico_range_high`, `last_fico_range_low`,
  `last_pymnt_amnt`, `last_pymnt_d`, `loan_status`, `next_pymnt_d`,
  `orig_projected_additional_accrued_interest`, `out_prncp`, `out_prncp_inv`,
  `payment_plan_start_date`, `recoveries`, `settlement_amount`, `settlement_date`,
  `settlement_percentage`, `settlement_status`, `settlement_term`,
  `total_pymnt`, `total_pymnt_inv`, `total_rec_int`, `total_rec_late_fee`,
  `total_rec_prncp`

### Underwriting or Funded-Loan Outputs

Excluded because these are outcomes of underwriting or loan issuance rather than
pre-decision applicant inputs:

- `funded_amnt`, `funded_amnt_inv`, `grade`, `initial_list_status`, `installment`,
  `int_rate`, `policy_code`, `pymnt_plan`, `sub_grade`

### Other Source Fields Not Selected

Omitted because they were not needed in the minimal application-time feature set or
their application-time availability, meaning, or reliability was not validated for
this initial model (full list in `reports/default_risk/feature_eligibility.md`):

Examples include `acc_now_delinq`, `all_util`, `annual_inc_joint`, `bc_util`,
`chargeoff_within_12_mths`, `collections_12_mths_ex_med`, `delinq_amnt`,
`dti_joint`, `il_util`, `inq_fi`, `inq_last_12m`, `max_bal_bc`,
`mort_acc`, `mths_since_last_delinq`, `mths_since_last_major_derog`,
`mths_since_last_record`, `mths_since_rcnt_il`, `mths_since_recent_bc`,
`mths_since_recent_bc_dlq`, `mths_since_recent_inq`,
`mths_since_recent_revol_delinq`, `num_accts_ever_120_pd`, `num_actv_bc_tl`,
`num_actv_rev_tl`, `num_bc_sats`, `num_bc_tl`, `num_il_tl`, `num_op_rev_tl`,
`num_rev_accts`, `num_rev_tl_bal_gt_0`, `num_sats`, `num_tl_120dpd_2m`,
`num_tl_30dpd`, `num_tl_90g_dpd_24m`, `num_tl_op_past_12m`, `open_acc_6m`,
`open_act_il`, `open_il_12m`, `open_il_24m`, `open_rv_12m`, `open_rv_24m`,
`pct_tl_nvr_dlq`, `percent_bc_gt_75`, `pub_rec_bankruptcies`, `revol_bal_joint`,
`sec_app_*`, `tax_liens`, `tot_coll_amt`, `tot_cur_bal`, `tot_hi_cred_lim`,
`total_bal_ex_mort`, `total_bal_il`, `total_bc_limit`, `total_cu_tl`,
`total_il_high_credit_limit`, `total_rev_hi_lim`, `verification_status_joint`

### Target and Date Fields (Not Model Features)

| Field | Use | Not a feature because |
|-------|-----|----------------------|
| `loan_status` | Target definition only | Target leakage if used as input |
| `issue_d` | Temporal split; `credit_history_months` derivation | Raw date not a predictor |
| `earliest_cr_line` | `credit_history_months` derivation only | Raw date not a predictor |

## Feature Eligibility Report

A machine-readable CSV version of this eligibility policy is available at:

`data/default_risk/reports/feature_eligibility_report.csv`

Columns: `feature_name`, `data_type`, `description`, `eligible`, `reason`,
`available_at_origination`, `leakage_risk`

## Data Quality and Constraints

- All numeric features are validated as finite numbers during inference.
- `dti` and `revol_util` are constrained to [0, 200].
- `fico_range_low` and `fico_range_high` are constrained to [300, 850].
- All other numeric features must be non-negative.
- Categorical features accept any string; unseen categories are handled by
  `OneHotEncoder(handle_unknown="ignore")` at inference time.
- Missing values are imputed during preprocessing (median for numeric, most-frequent
  for categorical).
- The trained model expects exactly the 20 feature columns listed above, in the
  schema order recorded in the model metadata.

## Source Dataset

- **Name:** LendingClub Loan Data
- **File:** `accepted_2007_to_2018Q4.csv.gz`
- **Location:** `data/default_risk/raw/accepted_2007_to_2018Q4.csv.gz`
- **Source:** https://www.kaggle.com/datasets/wordsforthewise/lending-club
- **License:** CC0: Public Domain (as declared on Kaggle)
- **Coverage:** Loans issued 2007 through Q4 2018
