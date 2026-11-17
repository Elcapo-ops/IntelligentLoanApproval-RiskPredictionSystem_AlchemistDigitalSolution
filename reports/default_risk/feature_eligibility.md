# LendingClub feature eligibility policy

Prediction time is defined as loan application/origination using the loan's reported application and bureau attributes. The cohort issue date is used only for chronological splitting and to calculate credit-history length; issue year/month are not model predictors.

## Included source fields

- `loan_amnt`
- `term`
- `emp_length`
- `home_ownership`
- `annual_inc`
- `verification_status`
- `purpose`
- `addr_state`
- `dti`
- `delinq_2yrs`
- `fico_range_low`
- `fico_range_high`
- `inq_last_6mths`
- `open_acc`
- `pub_rec`
- `revol_bal`
- `revol_util`
- `total_acc`
- `application_type`
- `earliest_cr_line` is used only with `issue_d` to derive `credit_history_months`; the raw date is not a feature.

## Excluded fields and reasons

### Identifiers, links, location granularity, and free text

Excluded to prevent record memorization, unnecessary personal data exposure, or unbounded text features:
- `desc`
- `emp_title`
- `member_id`
- `title`
- `url`
- `zip_code`

### Post-origination outcomes or servicing information

Excluded because these can reveal subsequent payment performance, delinquency, hardship, recovery, collection, or settlement activity:
- `collection_recovery_fee`
- `debt_settlement_flag`
- `debt_settlement_flag_date`
- `deferral_term`
- `hardship_amount`
- `hardship_dpd`
- `hardship_end_date`
- `hardship_flag`
- `hardship_last_payment_amount`
- `hardship_length`
- `hardship_loan_status`
- `hardship_payoff_balance_amount`
- `hardship_reason`
- `hardship_start_date`
- `hardship_status`
- `hardship_type`
- `last_credit_pull_d`
- `last_fico_range_high`
- `last_fico_range_low`
- `last_pymnt_amnt`
- `last_pymnt_d`
- `loan_status`
- `next_pymnt_d`
- `orig_projected_additional_accrued_interest`
- `out_prncp`
- `out_prncp_inv`
- `payment_plan_start_date`
- `recoveries`
- `settlement_amount`
- `settlement_date`
- `settlement_percentage`
- `settlement_status`
- `settlement_term`
- `total_pymnt`
- `total_pymnt_inv`
- `total_rec_int`
- `total_rec_late_fee`
- `total_rec_prncp`

### Underwriting or funded-loan outputs

Excluded because these are outcomes of underwriting or loan issuance rather than pre-decision applicant inputs:
- `funded_amnt`
- `funded_amnt_inv`
- `grade`
- `initial_list_status`
- `installment`
- `int_rate`
- `policy_code`
- `pymnt_plan`
- `sub_grade`

### Other source fields not selected

These fields are omitted because they were not needed in the minimal application-time feature set or their application-time availability, meaning, or reliability was not validated for this initial model:
- `acc_now_delinq`
- `acc_open_past_24mths`
- `all_util`
- `annual_inc_joint`
- `avg_cur_bal`
- `bc_open_to_buy`
- `bc_util`
- `chargeoff_within_12_mths`
- `collections_12_mths_ex_med`
- `delinq_amnt`
- `disbursement_method`
- `dti_joint`
- `il_util`
- `inq_fi`
- `inq_last_12m`
- `max_bal_bc`
- `mo_sin_old_il_acct`
- `mo_sin_old_rev_tl_op`
- `mo_sin_rcnt_rev_tl_op`
- `mo_sin_rcnt_tl`
- `mort_acc`
- `mths_since_last_delinq`
- `mths_since_last_major_derog`
- `mths_since_last_record`
- `mths_since_rcnt_il`
- `mths_since_recent_bc`
- `mths_since_recent_bc_dlq`
- `mths_since_recent_inq`
- `mths_since_recent_revol_delinq`
- `num_accts_ever_120_pd`
- `num_actv_bc_tl`
- `num_actv_rev_tl`
- `num_bc_sats`
- `num_bc_tl`
- `num_il_tl`
- `num_op_rev_tl`
- `num_rev_accts`
- `num_rev_tl_bal_gt_0`
- `num_sats`
- `num_tl_120dpd_2m`
- `num_tl_30dpd`
- `num_tl_90g_dpd_24m`
- `num_tl_op_past_12m`
- `open_acc_6m`
- `open_act_il`
- `open_il_12m`
- `open_il_24m`
- `open_rv_12m`
- `open_rv_24m`
- `pct_tl_nvr_dlq`
- `percent_bc_gt_75`
- `pub_rec_bankruptcies`
- `revol_bal_joint`
- `sec_app_chargeoff_within_12_mths`
- `sec_app_collections_12_mths_ex_med`
- `sec_app_earliest_cr_line`
- `sec_app_fico_range_high`
- `sec_app_fico_range_low`
- `sec_app_inq_last_6mths`
- `sec_app_mort_acc`
- `sec_app_mths_since_last_major_derog`
- `sec_app_num_rev_accts`
- `sec_app_open_acc`
- `sec_app_open_act_il`
- `sec_app_revol_util`
- `tax_liens`
- `tot_coll_amt`
- `tot_cur_bal`
- `tot_hi_cred_lim`
- `total_bal_ex_mort`
- `total_bal_il`
- `total_bc_limit`
- `total_cu_tl`
- `total_il_high_credit_limit`
- `total_rev_hi_lim`
- `verification_status_joint`

## Target fields

- `loan_status` is used only to define the target, never as a feature.
- `issue_d` is used only for the time split and `earliest_cr_line` transformation; neither raw date is a model feature.
- Exact `Charged Off` rows map to default (1); exact `Fully Paid` rows map to non-default (0).
- Current, late, grace-period, `Default`, policy-exception, missing, and unknown statuses are not mapped to either class.

Feature importance is descriptive predictive contribution in this historical sample, not causation or a lending rationale.
