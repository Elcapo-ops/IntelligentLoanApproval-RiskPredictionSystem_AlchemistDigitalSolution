# LendingClub dataset audit

- Source: https://www.kaggle.com/datasets/wordsforthewise/lending-club
- File: `accepted_2007_to_2018Q4.csv.gz`
- SHA-256: `55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d`
- File size: 392,582,231 bytes
- Rows: 2,260,701
- Columns: 151
- Issue-date coverage: 2007-06 to 2018-12
- Duplicate IDs: 0
- Malformed summary rows excluded: 33
- Unresolved statuses excluded: 915,358

## Target status counts

| Status | Rows |
|---|---:|
| <MISSING> | 33 |
| Charged Off | 268,559 |
| Current | 878,317 |
| Default | 40 |
| Does not meet the credit policy. Status:Charged Off | 761 |
| Does not meet the credit policy. Status:Fully Paid | 1,988 |
| Fully Paid | 1,076,751 |
| In Grace Period | 8,436 |
| Late (16-30 days) | 4,349 |
| Late (31-120 days) | 21,467 |

## Binary target rows

- Default (`Charged Off`): 268,559
- Non-default (`Fully Paid`): 1,076,751

Only these exact matured statuses are retained for the initial target. Current, late, grace-period, `Default`, policy-exception, missing, and other labels are excluded.

## Missing values in inspected columns

| Column | Missing rows |
|---|---:|
| `acc_now_delinq` | 62 |
| `acc_open_past_24mths` | 50,063 |
| `addr_state` | 33 |
| `all_util` | 866,381 |
| `annual_inc` | 37 |
| `annual_inc_joint` | 2,139,991 |
| `application_type` | 33 |
| `avg_cur_bal` | 70,379 |
| `bc_open_to_buy` | 74,968 |
| `bc_util` | 76,104 |
| `chargeoff_within_12_mths` | 178 |
| `collection_recovery_fee` | 33 |
| `collections_12_mths_ex_med` | 178 |
| `debt_settlement_flag` | 33 |
| `debt_settlement_flag_date` | 2,226,455 |
| `deferral_term` | 2,249,784 |
| `delinq_2yrs` | 62 |
| `delinq_amnt` | 62 |
| `desc` | 2,134,636 |
| `disbursement_method` | 33 |
| `dti` | 1,744 |
| `dti_joint` | 2,139,995 |
| `earliest_cr_line` | 62 |
| `emp_length` | 146,940 |
| `emp_title` | 167,002 |
| `fico_range_high` | 33 |
| `fico_range_low` | 33 |
| `funded_amnt` | 33 |
| `funded_amnt_inv` | 33 |
| `grade` | 33 |
| `hardship_amount` | 2,249,784 |
| `hardship_dpd` | 2,249,784 |
| `hardship_end_date` | 2,249,784 |
| `hardship_flag` | 33 |
| `hardship_last_payment_amount` | 2,249,784 |
| `hardship_length` | 2,249,784 |
| `hardship_loan_status` | 2,249,784 |
| `hardship_payoff_balance_amount` | 2,249,784 |
| `hardship_reason` | 2,249,784 |
| `hardship_start_date` | 2,249,784 |
| `hardship_status` | 2,249,784 |
| `hardship_type` | 2,249,784 |
| `home_ownership` | 33 |
| `id` | 0 |
| `il_util` | 1,068,883 |
| `initial_list_status` | 33 |
| `inq_fi` | 866,162 |
| `inq_last_12m` | 866,163 |
| `inq_last_6mths` | 63 |
| `installment` | 33 |
| `int_rate` | 33 |
| `issue_d` | 33 |
| `last_credit_pull_d` | 105 |
| `last_fico_range_high` | 33 |
| `last_fico_range_low` | 33 |
| `last_pymnt_amnt` | 33 |
| `last_pymnt_d` | 2,460 |
| `loan_amnt` | 33 |
| `loan_status` | 33 |
| `max_bal_bc` | 866,162 |
| `member_id` | 2,260,701 |
| `mo_sin_old_il_acct` | 139,104 |
| `mo_sin_old_rev_tl_op` | 70,310 |
| `mo_sin_rcnt_rev_tl_op` | 70,310 |
| `mo_sin_rcnt_tl` | 70,309 |
| `mort_acc` | 50,063 |
| `mths_since_last_delinq` | 1,158,535 |
| `mths_since_last_major_derog` | 1,679,926 |
| `mths_since_last_record` | 1,901,545 |
| `mths_since_rcnt_il` | 909,957 |
| `mths_since_recent_bc` | 73,445 |
| `mths_since_recent_bc_dlq` | 1,741,000 |
| `mths_since_recent_inq` | 295,468 |
| `mths_since_recent_revol_delinq` | 1,520,342 |
| `next_pymnt_d` | 1,345,343 |
| `num_accts_ever_120_pd` | 70,309 |
| `num_actv_bc_tl` | 70,309 |
| `num_actv_rev_tl` | 70,309 |
| `num_bc_sats` | 58,623 |
| `num_bc_tl` | 70,309 |
| `num_il_tl` | 70,309 |
| `num_op_rev_tl` | 70,309 |
| `num_rev_accts` | 70,310 |
| `num_rev_tl_bal_gt_0` | 70,309 |
| `num_sats` | 58,623 |
| `num_tl_120dpd_2m` | 153,690 |
| `num_tl_30dpd` | 70,309 |
| `num_tl_90g_dpd_24m` | 70,309 |
| `num_tl_op_past_12m` | 70,309 |
| `open_acc` | 62 |
| `open_acc_6m` | 866,163 |
| `open_act_il` | 866,162 |
| `open_il_12m` | 866,162 |
| `open_il_24m` | 866,162 |
| `open_rv_12m` | 866,162 |
| `open_rv_24m` | 866,162 |
| `orig_projected_additional_accrued_interest` | 2,252,050 |
| `out_prncp` | 33 |
| `out_prncp_inv` | 33 |
| `payment_plan_start_date` | 2,249,784 |
| `pct_tl_nvr_dlq` | 70,464 |
| `percent_bc_gt_75` | 75,412 |
| `policy_code` | 33 |
| `pub_rec` | 62 |
| `pub_rec_bankruptcies` | 1,398 |
| `purpose` | 33 |
| `pymnt_plan` | 33 |
| `recoveries` | 33 |
| `revol_bal` | 33 |
| `revol_bal_joint` | 2,152,681 |
| `revol_util` | 1,835 |
| `sec_app_chargeoff_within_12_mths` | 2,152,680 |
| `sec_app_collections_12_mths_ex_med` | 2,152,680 |
| `sec_app_earliest_cr_line` | 2,152,680 |
| `sec_app_fico_range_high` | 2,152,680 |
| `sec_app_fico_range_low` | 2,152,680 |
| `sec_app_inq_last_6mths` | 2,152,680 |
| `sec_app_mort_acc` | 2,152,680 |
| `sec_app_mths_since_last_major_derog` | 2,224,759 |
| `sec_app_num_rev_accts` | 2,152,680 |
| `sec_app_open_acc` | 2,152,680 |
| `sec_app_open_act_il` | 2,152,680 |
| `sec_app_revol_util` | 2,154,517 |
| `settlement_amount` | 2,226,455 |
| `settlement_date` | 2,226,455 |
| `settlement_percentage` | 2,226,455 |
| `settlement_status` | 2,226,455 |
| `settlement_term` | 2,226,455 |
| `sub_grade` | 33 |
| `tax_liens` | 138 |
| `term` | 33 |
| `title` | 23,359 |
| `tot_coll_amt` | 70,309 |
| `tot_cur_bal` | 70,309 |
| `tot_hi_cred_lim` | 70,309 |
| `total_acc` | 62 |
| `total_bal_ex_mort` | 50,063 |
| `total_bal_il` | 866,162 |
| `total_bc_limit` | 50,063 |
| `total_cu_tl` | 866,163 |
| `total_il_high_credit_limit` | 70,309 |
| `total_pymnt` | 33 |
| `total_pymnt_inv` | 33 |
| `total_rec_int` | 33 |
| `total_rec_late_fee` | 33 |
| `total_rec_prncp` | 33 |
| `total_rev_hi_lim` | 70,309 |
| `url` | 33 |
| `verification_status` | 33 |
| `verification_status_joint` | 2,144,971 |
| `zip_code` | 34 |

## Important limitation

The dataset snapshot's outcome observation cutoff and censoring process are not sufficiently documented to treat unresolved loans as non-default. They are excluded. The resulting target is classification among rows already labeled Fully Paid or Charged Off, not a fully observed fixed horizon default incidence estimate.
