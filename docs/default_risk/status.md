# Default-risk module status

## Scope

The existing project is intentionally scoped to a historical `Loan_Status`
approval classifier. The separate default-risk module is not trained until a
reliable repayment/default dataset is acquired and validated.

## Primary dataset

- LendingClub Loan Data
- Source: https://www.kaggle.com/datasets/wordsforthewise/lending-club
- Kaggle metadata version: 3; last update: 2019-04-10
- Kaggle-declared license: CC0: Public Domain (upstream terms not independently reviewed)
- Status: user-provided archive is present and audited
- Verified archive: `data/default_risk/raw/accepted_2007_to_2018Q4.csv.gz`
- SHA-256: `55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d`
- Verified row count: 2,260,701, including 33 malformed trailing summary rows
- Verified resolved classes: 268,559 `Charged Off`, 1,076,751 `Fully Paid`
- Verified duplicate loan IDs: 0
- Acquisition steps: see [dataset acquisition instructions](dataset_acquisition.md)

## Alternative dataset

- Give Me Some Credit
- Source: https://www.kaggle.com/competitions/GiveMeSomeCredit
- Status: not downloaded in this workspace

## Mandatory stop condition

A model must not be trained until the dataset is available and the target is
validated. The required steps are:

1. Download one dataset only.
2. Verify files, schemas, and licensing conditions.
3. Confirm the target variable and class definitions.
4. Exclude unresolved outcomes and post-origination leakage features.
5. Train and evaluate only after a valid train/test split.

## Current status for this repository

Training status depends on the registered artifact. The implementation uses
only exact `Charged Off` and `Fully Paid` labels. Other states are excluded.
See the actual evaluation report after training; no fabricated labels or
metrics are included.
