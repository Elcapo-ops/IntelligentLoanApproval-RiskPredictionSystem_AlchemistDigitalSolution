# LendingClub dataset acquisition

## Source review

- Dataset: [All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- Owner: `wordsforthewise`
- Kaggle dataset version reported by its public metadata API: 3
- Last update reported by Kaggle: 2019-04-10
- Kaggle-declared license: CC0: Public Domain
- Dataset description: LendingClub accepted and rejected loans; accepted records include loan performance fields.
- Kaggle's published size: 648,053,013 bytes for the dataset. Its file listing includes `accepted_2007_to_2018Q4.csv.gz` (392,582,231 bytes), the uncompressed accepted CSV (1,675,133,810 bytes), and rejected-loan files.

The Kaggle page also notes that the source data came from LendingClub and warns
that its availability could be affected by LendingClub's terms. The Kaggle
license declaration is recorded, but upstream legal terms have not been
independently reviewed. Do not redistribute the dataset without checking the
applicable terms.

## Acquisition in this workspace

The user supplied `accepted_2007_to_2018Q4.csv.gz` in
`data/default_risk/raw/`. The public dataset metadata API was checked on
2026-11-15. An earlier unauthenticated download attempt returned HTTP 404,
but the supplied local file was subsequently found and audited. No Kaggle
credentials were requested, read, or exposed.

Verified file facts:

- Compressed size: 392,582,231 bytes
- SHA-256: `55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d`
- Parsed rows: 2,260,701 (including 33 trailing summary rows with no loan status/date)
- Columns: 151
- Exact matured target statuses: 1,076,751 `Fully Paid`, 268,559 `Charged Off`
- Unique IDs: 2,260,701; duplicate IDs: 0
- Issue date range: June 2007 through December 2018

The reproducible audit is saved in `reports/default_risk/lendingclub_dataset_audit.md`
and its JSON companion. The acquired source file remains unchanged.

## Download instructions

Choose **one dataset only**. This implementation uses only the accepted-loans
file. For a fresh checkout, download that file from the Kaggle page and place it here:

```text
data/default_risk/raw/accepted_2007_to_2018Q4.csv.gz
```

Alternatively, if you have already configured Kaggle credentials locally, use
the Kaggle CLI without putting credentials in source code or command history:

```powershell
kaggle datasets download -d wordsforthewise/lending-club `
  -f accepted_2007_to_2018Q4.csv.gz `
  -p data/default_risk/raw
```

The CLI must be authenticated using credentials configured on the local
machine. Never paste Kaggle tokens into this repository or share them in chat.

After download, retain the original compressed file in `raw/`. The audit
workflow verifies its archive integrity, records the SHA-256, inspects all
columns and `loan_status` values, validates dates and duplicate IDs, and
generates the dataset audit report. Do not treat file presence alone as
evidence that it is valid.

## Target and leakage gate

The proposed initial target is restricted to matured historical outcomes:

- `Charged Off` -> default
- `Fully Paid` -> non-default
- Current, late, grace-period, and other unresolved statuses -> excluded

This mapping must be checked against the actual downloaded values before use.
Post-origination fields (including status, payments, recoveries, collections,
charge-off details, and future delinquency) must not enter the application-time
feature set. No model is to be trained until this audit passes.
