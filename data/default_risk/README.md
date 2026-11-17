# Default-risk data directory

This directory is reserved for the independent loan default and credit risk module.

Current status: the dataset has not been downloaded and validated, so no model
training or predictions are being generated here.

Required contents once the dataset is available:
- `raw/`: original LendingClub or Give Me Some Credit files
- `processed/`: cleaned, validation-ready data for modeling

Important:
- Do not overwrite the existing approval-history dataset under `data/raw/`.
- Keep the default-risk data separate from the historical Loan_Status project.
- Do not create synthetic or fabricated default labels.
