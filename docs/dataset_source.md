# Dataset source and acquisition report

## Source and acquisition

- Dataset: [Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- Publisher/account: `altruistdelhite04` (as indicated by the dataset URL)
- Acquired: 2026-11-14 from the public Kaggle dataset download endpoint
- Authentication: no Kaggle credentials were supplied or used
- License/usage terms: not independently verified; consult Kaggle before
  redistribution or reuse

The downloaded ZIP passed its integrity check and contained separate CSVs.
The files are stored under the git-ignored `data/raw/` directory:

- `train_u6lujuX_CVtuZ9i.csv`: labeled data
- `test_Y3wMUE5_7gLdaTN.csv`: unlabeled data

The configured label contract (`Loan_Status` with `Y`/`N`) and identifier
(`Loan_ID`) were checked against both files. The loader keeps the two frames
separate, verifies matching feature names, and checks for identifier overlap.
No rows were combined, transformed, or used to train a model.

## Measured profile

The full machine-readable schema, dtypes, missingness, unique-value counts,
and per-dataset metrics are in
[`artifacts/reports/dataset_profile.json`](../artifacts/reports/dataset_profile.json).

| Dataset | Rows | Columns | Duplicate rows | Target |
|---|---:|---:|---:|---|
| Labeled training CSV | 614 | 13 | 0 | `Loan_Status` |
| Unlabeled test CSV | 367 | 12 | 0 | Not present |

The labeled CSV schema and observed pandas dtypes are:
`Loan_ID` (string), `Gender` (string), `Married` (string),
`Dependents` (string), `Education` (string), `Self_Employed` (string),
`ApplicantIncome` (int64), `CoapplicantIncome` (float64),
`LoanAmount` (float64), `Loan_Amount_Term` (float64),
`Credit_History` (float64), `Property_Area` (string), and
`Loan_Status` (string). The unlabeled CSV has the same feature columns but no
target; its observed `CoapplicantIncome` dtype is int64. Dtypes are reported
as read and have not been standardized.

Target distribution in labeled training data:

| `Loan_Status` | Count | Proportion |
|---|---:|---:|
| `Y` | 422 | 68.7296% |
| `N` | 192 | 31.2704% |

`Loan_ID` is the identified record identifier. In the labeled data, missing
values were observed in `Gender` (13), `Married` (3), `Dependents` (15),
`Self_Employed` (32), `LoanAmount` (22), `Loan_Amount_Term` (14), and
`Credit_History` (50). In the unlabeled data, missing values were observed in
`Gender` (11), `Dependents` (10), `Self_Employed` (23), `LoanAmount` (5),
`Loan_Amount_Term` (6), and `Credit_History` (29). These are descriptive
counts only; no imputation or other processing was performed.

The unlabeled CSV has no target values, so its target distribution is recorded
as `null` in the profile. No target distribution or prediction is inferred
for it.
