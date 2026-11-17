"""Dataset inspection and leakage-aware target construction for LendingClub."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils.exceptions import DataValidationError

DATASET_SOURCE = "https://www.kaggle.com/datasets/wordsforthewise/lending-club"
DATASET_FILENAME = "accepted_2007_to_2018Q4.csv.gz"
TARGET_COLUMN = "loan_status"
DEFAULT_LABELS = {"Charged Off"}
NON_DEFAULT_LABELS = {"Fully Paid"}
TARGET_STATUSES = DEFAULT_LABELS | NON_DEFAULT_LABELS

# These are restricted to information available at loan application/origination.
# LendingClub assigns interest rates/grades during underwriting; those fields
# are deliberately omitted to avoid predicting outcomes with decision outputs.
NUMERIC_FEATURES = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "credit_history_months",
]
CATEGORICAL_FEATURES = [
    "term",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "application_type",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RAW_FEATURE_COLUMNS = [
    "loan_amnt",
    "term",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "application_type",
]
RAW_REQUIRED_COLUMNS = ["id", "loan_status", "issue_d", *RAW_FEATURE_COLUMNS]

POST_ORIGINATION_COLUMNS = {
    "loan_status", "out_prncp", "out_prncp_inv", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date",
    "hardship_end_date", "payment_plan_start_date", "hardship_length",
    "hardship_dpd", "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage",
    "settlement_term",
}
IDENTIFIER_OR_FREE_TEXT_COLUMNS = {
    "id", "member_id", "url", "desc", "emp_title", "title", "zip_code",
}
UNDERWRITING_OUTPUT_COLUMNS = {
    "funded_amnt", "funded_amnt_inv", "int_rate", "installment", "grade",
    "sub_grade", "pymnt_plan", "initial_list_status", "policy_code",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_header(path: Path) -> list[str]:
    try:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as source:
            return list(pd.read_csv(source, nrows=0).columns)
    except (OSError, EOFError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Unable to read LendingClub CSV archive: {path.name}") from exc


def audit_lendingclub_dataset(
    path: Path,
    *,
    chunksize: int = 25_000,
) -> dict[str, Any]:
    """Profile source rows in chunks and validate the target/date/schema.

    The source is not modified. Summary rows appended to the CSV with no
    application status/date are counted as malformed and excluded.
    """
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise DataValidationError(f"Dataset file is missing or unsafe: {path}")
    if path.suffix.lower() != ".gz":
        raise DataValidationError("Expected a gzip-compressed LendingClub CSV file")
    header = _read_header(path)
    missing_columns = sorted(set(RAW_REQUIRED_COLUMNS).difference(header))
    if missing_columns:
        raise DataValidationError(
            f"LendingClub schema is missing required columns: {missing_columns}"
        )

    counts: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    monthly_outcomes: Counter[tuple[str, str]] = Counter()
    seen_ids: set[str] = set()
    duplicate_ids = 0
    row_count = 0
    malformed_rows = 0
    try:
        for chunk in pd.read_csv(
            path,
            compression="gzip",
            usecols=header,
            chunksize=chunksize,
            low_memory=False,
        ):
            row_count += len(chunk)
            missing.update(chunk.isna().sum().to_dict())
            statuses = chunk["loan_status"].fillna("<MISSING>").astype(str)
            counts.update(statuses.value_counts().to_dict())
            dates = chunk["issue_d"].fillna("<MISSING>").astype(str)
            month_groups = pd.DataFrame({"date": dates, "status": statuses}).value_counts()
            for (date, status), amount in month_groups.items():
                monthly_outcomes[(str(date), str(status))] += int(amount)
            malformed_rows += int(
                (statuses.eq("<MISSING>") | dates.eq("<MISSING>")).sum()
            )
            for identifier in chunk["id"].dropna().astype(str):
                if identifier in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(identifier)
    except (OSError, EOFError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise DataValidationError(
            f"Unable to parse LendingClub dataset: {path.name}"
        ) from exc

    if not row_count:
        raise DataValidationError("LendingClub dataset contains no records")
    resolved_count = sum(counts[label] for label in TARGET_STATUSES)
    if resolved_count == 0:
        raise DataValidationError(
            "Dataset contains no rows with the validated Fully Paid/Charged Off statuses"
        )

    valid_months = sorted(
        date
        for date, status in monthly_outcomes
        if status in TARGET_STATUSES and date != "<MISSING>"
    )
    parsed_dates = pd.to_datetime(valid_months, format="%b-%Y", errors="coerce")
    valid_date_values = parsed_dates.dropna()
    return {
        "source_url": DATASET_SOURCE,
        "dataset_file": path.name,
        "dataset_version": 3,
        "dataset_last_updated": "2019-04-10",
        "dataset_license": "CC0: Public Domain (as declared on Kaggle)",
        "acquisition_date": None,
        "local_file_modified_utc": datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).isoformat(),
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": row_count,
        "column_count": len(header),
        "columns": header,
        "missing_values": dict(sorted(missing.items())),
        "loan_status_counts": dict(sorted(counts.items())),
        "resolved_target_rows": int(resolved_count),
        "target_class_counts": {
            "default": int(counts["Charged Off"]),
            "non_default": int(counts["Fully Paid"]),
        },
        "unresolved_rows_excluded": int(row_count - malformed_rows - resolved_count),
        "malformed_summary_rows_excluded": malformed_rows,
        "duplicate_id_count": duplicate_ids,
        "unique_id_count": len(seen_ids),
        "issue_date_start": (
            valid_date_values.min().strftime("%Y-%m")
            if not valid_date_values.empty
            else None
        ),
        "issue_date_end": (
            valid_date_values.max().strftime("%Y-%m")
            if not valid_date_values.empty
            else None
        ),
        "monthly_target_counts": [
            {
                "issue_month": date,
                "loan_status": status,
                "count": amount,
            }
            for (date, status), amount in sorted(monthly_outcomes.items())
            if status in TARGET_STATUSES
        ],
    }


def prepare_lendingclub_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Map matured outcomes and create application-time predictors."""
    required = set(RAW_REQUIRED_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(f"Input data is missing columns: {missing}")
    eligible = frame.loc[frame[TARGET_COLUMN].isin(TARGET_STATUSES)].copy()
    if eligible.empty:
        raise DataValidationError("No matured Fully Paid or Charged Off outcomes found")

    issue_date = pd.to_datetime(eligible["issue_d"], format="%b-%Y", errors="coerce")
    credit_date = pd.to_datetime(
        eligible["earliest_cr_line"], format="%b-%Y", errors="coerce"
    )
    eligible["credit_history_months"] = (
        (issue_date.dt.year - credit_date.dt.year) * 12
        + issue_date.dt.month
        - credit_date.dt.month
    ).where(issue_date.notna() & credit_date.notna())
    eligible["default_target"] = eligible[TARGET_COLUMN].map(
        {"Charged Off": 1, "Fully Paid": 0}
    ).astype("int8")

    result = eligible.loc[
        :,
        [
            "id",
            "issue_d",
            "default_target",
            "credit_history_months",
            *RAW_FEATURE_COLUMNS,
        ],
    ].copy()
    for column in NUMERIC_FEATURES:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in ("term", "emp_length", *CATEGORICAL_FEATURES):
        result[column] = result[column].astype(object)
        result[column] = result[column].where(result[column].notna(), float("nan"))
    result["term"] = result["term"].str.strip()
    result["emp_length"] = result["emp_length"].str.strip()
    result["issue_d"] = issue_date
    result = result.dropna(subset=["issue_d"])
    result = result.drop(columns=["earliest_cr_line"])
    if result["default_target"].isna().any():
        raise DataValidationError("Target mapping produced missing default labels")
    return result.reset_index(drop=True)


def save_audit_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write machine-readable and human-readable audit reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "lendingclub_dataset_audit.json"
    markdown_path = output_dir / "lendingclub_dataset_audit.md"
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    markdown = [
        "# LendingClub dataset audit",
        "",
        f"- Source: {report['source_url']}",
        f"- File: `{report['dataset_file']}`",
        f"- SHA-256: `{report['sha256']}`",
        f"- File size: {report['file_size_bytes']:,} bytes",
        f"- Rows: {report['row_count']:,}",
        f"- Columns: {report['column_count']}",
        f"- Issue-date coverage: {report['issue_date_start']} to {report['issue_date_end']}",
        f"- Duplicate IDs: {report['duplicate_id_count']:,}",
        f"- Malformed summary rows excluded: {report['malformed_summary_rows_excluded']:,}",
        f"- Unresolved statuses excluded: {report['unresolved_rows_excluded']:,}",
        "",
        "## Target status counts",
        "",
        "| Status | Rows |",
        "|---|---:|",
        *[
            f"| {status} | {count:,} |"
            for status, count in report["loan_status_counts"].items()
        ],
        "",
        "## Binary target rows",
        "",
        f"- Default (`Charged Off`): {report['target_class_counts']['default']:,}",
        f"- Non-default (`Fully Paid`): {report['target_class_counts']['non_default']:,}",
        "",
        "Only these exact matured statuses are retained for the initial target. "
        "Current, late, grace-period, `Default`, policy-exception, missing, and "
        "other labels are excluded.",
        "",
        "## Missing values in inspected columns",
        "",
        "| Column | Missing rows |",
        "|---|---:|",
        *[
            f"| `{column}` | {count:,} |"
            for column, count in report["missing_values"].items()
        ],
        "",
        "## Important limitation",
        "",
        "The dataset snapshot's outcome observation cutoff and censoring process "
        "are not sufficiently documented to treat unresolved loans as non-default. "
        "They are excluded. The resulting target is classification among rows "
        "already labeled Fully Paid or Charged Off, not a fully observed fixed "
        "horizon default incidence estimate.",
        "",
    ]
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    eligibility_path = output_dir / "feature_eligibility.md"
    eligible_source = set(RAW_FEATURE_COLUMNS) - {"earliest_cr_line"}
    excluded = set(report["columns"]) - eligible_source - {
        "id",
        "issue_d",
        "earliest_cr_line",
    }
    post_origination = sorted(excluded & POST_ORIGINATION_COLUMNS)
    identifiers = sorted(excluded & IDENTIFIER_OR_FREE_TEXT_COLUMNS)
    underwriting = sorted(excluded & UNDERWRITING_OUTPUT_COLUMNS)
    other_excluded = sorted(
        excluded - set(post_origination) - set(identifiers) - set(underwriting)
    )
    eligibility = [
        "# LendingClub feature eligibility policy",
        "",
        "Prediction time is defined as loan application/origination using the loan's reported application and bureau attributes. The cohort issue date is used only for chronological splitting and to calculate credit-history length; issue year/month are not model predictors.",
        "",
        "## Included source fields",
        "",
        *[f"- `{column}`" for column in RAW_FEATURE_COLUMNS if column != "earliest_cr_line"],
        "- `earliest_cr_line` is used only with `issue_d` to derive `credit_history_months`; the raw date is not a feature.",
        "",
        "## Excluded fields and reasons",
        "",
        "### Identifiers, links, location granularity, and free text",
        "",
        "Excluded to prevent record memorization, unnecessary personal data exposure, or unbounded text features:",
        *[f"- `{column}`" for column in identifiers],
        "",
        "### Post-origination outcomes or servicing information",
        "",
        "Excluded because these can reveal subsequent payment performance, delinquency, hardship, recovery, collection, or settlement activity:",
        *[f"- `{column}`" for column in post_origination],
        "",
        "### Underwriting or funded-loan outputs",
        "",
        "Excluded because these are outcomes of underwriting or loan issuance rather than pre-decision applicant inputs:",
        *[f"- `{column}`" for column in underwriting],
        "",
        "### Other source fields not selected",
        "",
        "These fields are omitted because they were not needed in the minimal application-time feature set or their application-time availability, meaning, or reliability was not validated for this initial model:",
        *[f"- `{column}`" for column in other_excluded],
        "",
        "## Target fields",
        "",
        "- `loan_status` is used only to define the target, never as a feature.",
        "- `issue_d` is used only for the time split and `earliest_cr_line` transformation; neither raw date is a model feature.",
        "- Exact `Charged Off` rows map to default (1); exact `Fully Paid` rows map to non-default (0).",
        "- Current, late, grace-period, `Default`, policy-exception, missing, and unknown statuses are not mapped to either class.",
        "",
        "Feature importance is descriptive predictive contribution in this historical sample, not causation or a lending rationale.",
        "",
    ]
    eligibility_path.write_text("\n".join(eligibility), encoding="utf-8")
    return json_path, markdown_path


__all__ = [
    "CATEGORICAL_FEATURES",
    "DATASET_FILENAME",
    "DATASET_SOURCE",
    "DEFAULT_LABELS",
    "FEATURE_COLUMNS",
    "NUMERIC_FEATURES",
    "RAW_REQUIRED_COLUMNS",
    "TARGET_COLUMN",
    "audit_lendingclub_dataset",
    "prepare_lendingclub_frame",
    "save_audit_report",
    "sha256_file",
]
