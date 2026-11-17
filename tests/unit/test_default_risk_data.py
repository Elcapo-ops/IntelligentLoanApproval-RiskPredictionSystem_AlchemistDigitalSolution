"""Tests for LendingClub target mapping and feature eligibility."""

import pandas as pd
import pytest

from default_risk.data import (
    FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    prepare_lendingclub_frame,
)
from utils.exceptions import DataValidationError


def _row(status: str, issue_date: str, credit_date: str) -> dict[str, object]:
    row: dict[str, object] = {column: None for column in RAW_FEATURE_COLUMNS}
    row.update(
        {
            "id": "test-loan",
            "loan_status": status,
            "issue_d": issue_date,
            "loan_amnt": 12_000,
            "term": " 36 months",
            "emp_length": "10+ years",
            "home_ownership": "RENT",
            "annual_inc": 72_000,
            "verification_status": "Verified",
            "purpose": "debt_consolidation",
            "addr_state": "CA",
            "dti": 17.5,
            "delinq_2yrs": 0,
            "earliest_cr_line": credit_date,
            "fico_range_low": 700,
            "fico_range_high": 704,
            "inq_last_6mths": 1,
            "open_acc": 8,
            "pub_rec": 0,
            "revol_bal": 9_000,
            "revol_util": 42.0,
            "total_acc": 17,
            "application_type": "Individual",
        }
    )
    return row


def test_target_mapping_keeps_only_matured_statuses_and_engineers_features() -> None:
    source = pd.DataFrame(
        [
            _row("Charged Off", "Jan-2015", "Jan-2005"),
            _row("Fully Paid", "Feb-2015", "Feb-2005"),
            _row("Current", "Mar-2015", "Mar-2005"),
            _row("Late (31-120 days)", "Apr-2015", "Apr-2005"),
            _row("Default", "May-2015", "May-2005"),
        ]
    )

    prepared = prepare_lendingclub_frame(source)

    assert prepared["default_target"].tolist() == [1, 0]
    assert prepared.loc[0, "credit_history_months"] == 120
    assert set(FEATURE_COLUMNS).issubset(prepared.columns)
    assert "issue_year" not in FEATURE_COLUMNS
    assert "issue_month" not in FEATURE_COLUMNS
    assert not {
        "loan_status",
        "total_pymnt",
        "recoveries",
        "int_rate",
        "grade",
    }.intersection(FEATURE_COLUMNS)


def test_target_mapping_rejects_missing_required_schema() -> None:
    with pytest.raises(DataValidationError, match="missing columns"):
        prepare_lendingclub_frame(pd.DataFrame({"loan_status": ["Charged Off"]}))
