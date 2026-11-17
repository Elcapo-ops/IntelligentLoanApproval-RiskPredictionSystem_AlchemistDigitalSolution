import json
from pathlib import Path

import pandas as pd
import pytest

from config.settings import load_config
from data.loader import load_csv, load_datasets
from data.profiling import profile_datasets, write_profile_report
from data.validation import (
    identify_identifier_columns,
    validate_dataset_pair,
    validate_labeled_dataset,
)
from utils.exceptions import DataValidationError


@pytest.fixture
def labeled_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Loan_ID": ["LP001", "LP002", "LP003", "LP004"],
            "ApplicantIncome": [5000, 3000, 7000, 2500],
            "Credit_History": [1.0, 0.0, 1.0, 1.0],
            "Loan_Status": ["Y", "N", "Y", "Y"],
        }
    )


@pytest.fixture
def unlabeled_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Loan_ID": ["LP005", "LP006"],
            "ApplicantIncome": [4200, 3100],
            "Credit_History": [1.0, 0.0],
        }
    )


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def test_loads_two_csv_files_separately_and_validates_target(
    tmp_path: Path,
    labeled_frame: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
) -> None:
    labeled_path = _write_frame(labeled_frame, tmp_path / "train.csv")
    unlabeled_path = _write_frame(unlabeled_frame, tmp_path / "test.csv")

    datasets = load_datasets(labeled_path, unlabeled_path, load_config())

    assert datasets.labeled.equals(labeled_frame)
    assert datasets.unlabeled is not None
    assert datasets.unlabeled.equals(unlabeled_frame)
    assert datasets.target_column == "Loan_Status"
    assert datasets.identifier_columns == ("Loan_ID",)
    assert "Loan_Status" not in datasets.unlabeled.columns


def test_missing_unlabeled_file_is_reported_without_merging(
    tmp_path: Path,
    labeled_frame: pd.DataFrame,
) -> None:
    labeled_path = _write_frame(labeled_frame, tmp_path / "train.csv")

    config = load_config()
    paths = dict(config["paths"])
    paths["unlabeled_dataset"] = str(tmp_path / "not-present.csv")
    config["paths"] = paths
    with pytest.raises(DataValidationError, match="does not exist"):
        load_datasets(labeled_path=labeled_path, config=config)

    assert pd.read_csv(labeled_path)["Loan_Status"].tolist() == ["Y", "N", "Y", "Y"]


def test_load_csv_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="does not exist"):
        load_csv(tmp_path / "missing.csv")


def test_load_csv_rejects_duplicate_headers(tmp_path: Path) -> None:
    path = tmp_path / "duplicate_headers.csv"
    path.write_text("Loan_ID,Loan_ID,Loan_Status\n1,1,Y\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="duplicate header"):
        load_csv(path)


def test_load_csv_rejects_header_without_data_rows(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("Loan_ID,Loan_Status\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="no data rows"):
        load_csv(path)


def test_identifies_conventional_identifier_columns(
    labeled_frame: pd.DataFrame,
) -> None:
    assert identify_identifier_columns(labeled_frame) == ("Loan_ID",)


def test_rejects_missing_target(labeled_frame: pd.DataFrame) -> None:
    frame = labeled_frame.drop(columns="Loan_Status")

    with pytest.raises(DataValidationError, match="missing target"):
        validate_labeled_dataset(frame, identifier_columns=("Loan_ID",))


def test_rejects_invalid_target_values(labeled_frame: pd.DataFrame) -> None:
    frame = labeled_frame.copy()
    frame.loc[0, "Loan_Status"] = "Pending"

    with pytest.raises(DataValidationError, match="unsupported values"):
        validate_labeled_dataset(frame, identifier_columns=("Loan_ID",))


def test_rejects_missing_target_labels(labeled_frame: pd.DataFrame) -> None:
    frame = labeled_frame.copy()
    frame.loc[0, "Loan_Status"] = None

    with pytest.raises(DataValidationError, match="contains missing"):
        validate_labeled_dataset(frame, identifier_columns=("Loan_ID",))


def test_rejects_target_leakage_in_unlabeled_dataset(
    labeled_frame: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
) -> None:
    unlabeled = unlabeled_frame.assign(Loan_Status=["Y", "N"])

    with pytest.raises(DataValidationError, match="must not contain target"):
        validate_dataset_pair(
            labeled_frame,
            unlabeled,
            identifier_columns=("Loan_ID",),
        )


def test_rejects_feature_schema_mismatch(
    labeled_frame: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
) -> None:
    unlabeled = unlabeled_frame.drop(columns="Credit_History")

    with pytest.raises(DataValidationError, match="feature schemas differ"):
        validate_dataset_pair(
            labeled_frame,
            unlabeled,
            identifier_columns=("Loan_ID",),
        )


def test_rejects_duplicate_identifiers(
    labeled_frame: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
) -> None:
    duplicate_labeled = labeled_frame.copy()
    duplicate_labeled.loc[1, "Loan_ID"] = duplicate_labeled.loc[0, "Loan_ID"]

    with pytest.raises(DataValidationError, match="contains duplicates"):
        validate_dataset_pair(
            duplicate_labeled,
            unlabeled_frame,
            identifier_columns=("Loan_ID",),
        )


def test_rejects_identifier_overlap_across_files(
    labeled_frame: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
) -> None:
    overlapping_unlabeled = unlabeled_frame.copy()
    overlapping_unlabeled.loc[0, "Loan_ID"] = labeled_frame.loc[0, "Loan_ID"]

    with pytest.raises(DataValidationError, match="overlaps"):
        validate_dataset_pair(
            labeled_frame,
            overlapping_unlabeled,
            identifier_columns=("Loan_ID",),
        )


def test_profiles_schema_and_target_distribution_separately(
    labeled_frame: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
) -> None:
    report = profile_datasets(
        labeled_frame,
        unlabeled_frame,
        identifier_columns=("Loan_ID",),
        source_url="https://example.test/dataset",
    )

    assert report["status"] == "complete"
    assert report["labeled"]["row_count"] == 4
    assert report["unlabeled"]["row_count"] == 2
    assert report["labeled"]["target_distribution"] == {
        "N": {"count": 1, "proportion": 0.25},
        "Y": {"count": 3, "proportion": 0.75},
    }
    assert report["unlabeled"]["target_distribution"] is None
    applicant_income = next(
        column
        for column in report["labeled"]["schema"]
        if column["column"] == "ApplicantIncome"
    )
    assert applicant_income["dtype"] == "int64"
    assert report["identifier_columns"] == ["Loan_ID"]


def test_writes_json_profile_report(
    tmp_path: Path,
    labeled_frame: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
) -> None:
    report = profile_datasets(
        labeled_frame,
        unlabeled_frame,
        identifier_columns=("Loan_ID",),
    )
    output_path = write_profile_report(report, tmp_path / "nested" / "profile.json")

    saved_report = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_report == report
