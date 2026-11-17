from copy import deepcopy
from pathlib import Path

import pandas as pd

from config.paths import get_project_paths
from config.settings import load_config
from data.loader import DatasetBundle
from data.quality_audit import (
    audit_dataset_bundle,
    render_quality_report,
    write_quality_report,
)


def _small_config() -> dict:
    config = deepcopy(load_config())
    config["dataset"]["categorical_domains"] = {"Gender": ["Female", "Male"]}
    config["dataset"]["numeric_columns"] = ["ApplicantIncome", "LoanAmount"]
    config["dataset"]["positive_numeric_columns"] = ["LoanAmount"]
    config["feature_engineering"]["log1p_columns"] = [
        "ApplicantIncome",
        "LoanAmount",
        "Total_Income",
    ]
    return config


def _small_bundle() -> DatasetBundle:
    labeled = pd.DataFrame(
        {
            "Loan_ID": ["L1", "L2", "L3", "L3"],
            "Gender": ["Female", "Unknown", None, "Male"],
            "ApplicantIncome": [0, -10, 100, 100000],
            "LoanAmount": [10, 0, 20, 20],
            "Loan_Status": ["Y", "N", "Y", "maybe"],
        }
    )
    labeled.loc[len(labeled)] = labeled.iloc[0]
    unlabeled = pd.DataFrame(
        {
            "Loan_ID": ["L4", "L5"],
            "Gender": ["Male", "Other"],
            "ApplicantIncome": [0, 50],
            "LoanAmount": [30, 40],
        }
    )
    return DatasetBundle(
        labeled=labeled,
        unlabeled=unlabeled,
        target_column="Loan_Status",
        identifier_columns=("Loan_ID",),
    )


def test_audit_reports_quality_issues_without_modifying_input_frames() -> None:
    datasets = _small_bundle()
    labeled_before = datasets.labeled.copy(deep=True)
    unlabeled_before = datasets.unlabeled.copy(deep=True)

    report = audit_dataset_bundle(datasets, config=_small_config())

    assert report["labeled"]["duplicate_row_count"] == 1
    assert report["labeled"]["duplicate_identifier_rows"]["Loan_ID"] == 4
    assert report["labeled"]["invalid_categories"]["Gender"] == {"Unknown": 1}
    assert report["unlabeled"]["invalid_categories"]["Gender"] == {"Other": 1}
    assert report["labeled"]["target"]["invalid_label_count"] == 1
    assert report["labeled"]["target"]["class_counts"] == {
        "N": 1,
        "Y": 3,
        "maybe": 1,
    }
    assert report["labeled"]["numeric_checks"]["ApplicantIncome"]["negative_count"] == 1
    assert report["labeled"]["numeric_checks"]["ApplicantIncome"]["zero_count"] == 2
    assert report["labeled"]["numeric_checks"]["LoanAmount"]["non_positive_count"] == 1
    assert (
        report["labeled"]["numeric_checks"]["ApplicantIncome"]["iqr_outlier_count"]
        >= 1
    )
    assert report["labeled"]["missing_values"]["Gender"]["count"] == 1
    assert report["identifier_overlap"]["Loan_ID"] == 0
    assert report["feature_schema_matches"] is True
    assert report["labeled"]["target"]["majority_to_minority_ratio"] == 3.0
    pd.testing.assert_frame_equal(datasets.labeled, labeled_before)
    pd.testing.assert_frame_equal(datasets.unlabeled, unlabeled_before)


def test_render_report_distinguishes_outliers_from_domain_violations() -> None:
    report = audit_dataset_bundle(_small_bundle(), config=_small_config())

    markdown = render_quality_report(report)

    assert "IQR fences are descriptive screening flags only" in markdown
    assert "unusual but valid applicants" in markdown
    assert "loan-term quartiles are all 360 months" in markdown
    assert "No observations were removed, imputed, or changed" in markdown
    assert "Negative" in markdown
    assert get_project_paths().quality_report == (
        get_project_paths().root / "reports" / "data_quality_report.md"
    )


def test_audit_flags_nonnumeric_and_nonfinite_numeric_values() -> None:
    datasets = _small_bundle()
    datasets.labeled["ApplicantIncome"] = datasets.labeled[
        "ApplicantIncome"
    ].astype(object)
    datasets.labeled.loc[0, "ApplicantIncome"] = "unknown"
    datasets.labeled.loc[1, "ApplicantIncome"] = float("inf")

    report = audit_dataset_bundle(datasets, config=_small_config())

    numeric = report["labeled"]["numeric_checks"]["ApplicantIncome"]
    assert numeric["non_numeric_count"] == 1
    assert numeric["non_finite_count"] == 1


def test_writes_markdown_report(tmp_path: Path) -> None:
    report = audit_dataset_bundle(_small_bundle(), config=_small_config())

    path = write_quality_report(report, tmp_path / "nested" / "report.md")

    assert path.is_file()
    assert path.read_text(encoding="utf-8").startswith("# Data quality audit")
