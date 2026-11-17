import pytest

from config.paths import get_project_paths
from data.quality_audit import audit_configured_datasets

_PATHS = get_project_paths()
_DATA_FILES_PRESENT = (
    _PATHS.labeled_dataset.is_file() and _PATHS.unlabeled_dataset.is_file()
)


@pytest.mark.skipif(
    not _DATA_FILES_PRESENT,
    reason="Acquired Kaggle CSVs are not included in source control",
)
def test_acquired_dataset_has_no_invalid_categories_or_domain_errors() -> None:
    report = audit_configured_datasets()

    for dataset_name in ("labeled", "unlabeled"):
        dataset = report[dataset_name]
        assert all(not invalid for invalid in dataset["invalid_categories"].values())
        assert dataset["duplicate_row_count"] == 0
        assert all(
            duplicate_count == 0
            for duplicate_count in dataset["duplicate_identifier_rows"].values()
        )
        for numeric in dataset["numeric_checks"].values():
            assert numeric["negative_count"] == 0
            assert numeric["non_numeric_count"] == 0
            assert numeric["non_finite_count"] == 0

    assert report["labeled"]["target"]["invalid_label_count"] == 0
    assert report["labeled"]["target"]["missing_count"] == 0
    assert report["labeled"]["target"]["class_counts"] == {"Y": 422, "N": 192}
    assert report["labeled"]["missing_values"]["Credit_History"]["count"] == 50
    assert report["identifier_overlap"]["Loan_ID"] == 0
    assert report["feature_schema_matches"] is True
