"""Tests for the independent default-risk module status gate."""

from pathlib import Path

from default_risk.dataset_status import (
    evaluate_default_risk_dataset,
    get_default_risk_target_mapping,
)


def test_default_risk_dataset_file_is_detected_but_requires_pipeline_audit() -> None:
    root = Path(__file__).resolve().parents[2]
    status = evaluate_default_risk_dataset(root)

    assert status.status in {
        "PRESENT_NOT_VALIDATED",
        "AUDITED_NOT_TRAINED",
        "MODEL_TRAINED",
    }
    assert "accepted_2007_to_2018Q4.csv.gz" in status.raw_files
    assert status.missing_requirements == []
    assert status.notes


def test_default_risk_target_mapping_documents_valid_default_targets() -> None:
    target_map = get_default_risk_target_mapping()

    assert "lendingclub" in target_map
    assert "givemecredit" in target_map
    assert "Charged Off" in target_map["lendingclub"]["default_label"]
    assert "Fully Paid" in target_map["lendingclub"]["non_default_label"]
    assert "SeriousDlqin2yrs" in target_map["givemecredit"]["target_column"]


def test_compressed_lendingclub_file_is_detected_but_not_marked_validated(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "data" / "default_risk" / "raw"
    processed_dir = tmp_path / "data" / "default_risk" / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    (raw_dir / "accepted_2007_to_2018Q4.csv.gz").write_bytes(b"placeholder")

    status = evaluate_default_risk_dataset(tmp_path)

    assert status.raw_files == ["accepted_2007_to_2018Q4.csv.gz"]
    assert status.status == "PRESENT_NOT_VALIDATED"


def test_default_risk_status_is_not_available_without_dataset(tmp_path: Path) -> None:
    (tmp_path / "data" / "default_risk" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "default_risk" / "processed").mkdir(parents=True)
    status = evaluate_default_risk_dataset(tmp_path)

    assert status.status == "NOT_AVAILABLE"
    assert status.raw_files == []
    assert any("No raw CSV dataset" in item for item in status.missing_requirements)
