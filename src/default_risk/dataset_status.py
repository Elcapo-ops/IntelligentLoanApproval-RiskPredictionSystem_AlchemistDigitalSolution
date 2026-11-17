"""Status checks for the independent default-risk module.

This project remains intentionally scoped to the historical `Loan_Status`
approval classifier. The default-risk module is only activated after a valid
re-payment dataset is downloaded and validated.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class DefaultRiskDatasetStatus:
    """Data-acquisition readiness for the default-risk module."""

    dataset_name: str
    status: str
    dataset_dir: str
    raw_files: list[str]
    missing_requirements: list[str]
    notes: str


def get_default_risk_target_mapping() -> dict[str, dict[str, dict[str, str]]]:
    """Return the currently supported mapping definitions for genuine repayment data."""
    return {
        "lendingclub": {
            "target_column": {"loan_status": "loan_status"},
            "default_label": {"Charged Off": "default"},
            "non_default_label": {"Fully Paid": "non_default"},
            "excluded_statuses": {
                "Current": "excluded_unresolved",
                "Late": "excluded_unresolved",
                "In Grace Period": "excluded_unresolved",
                "Default": "excluded_unresolved",
            },
        },
        "givemecredit": {
            "target_column": {"SeriousDlqin2yrs": "SeriousDlqin2yrs"},
            "default_label": {"1": "default"},
            "non_default_label": {"0": "non_default"},
            "excluded_statuses": {},
        },
    }


def evaluate_default_risk_dataset(project_root: Path | str) -> DefaultRiskDatasetStatus:
    """Report whether a default-risk dataset is available for training.

    The default-risk module stays intentionally inactive until a genuine
    repayment/default dataset has been downloaded and validated. This function
    only checks file availability and project layout; it never invents labels.
    """
    root = Path(project_root)
    dataset_dir = root / "data" / "default_risk"
    raw_dir = dataset_dir / "raw"
    processed_dir = dataset_dir / "processed"

    missing_requirements: list[str] = []
    if not dataset_dir.exists():
        missing_requirements.append("data/default_risk directory is missing")
    if not raw_dir.exists():
        missing_requirements.append("data/default_risk/raw directory is missing")
    if not processed_dir.exists():
        missing_requirements.append("data/default_risk/processed directory is missing")

    supported_extensions = (".csv", ".csv.gz", ".zip")
    raw_files = (
        sorted(
            path.name
            for path in raw_dir.iterdir()
            if path.is_file() and path.name.lower().endswith(supported_extensions)
        )
        if raw_dir.exists()
        else []
    )
    if not raw_files:
        missing_requirements.append(
            "No raw CSV dataset file found for LendingClub or Give Me Some Credit"
        )

    if missing_requirements:
        return DefaultRiskDatasetStatus(
            dataset_name="lendingclub_or_givemecredit",
            status="NOT_AVAILABLE",
            dataset_dir=str(dataset_dir),
            raw_files=raw_files,
            missing_requirements=missing_requirements,
            notes=(
                "The default-risk module is intentionally blocked until the "
                "source repayment dataset has been downloaded and validated. "
                "This module is not trained and no default-risk model is produced "
                "without genuine default labels."
            ),
        )

    audit_path = root / "reports" / "default_risk" / "lendingclub_dataset_audit.json"
    audit_exists = audit_path.is_file()
    model_dir = root / "artifacts" / "models" / "default_risk"
    registry_path = model_dir / "registry.json"
    model_registered = False
    if registry_path.is_file() and not registry_path.is_symlink():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            model_registered = (
                "lendingclub_default_risk:0.1.0"
                in registry.get("models", {})
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            model_registered = False
    status = (
        "MODEL_TRAINED"
        if model_registered
        else "AUDITED_NOT_TRAINED"
        if audit_exists
        else "PRESENT_NOT_VALIDATED"
    )
    notes = (
        "A separately registered LendingClub model artifact is available."
        if model_registered
        else "The dataset audit report is present; model training is still required."
        if audit_exists
        else "A candidate data file is present but is not yet validated. A complete "
        "target-definition, feature-eligibility, and data-quality audit must pass "
        "before model training."
    )
    return DefaultRiskDatasetStatus(
        dataset_name="lendingclub_or_givemecredit",
        status=status,
        dataset_dir=str(dataset_dir),
        raw_files=raw_files,
        missing_requirements=[],
        notes=notes,
    )


__all__ = [
    "DefaultRiskDatasetStatus",
    "evaluate_default_risk_dataset",
    "get_default_risk_target_mapping",
]
