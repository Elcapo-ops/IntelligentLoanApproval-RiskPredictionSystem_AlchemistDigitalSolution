"""Non-transforming schema and distribution profiling for loan datasets."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from data.validation import validate_dataset_pair
from utils.exceptions import DataValidationError


def _profile_frame(
    frame: pd.DataFrame,
    target_column: str | None,
    identifier_columns: tuple[str, ...],
) -> dict[str, Any]:
    schema: list[dict[str, Any]] = []
    for column in frame.columns:
        values = frame[column]
        missing_count = int(values.isna().sum())
        schema.append(
            {
                "column": str(column),
                "dtype": str(values.dtype),
                "non_null_count": int(values.notna().sum()),
                "missing_count": missing_count,
                "missing_percentage": round(missing_count / len(frame) * 100, 4),
                "unique_values": int(values.nunique(dropna=True)),
                "identifier": column in identifier_columns,
                "target": column == target_column,
            }
        )

    profile: dict[str, Any] = {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "duplicate_row_count": int(frame.duplicated().sum()),
        "schema": schema,
        "identifier_columns": list(identifier_columns),
    }
    if target_column is not None:
        counts = frame[target_column].value_counts().sort_index()
        profile["target_distribution"] = {
            str(label): {
                "count": int(count),
                "proportion": round(float(count / len(frame)), 6),
            }
            for label, count in counts.items()
        }
    else:
        profile["target_distribution"] = None

    return profile


def profile_datasets(
    labeled: pd.DataFrame,
    unlabeled: pd.DataFrame,
    target_column: str = "Loan_Status",
    allowed_target_values: tuple[str, ...] = ("Y", "N"),
    identifier_columns: tuple[str, ...] = (),
    source_url: str | None = None,
) -> dict[str, Any]:
    """Build separate JSON-serializable profiles without combining datasets."""
    identified = validate_dataset_pair(
        labeled,
        unlabeled,
        target_column=target_column,
        allowed_target_values=allowed_target_values,
        identifier_columns=identifier_columns,
    )

    report: dict[str, Any] = {
        "status": "complete",
        "source_url": source_url,
        "target_column": target_column,
        "identifier_columns": list(identified),
        "labeled": _profile_frame(
            labeled,
            target_column,
            identified,
        ),
    }
    report["unlabeled"] = _profile_frame(
        unlabeled,
        None,
        tuple(column for column in identified if column in unlabeled.columns),
    )
    return report


def write_profile_report(report: dict[str, Any], output_path: Path) -> Path:
    """Write a profile report as formatted JSON, creating its parent directory."""
    if report.get("status") != "complete":
        raise DataValidationError("Only complete profile reports can be written")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise DataValidationError(
            f"Unable to write profile report: {output_path}"
        ) from exc
    return output_path
