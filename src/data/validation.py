"""Schema, target, and identifier validation for loan CSV datasets."""

from collections.abc import Sequence

import pandas as pd

from utils.exceptions import DataValidationError


def identify_identifier_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Identify ID-like columns by conventional names.

    Identification is name-based to avoid mistaking naturally unique numeric
    features for record identifiers.
    """
    return tuple(
        str(column)
        for column in frame.columns
        if str(column).strip().lower() in {"id", "identifier"}
        or str(column).strip().lower().endswith(("_id", "_identifier", "_key"))
    )


def _validate_columns(frame: pd.DataFrame, dataset_name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError(f"{dataset_name} data must be a pandas DataFrame")
    if frame.empty:
        raise DataValidationError(f"{dataset_name} dataset contains no rows")
    if len(frame.columns) == 0:
        raise DataValidationError(f"{dataset_name} dataset contains no columns")
    if not frame.columns.is_unique:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise DataValidationError(
            f"{dataset_name} dataset has duplicate columns: {duplicates}"
        )
    blank_columns = [
        column
        for column in frame.columns
        if not isinstance(column, str) or not column.strip()
    ]
    if blank_columns:
        raise DataValidationError(
            f"{dataset_name} dataset has blank or invalid column names"
        )


def _validate_identifiers(
    frame: pd.DataFrame,
    columns: Sequence[str],
    dataset_name: str,
) -> None:
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise DataValidationError(
            f"{dataset_name} dataset is missing identifier columns: {missing_columns}"
        )

    for column in columns:
        values = frame[column]
        if values.isna().any():
            raise DataValidationError(
                f"{dataset_name} identifier column {column!r} contains missing values"
            )
        if values.astype("string").str.strip().eq("").any():
            raise DataValidationError(
                f"{dataset_name} identifier column {column!r} contains empty values"
            )
        if values.duplicated().any():
            raise DataValidationError(
                f"{dataset_name} identifier column {column!r} contains duplicates"
            )


def validate_labeled_dataset(
    frame: pd.DataFrame,
    target_column: str = "Loan_Status",
    allowed_target_values: Sequence[str] = ("Y", "N"),
    identifier_columns: Sequence[str] = (),
) -> tuple[str, ...]:
    """Validate a labeled dataset and return its identified ID columns."""
    _validate_columns(frame, "Labeled")
    if target_column not in frame.columns:
        raise DataValidationError(
            f"Labeled dataset is missing target column {target_column!r}"
        )
    if not allowed_target_values:
        raise DataValidationError("At least one allowed target value must be provided")

    target = frame[target_column]
    if target.isna().any():
        raise DataValidationError(
            f"Labeled target column {target_column!r} contains missing values"
        )
    observed_values = set(target.unique())
    invalid_values = observed_values.difference(allowed_target_values)
    if invalid_values:
        raise DataValidationError(
            f"Target {target_column!r} contains unsupported values: "
            f"{sorted(map(str, invalid_values))}"
        )

    inferred_identifiers = identify_identifier_columns(frame)
    identifiers = tuple(dict.fromkeys((*identifier_columns, *inferred_identifiers)))
    if target_column in identifiers:
        raise DataValidationError("The target column cannot be an identifier")
    _validate_identifiers(frame, identifiers, "Labeled")
    return identifiers


def validate_unlabeled_dataset(
    frame: pd.DataFrame,
    target_column: str = "Loan_Status",
    identifier_columns: Sequence[str] = (),
) -> tuple[str, ...]:
    """Validate an unlabeled dataset and return its identified ID columns."""
    _validate_columns(frame, "Unlabeled")
    if target_column in frame.columns:
        raise DataValidationError(
            f"Unlabeled dataset must not contain target column {target_column!r}"
        )

    inferred_identifiers = identify_identifier_columns(frame)
    identifiers = tuple(dict.fromkeys((*identifier_columns, *inferred_identifiers)))
    _validate_identifiers(frame, identifiers, "Unlabeled")
    return identifiers


def validate_dataset_pair(
    labeled: pd.DataFrame,
    unlabeled: pd.DataFrame,
    target_column: str = "Loan_Status",
    allowed_target_values: Sequence[str] = ("Y", "N"),
    identifier_columns: Sequence[str] = (),
) -> tuple[str, ...]:
    """Validate separately sourced labeled/unlabeled datasets without merging."""
    labeled_identifiers = validate_labeled_dataset(
        labeled,
        target_column=target_column,
        allowed_target_values=allowed_target_values,
        identifier_columns=identifier_columns,
    )
    unlabeled_identifiers = validate_unlabeled_dataset(
        unlabeled,
        target_column=target_column,
        identifier_columns=identifier_columns,
    )

    labeled_features = set(labeled.columns).difference({target_column})
    unlabeled_features = set(unlabeled.columns)
    if labeled_features != unlabeled_features:
        missing_from_unlabeled = sorted(labeled_features - unlabeled_features)
        missing_from_labeled = sorted(unlabeled_features - labeled_features)
        raise DataValidationError(
            "Labeled and unlabeled feature schemas differ; "
            f"missing from unlabeled: {missing_from_unlabeled}; "
            f"missing from labeled: {missing_from_labeled}"
        )

    shared_identifiers = set(labeled_identifiers).intersection(unlabeled_identifiers)
    for column in sorted(shared_identifiers):
        overlap = set(labeled[column].astype("string")).intersection(
            unlabeled[column].astype("string")
        )
        if overlap:
            raise DataValidationError(
                f"Identifier column {column!r} overlaps between labeled and "
                f"unlabeled datasets ({len(overlap)} values)"
            )
    return labeled_identifiers
