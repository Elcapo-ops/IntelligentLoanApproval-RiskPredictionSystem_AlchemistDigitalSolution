"""Reusable CSV loading for separately labeled and unlabeled loan data."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config.paths import get_project_paths
from config.settings import load_config, validate_config
from data.validation import validate_dataset_pair
from utils.exceptions import DataValidationError


@dataclass(frozen=True)
class DatasetBundle:
    """Labeled and unlabeled datasets loaded and validated independently."""

    labeled: pd.DataFrame
    unlabeled: pd.DataFrame
    target_column: str
    identifier_columns: tuple[str, ...]


def load_csv(path: Path) -> pd.DataFrame:
    """Read one CSV file and reject unreadable or duplicate-header inputs."""
    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"Dataset file does not exist: {path}")

    try:
        with path.open(encoding="utf-8-sig", newline="") as csv_file:
            header = next(csv.reader(csv_file), None)
        if not header:
            raise DataValidationError(f"Dataset file has no CSV header: {path}")
        if len(header) != len(set(header)):
            duplicates = sorted({name for name in header if header.count(name) > 1})
            raise DataValidationError(
                f"Dataset file has duplicate header columns: {duplicates}"
            )
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except DataValidationError:
        raise
    except (
        OSError,
        UnicodeDecodeError,
        csv.Error,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise DataValidationError(f"Unable to read CSV dataset: {path}") from exc

    if not frame.columns.is_unique:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise DataValidationError(
            f"Dataset file has duplicate header columns: {duplicates}"
        )
    if frame.empty:
        raise DataValidationError(f"Dataset file contains no data rows: {path}")
    return frame


def load_datasets(
    labeled_path: Path | None = None,
    unlabeled_path: Path | None = None,
    config: dict[str, Any] | None = None,
) -> DatasetBundle:
    """Load configured CSVs separately and validate target/schema contracts.

    No concatenation, transformation, or train/test split is performed. When
    paths are omitted, both separately configured Kaggle train and test files
    are required; missing files are reported explicitly.
    """
    settings = validate_config(
        config if config is not None else load_config()
    )
    project_paths = get_project_paths(settings)
    dataset_settings = settings["dataset"]

    labeled_frame = load_csv(
        Path(labeled_path)
        if labeled_path is not None
        else project_paths.labeled_dataset
    )
    target_column = dataset_settings["target_column"]
    allowed_target_values = dataset_settings["target_classes"]
    identifiers = tuple(dataset_settings["identifier_columns"])

    unlabeled_frame = load_csv(
        Path(unlabeled_path)
        if unlabeled_path is not None
        else project_paths.unlabeled_dataset
    )
    identified = validate_dataset_pair(
        labeled_frame,
        unlabeled_frame,
        target_column=target_column,
        allowed_target_values=allowed_target_values,
        identifier_columns=identifiers,
    )
    return DatasetBundle(
        labeled=labeled_frame,
        unlabeled=unlabeled_frame,
        target_column=target_column,
        identifier_columns=identified,
    )
