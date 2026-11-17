"""Leakage-aware splitting of labeled data before preprocessing."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import load_config, validate_config
from data.validation import validate_labeled_dataset
from utils.exceptions import DataValidationError


@dataclass(frozen=True)
class DataSplit:
    """Model features and target in independent training and holdout sets."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    target_column: str


def get_model_feature_columns(config: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Return the configured, prediction-time feature allowlist."""
    settings = validate_config(config if config is not None else load_config())
    dataset = settings["dataset"]
    numeric_columns = tuple(dataset["numeric_columns"])
    categorical_columns = tuple(dataset["categorical_domains"])
    feature_columns = (*numeric_columns, *categorical_columns)

    if len(feature_columns) != len(set(feature_columns)):
        raise DataValidationError(
            "Configured numeric and categorical feature columns must be distinct"
        )
    excluded = set(dataset["identifier_columns"]).union(
        {dataset["target_column"]}
    )
    if excluded.intersection(feature_columns):
        raise DataValidationError(
            "Configured model features must not include target or identifier columns"
        )
    return feature_columns


def split_labeled_data(
    frame: pd.DataFrame,
    test_size: float | None = None,
    random_state: int | None = None,
    config: dict[str, Any] | None = None,
) -> DataSplit:
    """Validate and stratify labeled rows before fitting any transformations.

    The configured feature allowlist excludes the target, identifiers, and any
    other columns not explicitly declared as prediction-time inputs. Exact
    duplicate feature rows are rejected rather than allowed to cross
    partitions. The returned holdout is reserved for final evaluation.
    """
    settings = validate_config(config if config is not None else load_config())
    dataset_settings = settings["dataset"]
    training_settings = settings["training"]
    target_column = dataset_settings["target_column"]
    identifiers = tuple(dataset_settings["identifier_columns"])

    validate_labeled_dataset(
        frame,
        target_column=target_column,
        allowed_target_values=dataset_settings["target_classes"],
        identifier_columns=identifiers,
    )

    feature_columns = get_model_feature_columns(settings)
    missing_features = [name for name in feature_columns if name not in frame.columns]
    if missing_features:
        raise DataValidationError(
            f"Labeled dataset is missing configured model features: {missing_features}"
        )

    X = frame.loc[:, feature_columns].copy()
    y = frame.loc[:, target_column].copy()
    if X.duplicated(keep=False).any():
        raise DataValidationError(
            "Duplicate model-feature rows found; resolve duplicate or conflicting "
            "records before splitting to prevent partition leakage"
        )

    split_size = (
        training_settings["test_size"] if test_size is None else test_size
    )
    seed = settings["project"]["random_seed"] if random_state is None else random_state
    if isinstance(split_size, bool) or not isinstance(split_size, (int, float)):
        raise DataValidationError("test_size must be a number between 0 and 1")
    if not 0 < split_size < 1:
        raise DataValidationError("test_size must be greater than 0 and less than 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DataValidationError("random_state must be an integer")

    class_counts = y.value_counts()
    if len(class_counts) < 2 or (class_counts < 2).any():
        raise DataValidationError(
            "Stratified splitting requires at least two rows in every target class"
        )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=split_size,
            random_state=seed,
            stratify=y,
        )
    except ValueError as exc:
        raise DataValidationError(f"Unable to stratify labeled data: {exc}") from exc

    return DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_column=target_column,
    )
