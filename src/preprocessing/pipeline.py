"""Leakage-safe preprocessing and estimator pipeline construction."""

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from config.settings import load_config, validate_config
from data.splitting import get_model_feature_columns
from features.engineering import LoanFeatureEngineer
from utils.exceptions import DataValidationError
from preprocessing.transformers import (
    make_categorical_pipeline,
    make_numeric_pipeline,
)


def build_preprocessor(config: dict[str, Any] | None = None) -> ColumnTransformer:
    """Build an unfitted transformer for the configured model feature schema.

    Fitting this transformer must happen only on training data. When used in
    cross-validation, pass it inside an estimator pipeline so each fold fits
    its own imputers, scaler, and encoder.
    """
    settings = validate_config(config if config is not None else load_config())
    dataset = settings["dataset"]
    numeric_columns = list(dataset["numeric_columns"])
    engineered_numeric_columns = [
        "Total_Income",
        "Loan_to_Income_Ratio",
        *(
            f"{column}_log1p"
            for column in settings["feature_engineering"]["log1p_columns"]
        ),
    ]
    categorical_columns = list(dataset["categorical_domains"])
    feature_columns = get_model_feature_columns(settings)
    if not numeric_columns or not categorical_columns:
        raise DataValidationError(
            "Preprocessing requires configured numeric and categorical features"
        )
    if len(feature_columns) != len(numeric_columns) + len(categorical_columns):
        raise DataValidationError(
            "Numeric and categorical preprocessing columns must be distinct"
        )
    numeric_columns.extend(engineered_numeric_columns)

    return ColumnTransformer(
        transformers=[
            ("numeric", make_numeric_pipeline(), numeric_columns),
            ("categorical", make_categorical_pipeline(), categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(
    estimator: BaseEstimator,
    config: dict[str, Any] | None = None,
) -> Pipeline:
    """Place an estimator after preprocessing for fold-local fitting.

    Use this complete pipeline with cross-validation or hyperparameter search;
    do not preprocess the full dataset before passing data to those routines.
    """
    if estimator is None:
        raise DataValidationError("An estimator is required to build the pipeline")
    settings = validate_config(config if config is not None else load_config())
    return Pipeline(
        steps=[
            (
                "feature_engineering",
                LoanFeatureEngineer(
                    numeric_columns=settings["dataset"]["numeric_columns"],
                    log1p_columns=settings["feature_engineering"]["log1p_columns"],
                    excluded_columns=(
                        *settings["dataset"]["identifier_columns"],
                        settings["dataset"]["target_column"],
                    ),
                ),
            ),
            ("preprocessor", build_preprocessor(config)),
            ("estimator", estimator),
        ]
    )
