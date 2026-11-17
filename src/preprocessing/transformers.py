"""Reusable training-fold preprocessing transformers."""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_numeric_pipeline() -> Pipeline:
    """Impute numeric values with the training median and standardize them."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )


def make_categorical_pipeline() -> Pipeline:
    """Impute categories from training data and safely encode unseen values."""
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
