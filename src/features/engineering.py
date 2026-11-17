"""Deterministic loan-applicant feature engineering."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from utils.exceptions import DataValidationError


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create income/loan features without learning from rows or target values.

    Missing, non-numeric, and non-finite numeric inputs are represented as
    ``NaN`` for downstream training-fold imputation. Ratios with nonpositive
    total income and log1p values outside the nonnegative domain are ``NaN``.
    Finite extreme values are not clipped using arbitrary or holdout-derived
    limits; non-finite results from arithmetic are converted to ``NaN``.
    """

    def __init__(
        self,
        numeric_columns: Sequence[str] = (
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
        ),
        log1p_columns: Sequence[str] = (),
        excluded_columns: Sequence[str] = (),
    ) -> None:
        self.numeric_columns = numeric_columns
        self.log1p_columns = log1p_columns
        self.excluded_columns = excluded_columns

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "LoanFeatureEngineer":
        """Validate required inputs without estimating any data statistics."""
        del y
        if not isinstance(X, pd.DataFrame):
            raise DataValidationError("Feature engineering input must be a DataFrame")
        required_columns = set(self.numeric_columns).union(
            {"ApplicantIncome", "CoapplicantIncome", "LoanAmount"}
        )
        missing_columns = sorted(required_columns.difference(X.columns))
        if missing_columns:
            raise DataValidationError(
                f"Feature engineering input is missing required columns: "
                f"{missing_columns}"
            )
        valid_log_columns = set(X.columns).union({"Total_Income"})
        invalid_log_columns = sorted(set(self.log1p_columns).difference(valid_log_columns))
        if invalid_log_columns:
            raise DataValidationError(
                f"Cannot apply log1p to unconfigured feature columns: "
                f"{invalid_log_columns}"
            )
        input_columns = X.drop(columns=list(self.excluded_columns), errors="ignore")
        self.feature_names_in_ = np.asarray(input_columns.columns, dtype=object)
        self.n_features_in_ = len(input_columns.columns)
        return self

    @staticmethod
    def _finite_numeric(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
        return numeric.where(np.isfinite(numeric), np.nan)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return sanitized inputs plus derived numeric features."""
        check_is_fitted(self, attributes=["feature_names_in_"])
        if not isinstance(X, pd.DataFrame):
            raise DataValidationError("Feature engineering input must be a DataFrame")
        transformed = X.drop(
            columns=list(self.excluded_columns),
            errors="ignore",
        ).copy()
        missing_columns = sorted(
            set(self.feature_names_in_).difference(transformed.columns)
        )
        if missing_columns:
            raise DataValidationError(
                f"Feature engineering input is missing fitted columns: "
                f"{missing_columns}"
            )

        for column in self.numeric_columns:
            transformed[column] = self._finite_numeric(transformed[column])

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            total_income = (
                transformed["ApplicantIncome"] + transformed["CoapplicantIncome"]
            )
            total_income = total_income.where(np.isfinite(total_income), np.nan)
            transformed["Total_Income"] = total_income

            valid_ratio = (
                transformed["Total_Income"].gt(0)
                & transformed["LoanAmount"].ge(0)
            )
            ratio = transformed["LoanAmount"].div(
                transformed["Total_Income"].where(valid_ratio)
            )
            transformed["Loan_to_Income_Ratio"] = ratio.where(
                valid_ratio & np.isfinite(ratio), np.nan
            )

            for column in self.log1p_columns:
                values = (
                    transformed["Total_Income"]
                    if column == "Total_Income"
                    else self._finite_numeric(transformed[column])
                )
                valid_log = values.ge(0)
                logged = np.log1p(values.where(valid_log))
                transformed[f"{column}_log1p"] = logged.where(
                    valid_log & np.isfinite(logged), np.nan
                )

        return transformed.drop(
            columns=list(self.excluded_columns),
            errors="ignore",
        )

    def get_feature_names_out(
        self,
        input_features: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return names emitted by the transformation."""
        check_is_fitted(self, attributes=["feature_names_in_"])
        names = (
            list(input_features)
            if input_features is not None
            else list(self.feature_names_in_)
        )
        names.extend(["Total_Income", "Loan_to_Income_Ratio"])
        names.extend(f"{column}_log1p" for column in self.log1p_columns)
        return np.asarray(
            [name for name in names if name not in self.excluded_columns],
            dtype=object,
        )
