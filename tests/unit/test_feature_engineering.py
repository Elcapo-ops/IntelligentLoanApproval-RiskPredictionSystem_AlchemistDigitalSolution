import numpy as np
import pandas as pd
import pytest

from features.engineering import LoanFeatureEngineer
from utils.exceptions import DataValidationError


@pytest.fixture
def engineer() -> LoanFeatureEngineer:
    return LoanFeatureEngineer(
        log1p_columns=(
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Total_Income",
        ),
        excluded_columns=("Loan_ID", "Loan_Status"),
    )


def test_engineers_total_income_ratio_and_safe_logs(
    engineer: LoanFeatureEngineer,
) -> None:
    frame = pd.DataFrame(
        {
            "Loan_ID": ["A"],
            "Loan_Status": ["Y"],
            "ApplicantIncome": [100.0],
            "CoapplicantIncome": [50.0],
            "LoanAmount": [30.0],
            "Loan_Amount_Term": [360.0],
        }
    )

    transformed = engineer.fit_transform(frame)

    assert transformed.loc[0, "Total_Income"] == 150.0
    assert transformed.loc[0, "Loan_to_Income_Ratio"] == pytest.approx(0.2)
    assert transformed.loc[0, "ApplicantIncome_log1p"] == pytest.approx(
        np.log1p(100.0)
    )
    assert transformed.loc[0, "Total_Income_log1p"] == pytest.approx(
        np.log1p(150.0)
    )
    assert "Loan_ID" not in transformed
    assert "Loan_Status" not in transformed


def test_zero_or_missing_income_produces_missing_ratio(
    engineer: LoanFeatureEngineer,
) -> None:
    frame = pd.DataFrame(
        {
            "ApplicantIncome": [0.0, np.nan, 100.0],
            "CoapplicantIncome": [0.0, 10.0, np.nan],
            "LoanAmount": [50.0, 50.0, 50.0],
            "Loan_Amount_Term": [360.0, 360.0, 360.0],
        }
    )

    transformed = engineer.fit_transform(frame)

    assert transformed.loc[0, "Total_Income"] == 0.0
    assert transformed["Loan_to_Income_Ratio"].isna().all()


def test_nonfinite_invalid_and_extreme_values_are_handled(
    engineer: LoanFeatureEngineer,
) -> None:
    maximum = np.finfo(np.float64).max
    frame = pd.DataFrame(
        {
            "ApplicantIncome": [
                np.inf,
                maximum,
                -1.0,
                "not-a-number",
                1e-320,
                -2.0,
            ],
            "CoapplicantIncome": [1.0, maximum, 2.0, 10.0, 0.0, -1.0],
            "LoanAmount": [1.0, maximum, -5.0, 100.0, maximum, 10.0],
            "Loan_Amount_Term": [360.0] * 6,
        }
    )

    transformed = engineer.fit_transform(frame)

    assert pd.isna(transformed.loc[0, "ApplicantIncome"])
    assert pd.isna(transformed.loc[0, "Total_Income"])
    assert pd.isna(transformed.loc[0, "Loan_to_Income_Ratio"])
    assert pd.isna(transformed.loc[1, "Total_Income"])
    assert pd.isna(transformed.loc[1, "Loan_to_Income_Ratio"])
    assert np.isfinite(transformed.loc[1, "ApplicantIncome_log1p"])
    assert pd.isna(transformed.loc[2, "Loan_to_Income_Ratio"])
    assert pd.isna(transformed.loc[2, "ApplicantIncome_log1p"])
    assert pd.isna(transformed.loc[3, "ApplicantIncome"])
    assert pd.isna(transformed.loc[3, "Loan_to_Income_Ratio"])
    assert pd.isna(transformed.loc[4, "Loan_to_Income_Ratio"])
    assert transformed.loc[5, "Total_Income"] == -3.0
    assert pd.isna(transformed.loc[5, "Loan_to_Income_Ratio"])
    assert pd.isna(transformed.loc[5, "Total_Income_log1p"])


def test_fitted_transformation_is_identical_at_training_and_inference(
    engineer: LoanFeatureEngineer,
) -> None:
    training = pd.DataFrame(
        {
            "Loan_ID": ["train"],
            "Loan_Status": ["Y"],
            "ApplicantIncome": [1000.0],
            "CoapplicantIncome": [500.0],
            "LoanAmount": [200.0],
            "Loan_Amount_Term": [360.0],
        }
    )
    inference = training.drop(columns="Loan_Status").copy()
    inference.loc[:, "Loan_ID"] = "inference"
    inference.loc[:, "ApplicantIncome"] = 1000.0

    fitted = engineer.fit(training)
    train_result = fitted.transform(training.drop(columns="Loan_Status"))
    inference_result = fitted.transform(inference)

    pd.testing.assert_frame_equal(train_result, inference_result)


def test_engineering_never_uses_target_values(
    engineer: LoanFeatureEngineer,
) -> None:
    frame = pd.DataFrame(
        {
            "Loan_ID": ["A", "B"],
            "Loan_Status": ["Y", "N"],
            "ApplicantIncome": [100.0, 200.0],
            "CoapplicantIncome": [50.0, 100.0],
            "LoanAmount": [30.0, 60.0],
            "Loan_Amount_Term": [360.0, 180.0],
        }
    )
    changed_target = frame.assign(Loan_Status=["N", "Y"])

    first = engineer.fit_transform(frame)
    second = engineer.fit_transform(changed_target)

    pd.testing.assert_frame_equal(first, second)


def test_missing_required_feature_is_reported(
    engineer: LoanFeatureEngineer,
) -> None:
    frame = pd.DataFrame(
        {
            "ApplicantIncome": [100.0],
            "CoapplicantIncome": [50.0],
            "LoanAmount": [30.0],
        }
    )

    with pytest.raises(DataValidationError, match="missing required columns"):
        engineer.fit(frame)
