from typing import Any
import shutil

import numpy as np
import pandas as pd
import pytest

from config.paths import get_project_paths
from app.services.prediction_service import (
    get_form_defaults,
    load_prediction_model,
    predict_historical_approval,
)
from config.settings import load_config
from evaluation.evaluate import _positive_probability
from utils.exceptions import ArtifactError, DataValidationError


@pytest.fixture(scope="module")
def prediction_context() -> tuple[dict[str, Any], Any, dict[str, Any]]:
    config = load_config()
    model, _ = load_prediction_model(config)
    return config, model, get_form_defaults(config)


@pytest.mark.parametrize(
    "values",
    [
        {"ApplicantIncome": 0.0, "CoapplicantIncome": 0.0},
        {"ApplicantIncome": 1e9, "CoapplicantIncome": 1e9, "LoanAmount": 1e8},
    ],
)
def test_prediction_accepts_zero_income_and_extreme_finite_values(
    prediction_context: tuple[dict[str, Any], Any, dict[str, Any]],
    values: dict[str, float],
) -> None:
    config, model, defaults = prediction_context

    prediction = predict_historical_approval(model, {**defaults, **values}, config)

    assert prediction["predicted_loan_status"] in {"N", "Y"}


def test_empty_prediction_input_reports_schema_error(
    prediction_context: tuple[dict[str, Any], Any, dict[str, Any]],
) -> None:
    config, model, _ = prediction_context

    with pytest.raises(DataValidationError, match="schema mismatch"):
        predict_historical_approval(model, {}, config)


def test_unrepresentably_extreme_input_is_a_handled_validation_error(
    prediction_context: tuple[dict[str, Any], Any, dict[str, Any]],
) -> None:
    config, model, defaults = prediction_context

    with pytest.raises(DataValidationError, match="supported range"):
        predict_historical_approval(
            model,
            {
                **defaults,
                "ApplicantIncome": 1e100,
                "CoapplicantIncome": 1e100,
                "LoanAmount": 1e100,
            },
            config,
        )


def test_pipeline_ignores_unknown_categories_and_preserves_probability_contract(
    prediction_context: tuple[dict[str, Any], Any, dict[str, Any]],
) -> None:
    config, model, defaults = prediction_context
    unknown_category = {
        **defaults,
        "Gender": "FutureCategory",
        "Property_Area": "FutureArea",
    }

    with pytest.raises(DataValidationError, match="outside configured categories"):
        predict_historical_approval(model, unknown_category, config)

    frame = pd.DataFrame([unknown_category])
    probabilities = model.predict_proba(frame)

    assert model.predict(frame).shape == (1,)
    assert probabilities.shape == (1, 2)
    assert list(model.classes_) == ["N", "Y"]
    assert np.isfinite(probabilities).all()


def test_positive_probability_uses_label_lookup_and_checks_dimensions() -> None:
    class ReverseClassOrderModel:
        classes_ = np.array(["Y", "N"])

        @staticmethod
        def predict_proba(_frame: Any) -> np.ndarray:
            del _frame
            return np.array([[0.8, 0.2]])

    class InvalidProbabilityShapeModel(ReverseClassOrderModel):
        @staticmethod
        def predict_proba(_frame: Any) -> np.ndarray:
            del _frame
            return np.array([[1.0]])

    frame = object()
    assert _positive_probability(ReverseClassOrderModel(), frame).tolist() == [0.8]

    with pytest.raises(ArtifactError, match="does not match class ordering"):
        _positive_probability(InvalidProbabilityShapeModel(), frame)


def test_prediction_model_load_surfaces_corrupt_registered_artifact(
    prediction_context: tuple[dict[str, Any], Any, dict[str, Any]],
    tmp_path,
) -> None:
    from copy import deepcopy

    config, _, _ = prediction_context
    corrupted_config = deepcopy(config)
    corrupted_config["paths"]["models_dir"] = str(tmp_path)
    paths = get_project_paths(corrupted_config)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    original_model_dir = get_project_paths(config).models_dir
    for filename in (
        "registry.json",
        f"{config['explainability']['model_name']}-"
        f"{config['explainability']['model_version']}.joblib",
        f"{config['explainability']['model_name']}-"
        f"{config['explainability']['model_version']}.metadata.json",
    ):
        shutil.copy2(original_model_dir / filename, paths.models_dir / filename)
    artifact = paths.models_dir / (
        f"{config['explainability']['model_name']}-"
        f"{config['explainability']['model_version']}.joblib"
    )
    artifact.write_bytes(b"corrupt model bytes")

    with pytest.raises(ArtifactError, match="integrity check failed"):
        load_prediction_model(corrupted_config)
