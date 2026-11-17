"""Model loading, input validation, and live approval prediction."""

import json
import warnings
from typing import Any

import numpy as np
import pandas as pd

from config.paths import get_project_paths
from config.settings import load_config, validate_config
from data.loader import load_csv
from data.splitting import get_model_feature_columns, split_labeled_data
from models.registry import load_model_artifact
from prediction.scope import HISTORICAL_LOAN_STATUS_DISCLOSURE
from utils.exceptions import DataValidationError


def _load_training_features(settings: dict[str, Any]) -> pd.DataFrame:
    """Load only the configured training partition for form references."""
    paths = get_project_paths(settings)
    labeled_data = load_csv(paths.labeled_dataset)
    return split_labeled_data(labeled_data, config=settings).X_train


def _reference_values(
    training_features: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Calculate training-only medians and modes for display and explanations."""
    dataset = settings["dataset"]
    values: dict[str, Any] = {}
    for column in dataset["numeric_columns"]:
        values[column] = float(
            pd.to_numeric(training_features[column], errors="coerce").median()
        )
    for column in dataset["categorical_domains"]:
        modes = training_features[column].mode(dropna=True)
        if modes.empty:
            raise DataValidationError(
                f"Cannot determine a training reference value for {column}"
            )
        values[column] = modes.iloc[0]
    return values


def get_form_defaults(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute editable examples from training-partition medians and modes."""
    settings = validate_config(config if config is not None else load_config())
    training_features = _load_training_features(settings)
    return _reference_values(training_features, settings)


def load_prediction_model(
    config: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load and verify the configured local model and raw-feature schema."""
    settings = validate_config(config if config is not None else load_config())
    paths = get_project_paths(settings)
    dataset = load_csv(paths.labeled_dataset)
    feature_columns = get_model_feature_columns(settings)
    expected_schema = {column: str(dataset[column].dtype) for column in feature_columns}
    explainability = settings["explainability"]
    return load_model_artifact(
        model_name=explainability["model_name"],
        model_version=explainability["model_version"],
        model_dir=paths.models_dir,
        expected_feature_schema=expected_schema,
        expected_target_mapping={"N": 0, "Y": 1},
    )


def predict_historical_approval(
    model: Any,
    applicant: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict the historical approval class without exposing unvalidated scores."""
    settings = validate_config(config if config is not None else load_config())
    frame = _validated_applicant_row(applicant, settings)
    predicted_loan_status = _predict_class(model, frame, settings)
    return {
        "prediction_type": "historical_approval_prediction",
        "predicted_loan_status": predicted_loan_status,
        "model_name": settings["explainability"]["model_name"],
        "model_version": settings["explainability"]["model_version"],
        "target": "Loan_Status",
        "disclosure": HISTORICAL_LOAN_STATUS_DISCLOSURE,
    }


def explain_historical_approval_prediction(
    model: Any,
    applicant: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Report one-feature-at-a-time class sensitivity using training references.

    Replacing one input independently can create unrealistic combinations and
    does not establish a causal or additive explanation.
    """
    settings = validate_config(config if config is not None else load_config())
    feature_columns = get_model_feature_columns(settings)
    training_features = _load_training_features(settings)
    if list(training_features.columns) != list(feature_columns):
        raise DataValidationError(
            "Training reference schema does not match the prediction feature schema"
        )

    validated = _validated_applicant_row(applicant, settings)
    reference_values = _reference_values(training_features, settings)
    baseline = _predict_class(model, validated, settings)
    explanations: list[dict[str, Any]] = []
    for column in feature_columns:
        changed = validated.copy()
        changed.loc[0, column] = reference_values[column]
        changed_class = _predict_class(model, changed, settings)
        explanations.append(
            {
                "feature": column,
                "applicant_value": _display_value(validated.loc[0, column]),
                "training_reference": _display_value(reference_values[column]),
                "predicted_loan_status_after_replacement": changed_class,
                "changed": changed_class != baseline,
            }
        )
    return {
        "prediction_type": "historical_approval_prediction",
        "predicted_loan_status": baseline,
        "method": "one-feature-at-a-time replacement with training median/mode",
        "reference_rows": int(len(training_features)),
        "features": explanations,
        "disclosure": HISTORICAL_LOAN_STATUS_DISCLOSURE,
        "warning": (
            "Descriptive model sensitivity only. Replacements can create "
            "unrealistic profiles and do not establish causation or an "
            "individual lending rationale."
        ),
    }


def build_historical_prediction_exports(
    prediction: dict[str, Any],
    model_metadata: dict[str, Any],
) -> dict[str, str]:
    """Create downloadable JSON and CSV summaries without applicant inputs."""
    summary = {
        "prediction_type": "historical_approval_prediction",
        "target": "Loan_Status",
        "predicted_loan_status": prediction["predicted_loan_status"],
        "model_name": prediction["model_name"],
        "model_version": prediction["model_version"],
        "model_training_date": str(
            model_metadata.get("training_date", "unavailable")
        ),
        "disclosure": HISTORICAL_LOAN_STATUS_DISCLOSURE,
    }
    return {
        "json": json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        "csv": pd.DataFrame([summary]).to_csv(index=False),
    }


def _validated_applicant_row(
    applicant: dict[str, Any],
    settings: dict[str, Any],
) -> pd.DataFrame:
    """Validate form input and return an ordered raw-feature row."""
    feature_columns = get_model_feature_columns(settings)
    missing = sorted(set(feature_columns).difference(applicant))
    extra = sorted(set(applicant).difference(feature_columns))
    if missing or extra:
        raise DataValidationError(
            f"Prediction input schema mismatch; missing={missing}, extra={extra}"
        )

    input_row: dict[str, Any] = {}
    for column in feature_columns:
        value = applicant[column]
        if isinstance(value, str) and value == "Not provided":
            value = np.nan
        is_missing = value is None or value is pd.NA
        if not is_missing and not isinstance(value, (list, tuple, dict)):
            is_missing = bool(pd.isna(value))
        if not is_missing:
            if column in settings["dataset"]["numeric_columns"]:
                if isinstance(value, (bool, np.bool_)):
                    raise DataValidationError(
                        f"Prediction input {column} must be numeric"
                    )
                try:
                    value = float(value)
                except (TypeError, ValueError) as exc:
                    raise DataValidationError(
                        f"Prediction input {column} must be numeric"
                    ) from exc
                if not np.isfinite(value):
                    raise DataValidationError(
                        f"Prediction input {column} must be finite"
                    )
                if value < 0:
                    raise DataValidationError(
                        f"Prediction input {column} cannot be negative"
                    )
                if (
                    column in settings["dataset"]["positive_numeric_columns"]
                    and value <= 0
                ):
                    raise DataValidationError(
                        f"Prediction input {column} must be greater than zero"
                    )
            elif column in settings["dataset"]["categorical_domains"]:
                allowed = settings["dataset"]["categorical_domains"][column]
                if not any(value == allowed_value for allowed_value in allowed):
                    raise DataValidationError(
                        f"Prediction input {column} is outside configured categories"
                    )
            else:
                raise DataValidationError(
                    f"Prediction input {column} is not in the model schema"
                )
        input_row[column] = np.nan if is_missing else value
    return pd.DataFrame([input_row], columns=feature_columns)


def _predict_class(
    model: Any,
    frame: pd.DataFrame,
    settings: dict[str, Any],
) -> str:
    """Validate model target classes and return one supported prediction."""
    expected_classes = set(settings["dataset"]["target_classes"])
    model_classes = {str(label) for label in getattr(model, "classes_", ())}
    if model_classes != expected_classes:
        raise DataValidationError(
            "Prediction model classes do not match configured Loan_Status labels"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            predicted_loan_status = str(model.predict(frame)[0])
    except (RuntimeWarning, ValueError) as exc:
        raise DataValidationError(
            "Prediction model could not process the validated input; numeric "
            "values may exceed the fitted pipeline's supported range"
        ) from exc
    if predicted_loan_status not in expected_classes:
        raise DataValidationError(
            "Prediction model returned an unsupported Loan_Status label: "
            f"{predicted_loan_status}"
        )
    return predicted_loan_status


def _display_value(value: Any) -> Any:
    """Convert NumPy scalars and missing values to UI-friendly values."""
    if pd.isna(value):
        return "Not provided"
    if isinstance(value, np.generic):
        return value.item()
    return value
