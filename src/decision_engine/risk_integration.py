"""Integration adapter for the Loan Default / Credit Risk Prediction module.

This adapter loads the registered default-risk model through its existing
supported interface and obtains an estimated default probability for a loan
application. It is optional — the decision engine can operate without it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.registry import load_model_artifact
from utils.exceptions import ArtifactError, DataValidationError

from decision_engine.schemas import ApplicationInput, RiskAssessment


class DefaultRiskAdapter:
    """Adapter for the optional default-risk model integration."""

    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root).resolve()
        self._model: Any = None
        self._metadata: dict[str, Any] | None = None
        self._loaded: bool = False
        self._load_error: str | None = None

    def _ensure_loaded(self) -> None:
        """Load the model once on first use."""
        if self._loaded:
            return
        self._loaded = True
        # Lazy import to avoid pandas dependency at module import time
        try:
            from default_risk.prediction import load_default_risk_model
        except ImportError as exc:
            self._model = None
            self._metadata = None
            self._load_error = f"Default-risk module not available: {exc}"
            return
        try:
            model, metadata = load_default_risk_model(self.project_root)
            self._model = model
            self._metadata = metadata
        except (ArtifactError, DataValidationError, ImportError, ModuleNotFoundError) as exc:
            self._model = None
            self._metadata = None
            self._load_error = str(exc)
        except Exception as exc:  # pylint: disable=broad-except
            self._model = None
            self._metadata = None
            self._load_error = f"Unexpected error loading default-risk model: {exc}"

    def is_available(self) -> bool:
        """Return True if the default-risk model was loaded successfully."""
        self._ensure_loaded()
        return self._model is not None

    def get_load_error(self) -> str | None:
        """Return the error message if model loading failed, else None."""
        self._ensure_loaded()
        return self._load_error

    def get_metadata(self) -> dict[str, Any] | None:
        """Return the model metadata if loaded."""
        self._ensure_loaded()
        return self._metadata

    def predict(
        self,
        application: ApplicationInput,
    ) -> RiskAssessment:
        """Obtain default-risk assessment for an application.

        Returns a RiskAssessment with model_used=False if the model is unavailable
        or prediction fails. Never raises — failures are captured in the error field.
        """
        self._ensure_loaded()

        if not self.is_available():
            return RiskAssessment(
                model_used=False,
                model_name=None,
                model_version=None,
                estimated_default_probability=None,
                risk_category=None,
                prediction_timestamp=None,
                error=self._load_error or "Default-risk model is not available",
            )

        try:
            from default_risk.prediction import predict_default_risk
            from default_risk.data import FEATURE_COLUMNS as DEFAULT_RISK_FEATURES
        except ImportError as exc:
            return RiskAssessment(
                model_used=False,
                model_name=None,
                model_version=None,
                estimated_default_probability=None,
                risk_category=None,
                prediction_timestamp=datetime.now(timezone.utc).isoformat(),
                error=f"Default-risk module not available: {exc}",
            )

        # Build the feature dict expected by the default-risk model
        feature_dict: dict[str, Any] = {}
        missing_features: list[str] = []

        for feature in DEFAULT_RISK_FEATURES:
            value = self._map_application_to_feature(application, feature)
            if value is None:
                missing_features.append(feature)
                feature_dict[feature] = None
            else:
                feature_dict[feature] = value

        try:
            result = predict_default_risk(self._model, feature_dict, self._metadata or {})
            return RiskAssessment(
                model_used=True,
                model_name=result.get("model_name"),
                model_version=result.get("model_version"),
                estimated_default_probability=result.get("default_probability"),
                risk_category=result.get("risk_category"),
                prediction_timestamp=datetime.now(timezone.utc).isoformat(),
                error=None,
            )
        except DataValidationError as exc:
            return RiskAssessment(
                model_used=False,
                model_name=self._metadata.get("model_name") if self._metadata else None,
                model_version=self._metadata.get("model_version") if self._metadata else None,
                estimated_default_probability=None,
                risk_category=None,
                prediction_timestamp=datetime.now(timezone.utc).isoformat(),
                error=f"Default-risk prediction failed: {exc}",
            )
        except Exception as exc:  # pylint: disable=broad-except
            return RiskAssessment(
                model_used=False,
                model_name=self._metadata.get("model_name") if self._metadata else None,
                model_version=self._metadata.get("model_version") if self._metadata else None,
                estimated_default_probability=None,
                risk_category=None,
                prediction_timestamp=datetime.now(timezone.utc).isoformat(),
                error=f"Unexpected default-risk prediction error: {exc}",
            )

    def _map_application_to_feature(
        self,
        application: ApplicationInput,
        feature: str,
    ) -> Any:
        """Map decision engine application fields to default-risk model features.

        Only maps fields that exist in both schemas. Returns None for unmappable
        or missing values.
        """
        mapping = {
            "loan_amnt": application.requested_loan_amount,
            "annual_inc": application.annual_income,
            "dti": None,  # DTI is calculated separately, not stored on application
            "delinq_2yrs": None,
            "fico_range_low": None,
            "fico_range_high": None,
            "inq_last_6mths": None,
            "open_acc": None,
            "pub_rec": None,
            "revol_bal": None,
            "revol_util": None,
            "total_acc": None,
            "credit_history_months": None,
            "term": None,
            "emp_length": None,
            "home_ownership": application.home_ownership,
            "verification_status": application.verification_status,
            "purpose": application.loan_purpose,
            "addr_state": application.residency_state,
            "application_type": None,
        }
        return mapping.get(feature)
