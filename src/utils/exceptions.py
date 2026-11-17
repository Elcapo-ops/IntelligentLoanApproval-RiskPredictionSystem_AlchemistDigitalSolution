"""Application exception hierarchy for expected project errors."""


class LoanPredictionError(Exception):
    """Base class for expected loan prediction application errors."""


class ConfigurationError(LoanPredictionError):
    """Raised when project configuration cannot be loaded or validated."""


class DataValidationError(LoanPredictionError):
    """Raised when input data does not meet the expected data contract."""


class ArtifactError(LoanPredictionError):
    """Raised when a required model artifact cannot be read or used."""
