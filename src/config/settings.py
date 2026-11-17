"""Load and validate project-wide YAML settings."""

from pathlib import Path
from typing import Any

import yaml

from utils.exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_PATH_SETTINGS = (
    "raw_data_dir",
    "processed_data_dir",
    "external_data_dir",
    "raw_dataset",
    "labeled_dataset",
    "unlabeled_dataset",
    "processed_dataset",
    "models_dir",
    "reports_dir",
    "cv_results",
    "evaluation_report",
    "quality_report",
    "tuning_report",
    "tuning_results",
    "business_insights_report",
    "explainability_report",
    "global_importance",
    "fairness_results",
    "local_explanations",
    "figures_dir",
    "log_file",
)


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Missing or invalid configuration section: {key}")
    return value


def _require_nonempty_string(settings: dict[str, Any], key: str, section: str) -> str:
    value = settings.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Missing or invalid setting {section}.{key}")
    return value


def validate_config(config: Any) -> dict[str, Any]:
    """Validate all required settings and return the validated mapping.

    Raises:
        ConfigurationError: If a required section, setting, or value is invalid.
    """
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration root must be a mapping")

    project = _require_mapping(config, "project")
    seed = project.get("random_seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 2**32 - 1
    ):
        raise ConfigurationError(
            "project.random_seed must be an integer between 0 and 4294967295"
        )
    if not isinstance(project.get("deterministic"), bool):
        raise ConfigurationError("Missing or invalid setting project.deterministic")
    _require_nonempty_string(project, "name", "project")

    paths = _require_mapping(config, "paths")
    for key in _PATH_SETTINGS:
        _require_nonempty_string(paths, key, "paths")

    dataset = _require_mapping(config, "dataset")
    _require_nonempty_string(dataset, "source_url", "dataset")
    _require_nonempty_string(dataset, "target_column", "dataset")
    target_classes = dataset.get("target_classes")
    if (
        not isinstance(target_classes, list)
        or len(target_classes) < 2
        or any(not isinstance(value, str) or not value for value in target_classes)
        or len(set(target_classes)) != len(target_classes)
    ):
        raise ConfigurationError(
            "dataset.target_classes must contain at least two unique strings"
        )
    identifier_columns = dataset.get("identifier_columns")
    if (
        not isinstance(identifier_columns, list)
        or any(not isinstance(value, str) or not value for value in identifier_columns)
    ):
        raise ConfigurationError("dataset.identifier_columns must be a list of names")
    categorical_domains = dataset.get("categorical_domains")
    if not isinstance(categorical_domains, dict) or any(
        not isinstance(column, str)
        or not column
        or not isinstance(values, list)
        or not values
        for column, values in categorical_domains.items()
    ):
        raise ConfigurationError(
            "dataset.categorical_domains must map column names to non-empty lists"
        )
    for key in ("numeric_columns", "positive_numeric_columns"):
        columns = dataset.get(key)
        if (
            not isinstance(columns, list)
            or any(not isinstance(column, str) or not column for column in columns)
        ):
            raise ConfigurationError(f"dataset.{key} must be a list of column names")
    if not set(dataset["positive_numeric_columns"]).issubset(
        dataset["numeric_columns"]
    ):
        raise ConfigurationError(
            "dataset.positive_numeric_columns must be a subset of numeric_columns"
        )

    feature_engineering = _require_mapping(config, "feature_engineering")
    log1p_columns = feature_engineering.get("log1p_columns")
    valid_log_columns = set(dataset["numeric_columns"]).union({"Total_Income"})
    if (
        not isinstance(log1p_columns, list)
        or any(not isinstance(column, str) or not column for column in log1p_columns)
        or len(log1p_columns) != len(set(log1p_columns))
        or not set(log1p_columns).issubset(valid_log_columns)
    ):
        raise ConfigurationError(
            "feature_engineering.log1p_columns must contain unique numeric "
            "feature names or Total_Income"
        )

    training = _require_mapping(config, "training")
    test_size = training.get("test_size")
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, (int, float))
        or not 0 < test_size < 1
    ):
        raise ConfigurationError(
            "training.test_size must be greater than 0 and less than 1"
        )
    cv_folds = training.get("cv_folds")
    if isinstance(cv_folds, bool) or not isinstance(cv_folds, int) or cv_folds < 2:
        raise ConfigurationError("training.cv_folds must be an integer of at least 2")
    _require_nonempty_string(training, "model_version", "training")
    _require_nonempty_string(training, "optimized_model_version", "training")

    optimization = _require_mapping(config, "optimization")
    n_iter = optimization.get("n_iter")
    if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 1:
        raise ConfigurationError("optimization.n_iter must be a positive integer")
    scoring = _require_mapping(optimization, "scoring")
    if scoring.get("primary") != "balanced_accuracy":
        raise ConfigurationError(
            "optimization.scoring.primary must be 'balanced_accuracy'"
        )
    if scoring.get("secondary") != "roc_auc":
        raise ConfigurationError("optimization.scoring.secondary must be 'roc_auc'")
    parameter_distributions = optimization.get("parameter_distributions")
    if (
        not isinstance(parameter_distributions, dict)
        or not parameter_distributions
        or any(
            not isinstance(name, str)
            or not name.startswith("estimator__")
            or not isinstance(values, list)
            or not values
            for name, values in parameter_distributions.items()
        )
    ):
        raise ConfigurationError(
            "optimization.parameter_distributions must map estimator__ parameter "
            "names to non-empty lists"
        )

    explainability = _require_mapping(config, "explainability")
    _require_nonempty_string(explainability, "model_name", "explainability")
    _require_nonempty_string(explainability, "model_version", "explainability")
    for key in ("permutation_repeats", "fairness_min_group_size", "local_cases"):
        value = explainability.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigurationError(
                f"explainability.{key} must be a positive integer"
            )
    fairness_attributes = explainability.get("fairness_attributes")
    if (
        not isinstance(fairness_attributes, list)
        or not fairness_attributes
        or any(
            not isinstance(value, str) or not value
            for value in fairness_attributes
        )
        or len(fairness_attributes) != len(set(fairness_attributes))
    ):
        raise ConfigurationError(
        "explainability.fairness_attributes must contain unique feature names"
    )
    shap_background_rows = explainability.get("shap_background_rows")
    if (
        isinstance(shap_background_rows, bool)
        or not isinstance(shap_background_rows, int)
        or shap_background_rows < 1
    ):
        raise ConfigurationError(
            "explainability.shap_background_rows must be a positive integer"
        )

    models = _require_mapping(config, "models")
    candidates = models.get("candidates")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(
            not isinstance(candidate, str) or not candidate.strip()
            for candidate in candidates
        )
    ):
        raise ConfigurationError("models.candidates must be a non-empty list of names")
    parameters = _require_mapping(models, "parameters")
    for model_name, model_parameters in parameters.items():
        if not isinstance(model_name, str) or not isinstance(model_parameters, dict):
            raise ConfigurationError("models.parameters must map names to mappings")

    streamlit = _require_mapping(config, "streamlit")
    _require_nonempty_string(streamlit, "page_title", "streamlit")
    if streamlit.get("layout") not in {"centered", "wide"}:
        raise ConfigurationError("streamlit.layout must be 'centered' or 'wide'")

    calibration = _require_mapping(config, "calibration")
    if not isinstance(calibration.get("enabled"), bool):
        raise ConfigurationError("calibration.enabled must be a boolean")
    calibration_methods = calibration.get("methods")
    if not isinstance(calibration_methods, list) or not calibration_methods:
        raise ConfigurationError("calibration.methods must be a non-empty list")
    if any(method not in {"sigmoid", "isotonic"} for method in calibration_methods):
        raise ConfigurationError("calibration methods must be sigmoid or isotonic")
    if isinstance(calibration.get("cv_folds"), bool) or not isinstance(calibration.get("cv_folds"), int) or calibration.get("cv_folds") < 2:
        raise ConfigurationError("calibration.cv_folds must be an integer >= 2")
    if calibration.get("positive_label") not in {"Y", "N"}:
        raise ConfigurationError("calibration.positive_label must be 'Y' or 'N'")

    thresholds = _require_mapping(config, "thresholds")
    if not isinstance(thresholds.get("enabled"), bool):
        raise ConfigurationError("thresholds.enabled must be a boolean")
    threshold_candidates = thresholds.get("candidate_thresholds")
    if not isinstance(threshold_candidates, list) or not threshold_candidates:
        raise ConfigurationError("thresholds.candidate_thresholds must be a non-empty list")
    if any(not isinstance(value, (int, float)) or not 0 <= float(value) <= 1 for value in threshold_candidates):
        raise ConfigurationError("thresholds.candidate_thresholds must contain numbers in [0, 1]")

    default_risk = _require_mapping(config, "default_risk")
    if not isinstance(default_risk.get("enabled"), bool):
        raise ConfigurationError("default_risk.enabled must be a boolean")
    if default_risk.get("dataset_name") != "lendingclub":
        raise ConfigurationError("default_risk.dataset_name must be 'lendingclub'")
    if default_risk.get("target_name") != "loan_status":
        raise ConfigurationError("default_risk.target_name must be 'loan_status'")
    if default_risk.get("calibration_method") not in {"sigmoid", "isotonic"}:
        raise ConfigurationError(
            "default_risk.calibration_method must be 'sigmoid' or 'isotonic'"
        )
    for key in ("test_start_year", "max_train_rows", "max_test_rows", "cv_folds"):
        value = default_risk.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ConfigurationError(f"default_risk.{key} must be an integer >= 2")
    risk_bands = _require_mapping(default_risk, "risk_category_thresholds")
    band_values = [
        risk_bands.get("low_risk_max"),
        risk_bands.get("moderate_risk_max"),
        risk_bands.get("high_risk_max"),
    ]
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 < float(value) < 1
            for value in band_values
        )
        or not band_values[0] < band_values[1] < band_values[2]
    ):
        raise ConfigurationError(
            "default_risk risk-category boundaries must be strictly increasing in (0, 1)"
        )
    if not isinstance(default_risk.get("training_ready"), bool):
        raise ConfigurationError("default_risk.training_ready must be a boolean")

    logging_settings = _require_mapping(config, "logging")
    level = _require_nonempty_string(logging_settings, "level", "logging").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigurationError(f"Unsupported logging.level: {level}")
    log_format = _require_nonempty_string(logging_settings, "format", "logging")
    if log_format not in {"json", "text"}:
        raise ConfigurationError("logging.format must be 'json' or 'text'")
    _require_nonempty_string(logging_settings, "text_format", "logging")

    return config


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate settings from ``path``.

    The default path is rooted at the project, not the current working
    directory. Relative configured paths are resolved by ``ProjectPaths``.

    Raises:
        ConfigurationError: If the file cannot be read, parsed, or validated.
    """
    try:
        with path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except OSError as exc:
        raise ConfigurationError(f"Unable to read configuration file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML configuration in {path}") from exc

    return validate_config(config)
