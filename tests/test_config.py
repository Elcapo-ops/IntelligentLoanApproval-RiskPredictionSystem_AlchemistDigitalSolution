from pathlib import Path

import pytest

from config.settings import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    load_config,
    validate_config,
)
from utils.exceptions import ConfigurationError


def test_loads_shared_project_configuration() -> None:
    config = load_config()

    assert config["project"]["name"] == "intelligent-loan-approval"
    assert config["project"]["random_seed"] == 42
    assert config["paths"]["raw_data_dir"] == "data/raw"
    assert config["training"]["test_size"] == 0.2
    assert config["training"]["cv_folds"] == 5
    assert config["training"]["model_version"] == "1.0.0"
    assert config["training"]["optimized_model_version"] == "1.1.0"
    assert config["optimization"]["scoring"]["primary"] == "balanced_accuracy"
    assert config["explainability"]["model_version"] == "1.1.0"
    assert config["explainability"]["fairness_attributes"] == [
        "Gender",
        "Education",
        "Married",
    ]
    assert config["feature_engineering"]["log1p_columns"] == [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Total_Income",
    ]
    assert DEFAULT_CONFIG_PATH == PROJECT_ROOT / "config.yaml"


def test_rejects_log_feature_outside_configured_numeric_inputs() -> None:
    config = load_config()
    config["feature_engineering"]["log1p_columns"] = ["Loan_Status"]

    with pytest.raises(ConfigurationError, match="feature_engineering.log1p_columns"):
        validate_config(config)


def test_rejects_invalid_optimization_parameters() -> None:
    config = load_config()
    config["optimization"]["parameter_distributions"] = {
        "max_depth": [3]
    }

    with pytest.raises(ConfigurationError, match="estimator__"):
        validate_config(config)


def test_rejects_configuration_with_non_mapping_root(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("- item\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="root must be a mapping"):
        load_config(config_path)


def test_rejects_malformed_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "malformed.yaml"
    config_path.write_text("project: [", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid YAML configuration"):
        load_config(config_path)


def test_missing_settings_are_reported_explicitly(tmp_path: Path) -> None:
    config_path = tmp_path / "missing.yaml"
    config_path.write_text("project:\n  name: loan\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="project.random_seed"):
        load_config(config_path)


def test_default_config_is_independent_of_current_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config["project"]["name"] == "intelligent-loan-approval"
    assert DEFAULT_CONFIG_PATH == PROJECT_ROOT / "config.yaml"
