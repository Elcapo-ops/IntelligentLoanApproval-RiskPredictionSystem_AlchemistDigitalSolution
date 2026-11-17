import importlib
import json
import logging
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from config.paths import ProjectPaths, get_project_paths
from config.settings import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    load_config,
    validate_config,
)
from utils.exceptions import ConfigurationError
from utils.logging_config import configure_logging
from utils.reproducibility import set_random_seeds


def test_loads_project_configuration() -> None:
    config = load_config()

    assert DEFAULT_CONFIG_PATH == PROJECT_ROOT / "config.yaml"
    assert config["project"]["name"] == "intelligent-loan-approval"
    assert config["project"]["random_seed"] == 42
    assert config["project"]["deterministic"] is True
    assert config["paths"]["raw_data_dir"] == "data/raw"
    assert config["paths"]["raw_dataset"] == "data/raw/loan_applications.csv"
    assert config["paths"]["models_dir"] == "artifacts/models"
    assert config["paths"]["reports_dir"] == "artifacts/reports"
    assert config["training"]["test_size"] == 0.2
    assert config["training"]["cv_folds"] == 5
    assert config["streamlit"]["layout"] == "wide"
    assert config["logging"]["format"] == "json"
    assert config["logging"]["text_format"]


def test_rejects_invalid_yaml_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("project: [", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid YAML configuration"):
        load_config(config_path)


def test_rejects_invalid_random_seed(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_seed.yaml"
    config_path.write_text(
        "project:\n  random_seed: true\nlogging:\n  level: INFO\n"
        "  format: '%(message)s'\n  file: logs/app.log\npaths: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="project.random_seed"):
        load_config(config_path)


def test_seeds_python_and_numpy_generators() -> None:
    set_random_seeds(7)
    first_values = (random.random(), np.random.random())

    set_random_seeds(7)
    second_values = (random.random(), np.random.random())

    assert first_values == second_values


def test_rejects_boolean_as_random_seed() -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        set_random_seeds(True)


def test_rejects_numpy_seed_out_of_range() -> None:
    with pytest.raises(ValueError, match="between 0 and 4294967295"):
        set_random_seeds(2**32)


def test_path_manager_anchors_relative_paths_to_project_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    paths = get_project_paths()

    assert paths.root == PROJECT_ROOT
    assert paths.raw_data_dir == PROJECT_ROOT / "data" / "raw"
    assert paths.raw_dataset == PROJECT_ROOT / "data" / "raw" / "loan_applications.csv"
    assert paths.models_dir == PROJECT_ROOT / "artifacts" / "models"
    assert paths.reports_dir == PROJECT_ROOT / "artifacts" / "reports"


def test_path_manager_creates_configured_directories(tmp_path: Path) -> None:
    config = deepcopy(load_config())
    paths_config = config["paths"]
    for key, value in paths_config.items():
        paths_config[key] = str(tmp_path / Path(value))

    paths = ProjectPaths.from_config(config, root=tmp_path)
    paths.ensure_directories()

    for directory in (
        paths.raw_data_dir,
        paths.processed_data_dir,
        paths.external_data_dir,
        paths.models_dir,
        paths.reports_dir,
        paths.log_file.parent,
    ):
        assert directory.is_dir()
    assert not paths.raw_dataset.exists()
    assert not paths.processed_dataset.exists()


@pytest.mark.parametrize(
    ("section", "key", "invalid_value", "message"),
    [
        ("training", "test_size", 1, "training.test_size"),
        ("training", "cv_folds", 1, "training.cv_folds"),
        ("models", "candidates", [], "models.candidates"),
        ("streamlit", "layout", "full", "streamlit.layout"),
        ("logging", "level", "TRACE", "logging.level"),
    ],
)
def test_rejects_invalid_settings(
    section: str,
    key: str,
    invalid_value: object,
    message: str,
) -> None:
    config = deepcopy(load_config())
    config[section][key] = invalid_value

    with pytest.raises(ConfigurationError, match=message):
        validate_config(config)


def test_rejects_missing_path_setting() -> None:
    config = deepcopy(load_config())
    del config["paths"]["raw_dataset"]

    with pytest.raises(ConfigurationError, match="paths.raw_dataset"):
        validate_config(config)


def test_configures_structured_project_file_logging(tmp_path: Path) -> None:
    logger = logging.getLogger("loan_prediction")
    original_handlers = logger.handlers[:]
    original_level = logger.level
    original_propagate = logger.propagate
    for handler in original_handlers:
        logger.removeHandler(handler)

    log_path = tmp_path / "logs" / "app.log"
    try:
        config = deepcopy(load_config())
        config["paths"]["log_file"] = str(log_path)
        configured_logger = configure_logging(config)
        configure_logging(config)
        configured_logger.info(
            "foundation logging test",
            extra={"event": "test_event", "seed": 42},
        )
        for handler in configured_logger.handlers:
            handler.flush()

        records = log_path.read_text(encoding="utf-8").splitlines()
        assert len(records) == 1
        record = json.loads(records[0])
        assert record["message"] == "foundation logging test"
        assert record["event"] == "test_event"
        assert record["seed"] == 42
        assert record["level"] == "INFO"
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            if handler not in original_handlers:
                handler.close()
        for handler in original_handlers:
            logger.addHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_application_and_business_packages_import() -> None:
    package_names = (
        "app",
        "app.main",
        "app.pages",
        "app.components",
        "app.services",
        "config",
        "data",
        "features",
        "preprocessing",
        "models",
        "evaluation",
        "explainability",
        "prediction",
        "utils",
    )

    for package_name in package_names:
        assert importlib.import_module(package_name)
