"""Backward-compatible imports for the centralized configuration."""

from config.settings import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    load_config,
    validate_config,
)

__all__ = ["DEFAULT_CONFIG_PATH", "PROJECT_ROOT", "load_config", "validate_config"]
