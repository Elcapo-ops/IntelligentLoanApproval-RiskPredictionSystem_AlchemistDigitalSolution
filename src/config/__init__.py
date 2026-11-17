"""Centralized project configuration."""

from .paths import ProjectPaths, get_project_paths
from .settings import load_config, validate_config

__all__ = ["ProjectPaths", "get_project_paths", "load_config", "validate_config"]
