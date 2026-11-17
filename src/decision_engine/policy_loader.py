"""Load and validate lending policy configuration from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from utils.exceptions import ConfigurationError

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "lending_policy.yaml"


def load_lending_policy(path: Path | str | None = None) -> dict[str, Any]:
    """Load a lending policy YAML file.

    Args:
        path: Optional path to the policy file. Defaults to config/lending_policy.yaml.

    Returns:
        Parsed policy dictionary.

    Raises:
        ConfigurationError: If the file cannot be read or parsed.
    """
    policy_path = Path(path) if path is not None else DEFAULT_POLICY_PATH
    try:
        with policy_path.open(encoding="utf-8") as policy_file:
            policy = yaml.safe_load(policy_file)
    except OSError as exc:
        raise ConfigurationError(f"Unable to read lending policy file: {policy_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in lending policy: {policy_path}") from exc

    if not isinstance(policy, dict):
        raise ConfigurationError("Lending policy root must be a mapping")
    if not policy.get("policy_version"):
        raise ConfigurationError("Lending policy is missing policy_version")
    return policy
