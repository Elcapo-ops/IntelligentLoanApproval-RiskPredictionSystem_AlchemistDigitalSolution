"""Trusted-local Joblib model artifact registry and integrity validation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import re
import tempfile
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn

from utils.exceptions import ArtifactError

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_REGISTRY_FILENAME = "registry.json"


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash ordered dataframe contents, column names, and dtypes deterministically."""
    digest = hashlib.sha256()
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    digest.update(json.dumps(schema, separators=(",", ":"), ensure_ascii=True).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    """Return runtime versions relevant to serialized model compatibility."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _validate_identifier(value: str, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ArtifactError(f"Invalid model artifact {label}: {value!r}")
    return value


def _artifact_paths(
    model_dir: Path,
    model_name: str,
    model_version: str,
) -> tuple[Path, Path, Path]:
    name = _validate_identifier(model_name, _NAME_PATTERN, "name")
    version = _validate_identifier(model_version, _VERSION_PATTERN, "version")
    root = Path(model_dir).resolve()
    artifact = root / f"{name}-{version}.joblib"
    metadata = root / f"{name}-{version}.metadata.json"
    registry = root / _REGISTRY_FILENAME
    return artifact, metadata, registry


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, indent=2, sort_keys=True, allow_nan=False)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_registry(registry_path: Path) -> dict[str, Any]:
    if registry_path.is_symlink() or not registry_path.is_file():
        raise ArtifactError(f"Trusted model registry is missing: {registry_path}")
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Unable to read model registry: {registry_path}") from exc
    if not isinstance(registry, dict) or registry.get("format_version") != 1:
        raise ArtifactError("Model registry has an unsupported or invalid format")
    if not isinstance(registry.get("models"), dict):
        raise ArtifactError("Model registry is missing its models mapping")
    return registry


def save_model_artifact(
    model: Any,
    *,
    model_name: str,
    model_version: str,
    model_dir: Path,
    metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Save a fitted pipeline and register its integrity digest.

    Only the save function registers artifacts. Consumers must protect the
    local registry from unauthorized writes; unregistered files are never
    deserialized.
    """
    artifact_path, metadata_path, registry_path = _artifact_paths(
        model_dir,
        model_name,
        model_version,
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.is_symlink():
        raise ArtifactError(f"Refusing to use a symlinked registry: {registry_path}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=artifact_path.parent,
            prefix=f".{artifact_path.stem}.",
            suffix=".joblib.tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        joblib.dump(model, temporary_path)
        artifact_hash = _sha256_file(temporary_path)
        temporary_path.replace(artifact_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    complete_metadata = {
        **dict(metadata),
        "model_name": model_name,
        "model_version": model_version,
        "artifact_file": artifact_path.name,
        "artifact_sha256": artifact_hash,
    }
    _atomic_write_json(metadata_path, complete_metadata)
    metadata_hash = _sha256_file(metadata_path)

    if registry_path.exists():
        registry = _load_registry(registry_path)
    else:
        registry = {"format_version": 1, "models": {}}
    registry["models"][f"{model_name}:{model_version}"] = {
        "artifact_sha256": artifact_hash,
        "metadata_sha256": metadata_hash,
        "metadata_file": metadata_path.name,
    }
    _atomic_write_json(registry_path, registry)
    return artifact_path, metadata_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as artifact_file:
            for block in iter(lambda: artifact_file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ArtifactError(f"Unable to read model artifact: {path}") from exc
    return digest.hexdigest()


def load_model_artifact(
    *,
    model_name: str,
    model_version: str,
    model_dir: Path,
    expected_feature_schema: Sequence[str] | Mapping[str, str] | None = None,
    expected_target_mapping: Mapping[str, Any] | None = None,
    expected_dependency_versions: Mapping[str, str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a registered model only after trust, integrity, and schema checks.

    The registry directory is a local trust boundary and must be writable only
    by trusted project operators. This function will not load arbitrary
    unregistered Joblib files.
    """
    artifact_path, metadata_path, registry_path = _artifact_paths(
        model_dir,
        model_name,
        model_version,
    )
    root = Path(model_dir).resolve()
    for path in (artifact_path, metadata_path):
        if path.is_symlink() or path.resolve().parent != root:
            raise ArtifactError(f"Refusing unsafe model artifact path: {path}")
    if not registry_path.exists() and artifact_path.is_file():
        raise ArtifactError(
            f"Model artifact is not registered: {model_name}:{model_version}"
        )
    registry = _load_registry(registry_path)
    key = f"{model_name}:{model_version}"
    entry = registry["models"].get(key)
    if not isinstance(entry, dict):
        raise ArtifactError(f"Model artifact is not registered: {key}")
    if entry.get("metadata_file") != metadata_path.name:
        raise ArtifactError(f"Registry metadata reference is invalid for {key}")
    if not artifact_path.is_file() or not metadata_path.is_file():
        raise ArtifactError(f"Model artifact or metadata is missing: {key}")

    metadata_hash = _sha256_file(metadata_path)
    if metadata_hash != entry.get("metadata_sha256"):
        raise ArtifactError(f"Model metadata integrity check failed: {key}")
    artifact_hash = _sha256_file(artifact_path)
    if artifact_hash != entry.get("artifact_sha256"):
        raise ArtifactError(f"Model artifact integrity check failed: {key}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Unable to read model metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ArtifactError(f"Invalid metadata for registered model: {key}")
    if (
        metadata.get("model_name") != model_name
        or metadata.get("model_version") != model_version
        or metadata.get("artifact_file") != artifact_path.name
        or metadata.get("artifact_sha256") != artifact_hash
    ):
        raise ArtifactError(f"Model metadata integrity check failed: {key}")

    schema = metadata.get("feature_schema")
    if expected_feature_schema is not None:
        if isinstance(expected_feature_schema, Mapping):
            expected_schema: Any = dict(expected_feature_schema)
        else:
            expected_schema = list(expected_feature_schema)
        if schema != expected_schema:
            raise ArtifactError(
                f"Feature schema mismatch for {key}: expected "
                f"{expected_schema!r}, found {schema!r}"
            )
    if (
        expected_target_mapping is not None
        and metadata.get("target_mapping") != dict(expected_target_mapping)
    ):
        raise ArtifactError(f"Target mapping mismatch for {key}")
    if (
        expected_dependency_versions is not None
        and metadata.get("dependency_versions") != dict(expected_dependency_versions)
    ):
        raise ArtifactError(f"Dependency version mismatch for {key}")

    try:
        model = joblib.load(artifact_path)
    except (
        AttributeError,
        EOFError,
        ImportError,
        ModuleNotFoundError,
        OSError,
        pickle.UnpicklingError,
        ValueError,
    ) as exc:
        raise ArtifactError(f"Unable to deserialize registered model {key}") from exc
    return model, metadata
