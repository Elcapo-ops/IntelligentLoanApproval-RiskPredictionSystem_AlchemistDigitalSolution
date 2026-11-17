import json
from pathlib import Path

import numpy as np
import pytest
from joblib import dump
from sklearn.linear_model import LogisticRegression

from models.registry import (
    dataframe_sha256,
    dependency_versions,
    load_model_artifact,
    save_model_artifact,
)
from utils.exceptions import ArtifactError


@pytest.fixture
def registered_model(tmp_path: Path) -> tuple[Path, LogisticRegression, dict]:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression().fit(X, y)
    metadata = {
        "training_date": "2026-11-14T00:00:00+00:00",
        "dataset_hash": "a" * 64,
        "random_seed": 42,
        "feature_schema": {"income": "float64"},
        "target_mapping": {"N": 0, "Y": 1},
        "dependency_versions": dependency_versions(),
        "training_configuration": {"cv_folds": 5},
    }
    save_model_artifact(
        model,
        model_name="logistic_regression",
        model_version="1.0.0",
        model_dir=tmp_path,
        metadata=metadata,
    )
    return tmp_path, model, metadata


def test_saves_and_loads_registered_artifact_with_provenance(
    registered_model: tuple[Path, LogisticRegression, dict],
) -> None:
    model_dir, original, expected_metadata = registered_model

    loaded, metadata = load_model_artifact(
        model_name="logistic_regression",
        model_version="1.0.0",
        model_dir=model_dir,
        expected_feature_schema={"income": "float64"},
        expected_target_mapping={"N": 0, "Y": 1},
    )

    np.testing.assert_array_equal(
        loaded.predict(np.array([[0.5], [2.5]])),
        original.predict(np.array([[0.5], [2.5]])),
    )
    assert metadata["dataset_hash"] == expected_metadata["dataset_hash"]
    assert metadata["random_seed"] == 42
    assert metadata["dependency_versions"] == dependency_versions()
    assert len(metadata["artifact_sha256"]) == 64


def test_rejects_unregistered_joblib_file(tmp_path: Path) -> None:
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    dump(model, tmp_path / "unregistered-1.0.0.joblib")

    with pytest.raises(ArtifactError, match="not registered"):
        load_model_artifact(
            model_name="unregistered",
            model_version="1.0.0",
            model_dir=tmp_path,
        )


def test_rejects_modified_artifact_before_deserialization(
    registered_model: tuple[Path, LogisticRegression, dict],
) -> None:
    model_dir, _, _ = registered_model
    artifact = model_dir / "logistic_regression-1.0.0.joblib"
    artifact.write_bytes(b"modified artifact")

    with pytest.raises(ArtifactError, match="integrity check failed"):
        load_model_artifact(
            model_name="logistic_regression",
            model_version="1.0.0",
            model_dir=model_dir,
        )


def test_rejects_modified_metadata(
    registered_model: tuple[Path, LogisticRegression, dict],
) -> None:
    model_dir, _, _ = registered_model
    metadata_path = model_dir / "logistic_regression-1.0.0.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["random_seed"] = 999
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactError, match="metadata integrity check failed"):
        load_model_artifact(
            model_name="logistic_regression",
            model_version="1.0.0",
            model_dir=model_dir,
        )


def test_rejects_missing_artifact(
    registered_model: tuple[Path, LogisticRegression, dict],
) -> None:
    model_dir, _, _ = registered_model
    (model_dir / "logistic_regression-1.0.0.joblib").unlink()

    with pytest.raises(ArtifactError, match="artifact or metadata is missing"):
        load_model_artifact(
            model_name="logistic_regression",
            model_version="1.0.0",
            model_dir=model_dir,
        )


@pytest.mark.parametrize(
    ("schema", "target_mapping", "message"),
    [
        ({"income": "int64"}, {"N": 0, "Y": 1}, "Feature schema mismatch"),
        ({"income": "float64"}, {"N": 1, "Y": 0}, "Target mapping mismatch"),
    ],
)
def test_rejects_incompatible_expected_schema(
    registered_model: tuple[Path, LogisticRegression, dict],
    schema: dict,
    target_mapping: dict,
    message: str,
) -> None:
    model_dir, _, _ = registered_model

    with pytest.raises(ArtifactError, match=message):
        load_model_artifact(
            model_name="logistic_regression",
            model_version="1.0.0",
            model_dir=model_dir,
            expected_feature_schema=schema,
            expected_target_mapping=target_mapping,
        )


def test_rejects_missing_registry_and_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="registry is missing"):
        load_model_artifact(
            model_name="missing",
            model_version="1.0.0",
            model_dir=tmp_path,
        )
    with pytest.raises(ArtifactError, match="Invalid model artifact name"):
        load_model_artifact(
            model_name="..\\outside",
            model_version="1.0.0",
            model_dir=tmp_path,
        )


def test_dataframe_hash_tracks_values_and_schema() -> None:
    import pandas as pd

    frame = pd.DataFrame({"income": [100.0, 200.0]})
    assert dataframe_sha256(frame) == dataframe_sha256(frame.copy())
    assert dataframe_sha256(frame) != dataframe_sha256(
        pd.DataFrame({"income": [100.0, 201.0]})
    )
    assert dataframe_sha256(frame) != dataframe_sha256(
        frame.astype({"income": "int64"})
    )
