import pytest

from config.paths import get_project_paths
from config.settings import load_config
from data.loader import load_datasets
from data.splitting import get_model_feature_columns

_PROJECT_PATHS = get_project_paths(load_config())
_ACQUIRED_DATA_AVAILABLE = (
    _PROJECT_PATHS.labeled_dataset.is_file()
    and _PROJECT_PATHS.unlabeled_dataset.is_file()
)

@pytest.mark.skipif(
    not _ACQUIRED_DATA_AVAILABLE,
    reason="Acquired Kaggle CSVs are not included in source control",
)
def test_acquired_data_exists_and_meets_dataset_contract() -> None:
    config = load_config()
    paths = get_project_paths(config)
    assert paths.labeled_dataset.is_file()
    assert paths.unlabeled_dataset.is_file()

    datasets = load_datasets(config=config)
    labeled = datasets.labeled
    unlabeled = datasets.unlabeled
    target = config["dataset"]["target_column"]
    feature_columns = set(get_model_feature_columns(config))

    assert target in labeled.columns
    assert target not in unlabeled.columns
    assert set(labeled[target].dropna().unique()) == set(
        config["dataset"]["target_classes"]
    )
    assert labeled[target].notna().all()
    assert set(labeled.columns).difference({target}) == feature_columns.union(
        config["dataset"]["identifier_columns"]
    )
    assert set(unlabeled.columns) == feature_columns.union(
        config["dataset"]["identifier_columns"]
    )
    assert not labeled.duplicated().any()
    assert not unlabeled.duplicated().any()
    for identifier in datasets.identifier_columns:
        assert labeled[identifier].notna().all()
        assert labeled[identifier].is_unique
        assert unlabeled[identifier].notna().all()
        assert unlabeled[identifier].is_unique
        assert set(labeled[identifier]).isdisjoint(unlabeled[identifier])

    missing_counts = labeled.isna().sum()
    assert int(missing_counts.sum()) > 0
    assert missing_counts[target] == 0
    assert int(missing_counts["Credit_History"]) == 50
