"""Resolve configured dataset and artifact paths from a stable project root."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import PROJECT_ROOT, load_config
from utils.exceptions import ConfigurationError


def _resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved paths for project data, artifacts, and logs."""

    root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    external_data_dir: Path
    raw_dataset: Path
    labeled_dataset: Path
    unlabeled_dataset: Path
    processed_dataset: Path
    models_dir: Path
    reports_dir: Path
    cv_results: Path
    evaluation_report: Path
    quality_report: Path
    tuning_report: Path
    tuning_results: Path
    business_insights_report: Path
    explainability_report: Path
    global_importance: Path
    fairness_results: Path
    local_explanations: Path
    figures_dir: Path
    log_file: Path

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        root: Path = PROJECT_ROOT,
    ) -> "ProjectPaths":
        """Create paths from validated configuration and an optional root."""
        path_settings = config.get("paths")
        if not isinstance(path_settings, dict):
            raise ConfigurationError("Missing or invalid configuration section: paths")
        missing = [
            key
            for key in (
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
            if not isinstance(path_settings.get(key), str)
            or not path_settings[key].strip()
        ]
        if missing:
            raise ConfigurationError(
                f"Missing or invalid path settings: {', '.join(missing)}"
            )

        root = root.resolve()
        resolved = {
            key: _resolve(root, path_settings[key])
            for key in path_settings
            if key
            in {
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
            }
        }
        return cls(root=root, **resolved)

    def ensure_directories(self) -> None:
        """Create configured data, artifact, report, and log directories."""
        for directory in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.external_data_dir,
            self.models_dir,
            self.reports_dir,
            self.cv_results.parent,
            self.evaluation_report.parent,
            self.quality_report.parent,
            self.tuning_report.parent,
            self.tuning_results.parent,
            self.business_insights_report.parent,
            self.explainability_report.parent,
            self.global_importance.parent,
            self.fairness_results.parent,
            self.local_explanations.parent,
            self.figures_dir,
            self.log_file.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def get_project_paths(config: dict[str, Any] | None = None) -> ProjectPaths:
    """Return configured paths, loading the canonical config when omitted."""
    return ProjectPaths.from_config(config if config is not None else load_config())
