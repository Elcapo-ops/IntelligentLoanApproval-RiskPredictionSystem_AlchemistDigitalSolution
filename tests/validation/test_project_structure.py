from pathlib import Path

from config.settings import PROJECT_ROOT


def test_required_project_directories_exist() -> None:
    required_directories = (
        "data/raw",
        "data/processed",
        "data/external",
        "notebooks",
        "src/config",
        "src/data",
        "src/features",
        "src/preprocessing",
        "src/models",
        "src/evaluation",
        "src/explainability",
        "src/prediction",
        "src/utils",
        "app/pages",
        "app/components",
        "app/services",
        "artifacts/models",
        "artifacts/reports",
        "tests/unit",
        "tests/integration",
        "tests/validation",
        "docs",
        "logs",
    )

    for relative_path in required_directories:
        assert (PROJECT_ROOT / relative_path).is_dir(), relative_path


def test_required_root_files_exist() -> None:
    required_files = (
        "README.md",
        "requirements.txt",
        ".gitignore",
        "pyproject.toml",
        ".streamlit/config.toml",
        "config.yaml",
        "app.py",
        "app/pages/dashboard.py",
        "app/pages/loan_predictor.py",
        "app/pages/risk_analysis.py",
        "app/pages/model_performance.py",
        "app/pages/about.py",
        "app/services/dashboard_service.py",
        "app/services/prediction_service.py",
        "src/data/loader.py",
        "src/data/validation.py",
        "src/data/profiling.py",
        "src/data/quality_audit.py",
        "src/data/splitting.py",
        "src/features/engineering.py",
        "src/models/classifiers.py",
        "src/models/registry.py",
        "src/models/train.py",
        "src/evaluation/evaluate.py",
        "src/preprocessing/pipeline.py",
        "src/preprocessing/transformers.py",
        "docs/dataset_source.md",
        "notebooks/01_data_analysis.ipynb",
        "reports/eda_insights.md",
        "reports/model_training_report.md",
        "reports/model_artifact_report.md",
        "reports/evaluation_report.md",
        "artifacts/reports/model_cv_results.json",
        "artifacts/reports/global_permutation_importance.csv",
        "artifacts/reports/fairness_group_metrics.csv",
        "reports/model_comparison.csv",
        "notebooks/03_model_training.ipynb",
    )

    for relative_path in required_files:
        assert (PROJECT_ROOT / relative_path).is_file(), relative_path


def test_data_directories_keep_placeholders() -> None:
    for data_directory in ("raw", "processed", "external"):
        placeholder = PROJECT_ROOT / "data" / data_directory / ".gitkeep"
        assert placeholder.is_file()
