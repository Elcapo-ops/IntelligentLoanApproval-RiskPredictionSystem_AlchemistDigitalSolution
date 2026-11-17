"""Unified training script for all project models.

Trains both:
1. Historical Loan_Status approval classifier (Kaggle loan-prediction dataset)
2. Independent LendingClub default-risk classifier

Usage (from project root):
    python scripts/train_all_models.py

Prerequisites:
    - Place Kaggle CSVs in data/raw/ (train_u6lujuX_CVtuZ9i.csv, test_Y3wMUE5_7gLdaTN.csv)
    - Place LendingClub archive in data/external/ (accepted_loan_data.csv or .csv.gz)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path when run as `python scripts/train_all_models.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def train_approval_model(config: dict, project_root: Path) -> dict:
    """Train the historical Loan_Status approval model from Kaggle data."""
    from config.settings import load_config, validate_config
    from config.paths import get_project_paths
    from data.loader import load_csv
    from models.train import train_models

    settings = validate_config(config)
    paths = get_project_paths(settings)

    labeled_path = paths.raw_data_dir / "train_u6lujuX_CVtuZ9i.csv"
    if not labeled_path.exists():
        raise FileNotFoundError(
            f"Kaggle labeled dataset not found: {labeled_path}\n"
            "Download from Kaggle and place it in data/raw/ before training."
        )

    labeled_frame = load_csv(labeled_path)
    result = train_models(labeled_frame, config=settings)

    return {
        "model": "historical_approval",
        "cv_results": result.cv_results.to_dict(orient="records"),
        "artifacts": {k: str(v) for k, v in result.model_paths.items()},
        "report": str(result.report_path),
        "results": str(result.results_path),
        "training_rows": len(result.split.X_train),
        "holdout_rows_reserved": len(result.split.X_test),
    }


def train_default_risk_model_script(project_root: Path, args: argparse.Namespace) -> dict:
    """Train the independent LendingClub default-risk model."""
    from default_risk.train import main as default_risk_main

    # Temporarily replace argv so the default_risk CLI parses our arguments
    old_argv = sys.argv
    sys.argv = [
        "default_risk.train",
        "--project-root", str(project_root),
    ]
    if args.dataset:
        sys.argv += ["--dataset", str(args.dataset)]
    if args.test_start_year is not None:
        sys.argv += ["--test-start-year", str(args.test_start_year)]
    if args.max_train_rows is not None:
        sys.argv += ["--max-train-rows", str(args.max_train_rows)]
    if args.max_test_rows is not None:
        sys.argv += ["--max-test-rows", str(args.max_test_rows)]
    if args.cv_folds is not None:
        sys.argv += ["--cv-folds", str(args.cv_folds)]
    if args.random_state is not None:
        sys.argv += ["--random-state", str(args.random_state)]

    try:
        default_risk_main()
    finally:
        sys.argv = old_argv

    return {
        "model": "default_risk",
        "note": "See console output above and artifacts/models/default_risk/ for artifacts.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train all project models from local datasets."
    )
    parser.add_argument(
        "--skip-approval",
        action="store_true",
        help="Skip the historical Loan_Status approval model.",
    )
    parser.add_argument(
        "--skip-default-risk",
        action="store_true",
        help="Skip the independent LendingClub default-risk model.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to LendingClub accepted-loan CSV (or .csv.gz).",
    )
    parser.add_argument("--test-start-year", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)

    args = parser.parse_args()

    config_path = PROJECT_ROOT / "config.yaml"
    import yaml

    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    results: list[dict] = []

    if not args.skip_approval:
        print("=" * 70)
        print("Training historical Loan_Status approval model...")
        print("=" * 70)
        try:
            approval_result = train_approval_model(config, PROJECT_ROOT)
            results.append(approval_result)
            print("Approval model training complete.")
            print(f"  Artifacts: {approval_result['artifacts']}")
            print(f"  Report: {approval_result['report']}")
        except Exception as exc:  # noqa: BLE001
            print(f"Approval model training FAILED: {exc}")
            results.append({"model": "historical_approval", "error": str(exc)})

    if not args.skip_default_risk:
        print("=" * 70)
        print("Training independent LendingClub default-risk model...")
        print("=" * 70)
        try:
            default_risk_result = train_default_risk_model_script(
                PROJECT_ROOT, args
            )
            results.append(default_risk_result)
            print("Default-risk model training complete.")
        except Exception as exc:  # noqa: BLE001
            print(f"Default-risk model training FAILED: {exc}")
            results.append({"model": "default_risk", "error": str(exc)})

    print("=" * 70)
    print("ALL TRAINING FINISHED")
    print("=" * 70)
    print(json.dumps(results, indent=2, allow_nan=False))

    # Write summary for CI / collaborators
    summary_path = PROJECT_ROOT / "artifacts" / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
