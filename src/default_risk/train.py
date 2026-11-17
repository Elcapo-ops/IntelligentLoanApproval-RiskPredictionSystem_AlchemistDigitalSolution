"""Command-line entry point for the independent LendingClub default model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from default_risk.modeling import train_default_risk_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the separate LendingClub default-risk model."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to accepted LendingClub CSV.gz file (defaults to project data path).",
    )
    parser.add_argument("--test-start-year", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    arguments = parser.parse_args()
    result = train_default_risk_model(
        project_root=arguments.project_root,
        dataset_path=arguments.dataset,
        test_start_year=arguments.test_start_year,
        max_train_rows=arguments.max_train_rows,
        max_test_rows=arguments.max_test_rows,
        cv_folds=arguments.cv_folds,
        random_state=arguments.random_state,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
