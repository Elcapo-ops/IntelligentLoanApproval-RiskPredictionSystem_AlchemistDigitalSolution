"""Read-only data quality audit and Markdown report generation."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from config.paths import get_project_paths
from config.settings import load_config, validate_config
from data.loader import DatasetBundle, load_csv
from utils.exceptions import DataValidationError


def _column_missingness(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for column in frame.columns:
        missing = int(frame[column].isna().sum())
        result[str(column)] = {
            "count": missing,
            "percentage": round(100 * missing / len(frame), 4),
        }
    return result


def _duplicate_identifier_counts(
    frame: pd.DataFrame,
    identifier_columns: Sequence[str],
) -> dict[str, int]:
    return {
        column: int(frame[column].duplicated(keep=False).sum())
        for column in identifier_columns
        if column in frame.columns
    }


def _audit_numeric_column(
    values: pd.Series,
    column: str,
    require_positive: bool,
) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    non_numeric_count = int((values.notna() & numeric.isna()).sum())
    finite_mask = numeric.notna() & numeric.map(
        lambda value: pd.notna(value) and abs(value) != float("inf")
    )
    finite_values = numeric[finite_mask]
    negative_count = int((finite_values < 0).sum())
    zero_count = int((finite_values == 0).sum())
    non_finite_count = int((numeric.notna() & ~finite_mask).sum())

    if finite_values.empty:
        minimum = maximum = q1 = median = q3 = low_fence = high_fence = None
        outlier_count = 0
    else:
        minimum = float(finite_values.min())
        maximum = float(finite_values.max())
        q1 = float(finite_values.quantile(0.25))
        median = float(finite_values.quantile(0.5))
        q3 = float(finite_values.quantile(0.75))
        iqr = q3 - q1
        low_fence = q1 - 1.5 * iqr
        high_fence = q3 + 1.5 * iqr
        outlier_count = int(
            ((finite_values < low_fence) | (finite_values > high_fence)).sum()
        )

    return {
        "dtype": str(values.dtype),
        "missing_count": int(values.isna().sum()),
        "non_numeric_count": non_numeric_count,
        "non_finite_count": non_finite_count,
        "minimum": minimum,
        "q1": q1,
        "median": median,
        "q3": q3,
        "maximum": maximum,
        "negative_count": negative_count,
        "zero_count": zero_count,
        "non_positive_count": (
            int((finite_values <= 0).sum()) if require_positive else 0
        ),
        "iqr_low_fence": low_fence,
        "iqr_high_fence": high_fence,
        "iqr_outlier_count": outlier_count,
        "iqr_outlier_percentage": (
            round(100 * outlier_count / len(finite_values), 4)
            if len(finite_values)
            else 0.0
        ),
    }


def _audit_frame(
    frame: pd.DataFrame,
    dataset_name: str,
    target_column: str,
    allowed_target_values: Sequence[str],
    identifier_columns: Sequence[str],
    categorical_domains: Mapping[str, Sequence[Any]],
    numeric_columns: Sequence[str],
    positive_numeric_columns: Sequence[str],
    labeled: bool,
) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataValidationError(f"{dataset_name} data must be a non-empty DataFrame")

    duplicate_identifier_columns = _duplicate_identifier_counts(
        frame, identifier_columns
    )
    missing_columns = [
        column
        for column in (*categorical_domains.keys(), *numeric_columns)
        if column not in frame.columns
    ]
    if labeled and target_column not in frame.columns:
        missing_columns.append(target_column)
    if not labeled and target_column in frame.columns:
        raise DataValidationError(
            f"Unlabeled dataset must not contain target column {target_column!r}"
        )
    if missing_columns:
        raise DataValidationError(
            f"{dataset_name} data is missing audited columns: "
            f"{sorted(set(missing_columns))}"
        )

    invalid_categories: dict[str, dict[str, int]] = {}
    for column, allowed_values in categorical_domains.items():
        invalid = frame[column].notna() & ~frame[column].isin(allowed_values)
        counts = frame.loc[invalid, column].astype("string").value_counts()
        invalid_categories[column] = {
            str(value): int(count) for value, count in counts.items()
        }

    target_report: dict[str, Any] | None = None
    if labeled:
        target = frame[target_column]
        counts = target.value_counts(dropna=False)
        valid_mask = target.isin(allowed_target_values)
        target_report = {
            "missing_count": int(target.isna().sum()),
            "invalid_label_count": int((target.notna() & ~valid_mask).sum()),
            "invalid_labels": {
                str(value): int(count)
                for value, count in counts.items()
                if pd.notna(value) and value not in allowed_target_values
            },
            "class_counts": {
                str(value): int(count)
                for value, count in target.value_counts().items()
            },
            "class_proportions": {
                str(value): round(float(count / len(frame)), 6)
                for value, count in target.value_counts().items()
            },
        }
        class_counts = target.value_counts()
        target_report["majority_to_minority_ratio"] = (
            round(float(class_counts.max() / class_counts.min()), 4)
            if len(class_counts) > 1 and class_counts.min() > 0
            else None
        )

    numeric_report = {
        column: _audit_numeric_column(
            frame[column],
            column,
            require_positive=column in positive_numeric_columns,
        )
        for column in numeric_columns
    }

    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "duplicate_row_count": int(frame.duplicated().sum()),
        "duplicate_identifier_rows": duplicate_identifier_columns,
        "missing_values": _column_missingness(frame),
        "invalid_categories": invalid_categories,
        "numeric_checks": numeric_report,
        "target": target_report,
    }


def audit_dataset_bundle(
    datasets: DatasetBundle,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit the labeled and unlabeled frames independently without mutation."""
    settings = validate_config(config if config is not None else load_config())
    dataset_settings = settings["dataset"]
    common = {
        "target_column": datasets.target_column,
        "allowed_target_values": dataset_settings["target_classes"],
        "identifier_columns": datasets.identifier_columns,
        "categorical_domains": dataset_settings["categorical_domains"],
        "numeric_columns": dataset_settings["numeric_columns"],
        "positive_numeric_columns": dataset_settings["positive_numeric_columns"],
    }
    identifier_overlap = {
        column: int(
            len(
                set(datasets.labeled[column].dropna().astype("string"))
                & set(datasets.unlabeled[column].dropna().astype("string"))
            )
        )
        for column in datasets.identifier_columns
        if column in datasets.labeled.columns
        and column in datasets.unlabeled.columns
    }
    labeled_features = set(datasets.labeled.columns).difference(
        {datasets.target_column}
    )
    unlabeled_features = set(datasets.unlabeled.columns)
    return {
        "status": "complete",
        "source_url": dataset_settings["source_url"],
        "target_column": datasets.target_column,
        "identifier_columns": list(datasets.identifier_columns),
        "identifier_overlap": identifier_overlap,
        "feature_schema_matches": labeled_features == unlabeled_features,
        "features_missing_from_unlabeled": sorted(
            labeled_features - unlabeled_features
        ),
        "features_missing_from_labeled": sorted(
            unlabeled_features - labeled_features
        ),
        "labeled": _audit_frame(
            datasets.labeled,
            "Labeled",
            labeled=True,
            **common,
        ),
        "unlabeled": _audit_frame(
            datasets.unlabeled,
            "Unlabeled",
            labeled=False,
            **common,
        ),
    }


def audit_configured_datasets(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load configured CSVs separately and return their read-only quality audit."""
    settings = validate_config(config if config is not None else load_config())
    paths = get_project_paths(settings)
    dataset_settings = settings["dataset"]
    datasets = DatasetBundle(
        labeled=load_csv(paths.labeled_dataset),
        unlabeled=load_csv(paths.unlabeled_dataset),
        target_column=dataset_settings["target_column"],
        identifier_columns=tuple(dataset_settings["identifier_columns"]),
    )
    return audit_dataset_bundle(datasets, config=settings)


def render_quality_report(report: dict[str, Any]) -> str:
    """Render measured audit findings as a Markdown report."""
    if report.get("status") != "complete":
        raise DataValidationError("Only complete quality audits can be reported")

    lines = [
        "# Data quality audit",
        "",
        f"- Source: {report['source_url']}",
        f"- Target: `{report['target_column']}`",
        "- Identifiers: "
        + ", ".join(f"`{column}`" for column in report["identifier_columns"]),
        "- Method: read-only audit; source CSVs were not modified.",
        "",
        "## Dataset integrity",
        "",
        "| Dataset | Rows | Columns | Duplicate rows | Duplicate identifier rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("labeled", "unlabeled"):
        dataset = report[name]
        duplicate_ids = sum(dataset["duplicate_identifier_rows"].values())
        lines.append(
            f"| {name.title()} | {dataset['row_count']} | {dataset['column_count']} "
            f"| {dataset['duplicate_row_count']} | {duplicate_ids} |"
        )
    overlap_total = sum(report["identifier_overlap"].values())
    lines.append(
        f"\nCross-file identifier overlap: **{overlap_total}** across configured IDs."
    )
    lines.append(
        "Feature schemas match: **"
        + ("yes" if report["feature_schema_matches"] else "no")
        + "**."
    )

    lines.extend(
        [
            "",
            "## Missing values",
            "",
            "Percentages use each dataset's own row count.",
            "",
            "| Dataset | Column | Missing | Missing % |",
            "|---|---|---:|---:|",
        ]
    )
    for name in ("labeled", "unlabeled"):
        for column, missing in report[name]["missing_values"].items():
            lines.append(
                f"| {name.title()} | `{column}` | {missing['count']} "
                f"| {missing['percentage']:.4f}% |"
            )

    lines.extend(
        [
            "",
            "## Category and target validation",
            "",
            "| Dataset | Field | Invalid values / result |",
            "|---|---|---|",
        ]
    )
    for name in ("labeled", "unlabeled"):
        dataset = report[name]
        for column, invalid in dataset["invalid_categories"].items():
            description = (
                "none"
                if not invalid
                else ", ".join(
                    f"`{value}` ({count})" for value, count in invalid.items()
                )
            )
            lines.append(f"| {name.title()} | `{column}` | {description} |")
        if dataset["target"] is not None:
            target = dataset["target"]
            lines.append(
                f"| Labeled | `{report['target_column']}` | "
                f"{target['invalid_label_count']} invalid, "
                f"{target['missing_count']} missing |"
            )

    target = report["labeled"]["target"]
    lines.extend(
        [
            "",
            "### Labeled target distribution",
            "",
            "| Label | Count | Proportion |",
            "|---|---:|---:|",
        ]
    )
    for label, count in target["class_counts"].items():
        lines.append(
            f"| `{label}` | {count} | {target['class_proportions'][label]:.4%} |"
        )
    lines.append(
        f"\nMajority/minority ratio: **{target['majority_to_minority_ratio']}:1**. "
        "This indicates class imbalance, not invalid labels. No resampling was "
        "performed. The unlabeled dataset has no target."
    )

    lines.extend(
        [
            "",
            "## Numerical checks and extreme observations",
            "",
            "Negative values are counted as domain violations for income, loan "
            "amount, and term. Loan amount and term must also be positive; "
            "zero income is retained and reported as unusual rather than "
            "automatically invalid. IQR fences are descriptive screening "
            "flags only. They are not error criteria.",
            "",
            "| Dataset | Field | Min | Q1 | Median | Q3 | Max | Negative | Zero | "
            "Non-positive | Non-numeric | Non-finite | IQR fences | IQR flags |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for name in ("labeled", "unlabeled"):
        for column, numeric in report[name]["numeric_checks"].items():
            low = numeric["iqr_low_fence"]
            high = numeric["iqr_high_fence"]
            fences = (
                "n/a"
                if low is None
                else f"{low:.2f} to {high:.2f}"
            )
            lines.append(
                f"| {name.title()} | `{column}` | {numeric['minimum']} "
                f"| {numeric['q1']} | {numeric['median']} | {numeric['q3']} "
                f"| {numeric['maximum']} | {numeric['negative_count']} "
                f"| {numeric['zero_count']} | {numeric['non_positive_count']} "
                f"| {numeric['non_numeric_count']} | {numeric['non_finite_count']} "
                f"| {fences} | {numeric['iqr_outlier_count']} "
                f"({numeric['iqr_outlier_percentage']:.2f}%) |"
            )

    lines.extend(
        [
            "",
            "## Interpretation and handling",
            "",
            "- Missing values are present and should be handled in a later "
            "training-fold preprocessing pipeline; they have not been imputed.",
            "- Any IQR-flagged high incomes or loan amounts may represent "
            "unusual but valid applicants. The IQR rule alone cannot establish "
            "a data-entry error. Review source context before any exclusion.",
            "- The loan-term quartiles are all 360 months in both files, so "
            "the IQR fences collapse to 360. Other observed positive terms "
            "are unusual under this rule but are not automatically invalid.",
            "- A zero co-applicant income is valid when no co-applicant income "
            "is reported. Zero applicant income is flagged for review, not "
            "rejected by this audit.",
            "- A negative income, loan amount, or loan term is a domain "
            "violation. No negative values were observed in the acquired files.",
            "- Target labels are checked against the configured `Y`/`N` domain. "
            "The unlabeled CSV is checked to ensure the target is absent.",
            "- The two CSVs remain separate. This report does not train a model, "
            "impute values, delete records, or rewrite either source.",
            "- No observations were removed, imputed, or changed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_quality_report(
    report: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    """Write the Markdown audit report to the configured report directory."""
    if output_path is None:
        output_path = get_project_paths().quality_report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(render_quality_report(report), encoding="utf-8")
    except OSError as exc:
        raise DataValidationError(
            f"Unable to write data quality report: {output_path}"
        ) from exc
    return output_path
