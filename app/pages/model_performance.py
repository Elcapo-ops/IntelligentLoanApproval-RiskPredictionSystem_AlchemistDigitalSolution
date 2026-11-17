"""Saved cross-validation comparisons and explicitly versioned holdout results."""

import pandas as pd
import streamlit as st

from app.components.charts import render_metric_comparison_chart
from app.components.metrics import render_metrics
from app.components.theme import (
    render_disclaimer,
    render_historical_approval_disclosure,
    render_page_header,
)
from app.services.dashboard_service import (
    load_holdout_evaluation_data,
    load_model_performance_data,
)
from utils.exceptions import LoanPredictionError


_COMPARISON_METRICS = (
    "accuracy_mean",
    "precision_mean",
    "recall_mean",
    "f1_mean",
    "roc_auc_mean",
)


def _render_holdout_evaluation() -> None:
    """Render the saved v1.0.0 holdout outputs without assigning them to v1.1."""
    try:
        evaluation = load_holdout_evaluation_data()
    except LoanPredictionError as exc:
        st.warning(f"Holdout evaluation report is unavailable: {exc}")
        return

    st.subheader(
        f"Historical holdout evaluation — "
        f"{evaluation['model_name']} v{evaluation['model_version']}"
    )
    st.caption(
        f"{evaluation['holdout_rows']} holdout records · evaluation report "
        f"modified {evaluation['report_date']} (best available date; no "
        f"evaluation timestamp is recorded in the source report) · positive "
        f"class `{evaluation['positive_class']}`."
    )
    st.warning(
        "This holdout result is for the baseline model/version shown above. "
        "The tuned v1.1.0 model has not been evaluated on this holdout; the "
        "holdout was previously viewed and is not pristine."
    )
    metrics = evaluation["metrics"]
    render_metrics(
        [
            {"label": "Holdout accuracy", "value": _format_metric(metrics["accuracy"])},
            {
                "label": "Holdout precision (Y)",
                "value": _format_metric(metrics["precision"]),
            },
            {"label": "Holdout recall (Y)", "value": _format_metric(metrics["recall"])},
            {"label": "Holdout F1 (Y)", "value": _format_metric(metrics["f1"])},
            {"label": "Holdout ROC-AUC", "value": _format_metric(metrics["roc_auc"])},
        ]
    )

    matrix = evaluation["confusion_matrix"]
    matrix_frame = pd.DataFrame(
        [
            [matrix["true_negative"], matrix["false_positive"]],
            [matrix["false_negative"], matrix["true_positive"]],
        ],
        index=["Actual N", "Actual Y"],
        columns=["Predicted N", "Predicted Y"],
    )
    matrix_column, figure_column = st.columns(2)
    with matrix_column:
        st.markdown("**Confusion matrix**")
        st.dataframe(matrix_frame, use_container_width=True)
        figure = evaluation["figures"]["confusion_matrix"]
        if figure is not None:
            st.image(str(figure), caption="Saved baseline holdout confusion matrix")
        else:
            st.warning("The saved confusion-matrix figure is missing.")
    with figure_column:
        st.markdown("**Holdout curves**")
        for key, caption in (
            ("roc_curve", "Baseline v1.0.0 holdout ROC curve"),
            ("precision_recall", "Baseline v1.0.0 holdout precision-recall curve"),
        ):
            figure = evaluation["figures"][key]
            if figure is None:
                st.warning(f"Saved {key.replace('_', ' ')} figure is missing.")
            else:
                st.image(str(figure), caption=caption, use_container_width=True)


def _format_metric(value: float | None) -> str:
    return "Unavailable" if value is None else f"{value:.4f}"


def render() -> None:
    """Render actual saved CV and holdout evaluation results."""
    render_page_header(
        "Validation",
        "Model Performance",
        "Saved evaluation outputs with each result labeled by model version and protocol.",
    )
    render_historical_approval_disclosure()
    render_disclaimer(
        "Metrics describe agreement with historical approval and rejection "
        "labels; they are not measures of repayment, default, or financial loss."
    )

    try:
        results = load_model_performance_data()
    except LoanPredictionError as exc:
        st.warning(f"Training cross-validation reports are unavailable: {exc}")
    else:
        candidates = results["candidate_cv"]
        st.subheader("Candidate comparison: training cross-validation")
        st.caption(
            f"{results['cv_folds']}-fold stratified CV on "
            f"{results['training_rows']} training-partition rows. "
            "The holdout is not included. Candidate metrics are mean ± fold SD."
        )
        st.caption(
            f"Tuned pipeline: {results['model_name']} v{results['model_version']}."
        )
        columns = ["model"]
        for metric in _COMPARISON_METRICS:
            columns.extend([metric, metric.replace("_mean", "_std")])
        comparison = candidates[columns].rename(
            columns={
                "model": "Candidate",
                "accuracy_mean": "Accuracy mean",
                "accuracy_std": "Accuracy SD",
                "precision_mean": "Precision (Y) mean",
                "precision_std": "Precision (Y) SD",
                "recall_mean": "Recall (Y) mean",
                "recall_std": "Recall (Y) SD",
                "f1_mean": "F1 (Y) mean",
                "f1_std": "F1 (Y) SD",
                "roc_auc_mean": "ROC-AUC mean",
                "roc_auc_std": "ROC-AUC SD",
            }
        )
        st.dataframe(
            comparison.style.format(
                {
                    column: "{:.3f}"
                    for column in comparison.columns
                    if column != "Candidate"
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        render_metric_comparison_chart(candidates, list(_COMPARISON_METRICS))
        figure = results.get("cv_figure")
        if figure is not None:
            st.image(
                str(figure),
                caption="Saved candidate comparison with cross-validation fold variation",
                use_container_width=True,
            )
        else:
            st.warning("The saved cross-validation comparison figure is missing.")

        tuning = results["tuning"]
        st.subheader(
            f"Tuned {results['model_name']} v{results['model_version']}"
        )
        render_metrics(
            [
                {
                    "label": "Tuned CV balanced accuracy",
                    "value": f"{tuning['tuned_balanced_accuracy_mean']:.3f}",
                    "help": (
                        f"Mean ± SD: {tuning['tuned_balanced_accuracy_mean']:.3f} "
                        f"± {tuning['tuned_balanced_accuracy_std']:.3f}."
                    ),
                },
                {
                    "label": "Tuned CV ROC-AUC",
                    "value": f"{tuning['tuned_roc_auc_mean']:.3f}",
                    "help": (
                        f"Mean ± SD: {tuning['tuned_roc_auc_mean']:.3f} "
                        f"± {tuning['tuned_roc_auc_std']:.3f}."
                    ),
                },
                {
                    "label": "CV balanced-accuracy change",
                    "value": f"{tuning['balanced_accuracy_delta']:+.3f}",
                    "help": (
                        "Tuned minus untuned CV mean on the same folds; "
                        "not evidence of a practically meaningful gain by itself."
                    ),
                },
            ]
        )
        with st.expander("Show selected search parameters"):
            st.json(tuning["best_params"])

    _render_holdout_evaluation()
    render_disclaimer(
        "The saved holdout metrics, confusion matrix, ROC curve, and "
        "precision-recall curve describe baseline v1.0.0 only. They are not "
        "results for tuned v1.1.0."
    )
