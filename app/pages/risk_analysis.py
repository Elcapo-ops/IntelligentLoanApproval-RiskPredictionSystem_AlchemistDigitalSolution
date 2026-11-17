"""Historical approval patterns, predictive importance, and fairness review."""

import streamlit as st

from app.components.charts import render_historical_approval_pattern
from app.components.theme import (
    render_disclaimer,
    render_historical_approval_disclosure,
    render_page_header,
)
from app.services.dashboard_service import load_explainability_data
from utils.exceptions import LoanPredictionError


def _show_optional_markdown(text: str | None, label: str) -> None:
    if text:
        st.markdown(text)
    else:
        st.info(f"{label} details are unavailable in the saved reports.")


def render() -> None:
    """Render saved global importance, observed patterns, and subgroup metrics."""
    render_page_header(
        "Model review",
        "Risk Analysis",
        "Historical Approval Patterns, model sensitivity, and an exploratory fairness review.",
    )
    render_historical_approval_disclosure()
    render_disclaimer(
        "This page analyzes historical approval labels and model behavior. "
        "Its error rates refer to historical approval/rejection classes, not "
        "loan defaults or repayments."
    )
    try:
        outputs = load_explainability_data()
    except LoanPredictionError as exc:
        st.error(f"Risk analysis data could not be loaded: {exc}")
        return

    for warning in outputs["warnings"]:
        st.warning(warning)

    st.subheader("Historical Approval Patterns")
    if outputs["historical_application_count"] is not None:
        st.caption(
            f"Computed from {outputs['historical_application_count']} labeled "
            "source records. These are unadjusted descriptive rates, not "
            "causal effects or current portfolio statistics."
        )
    else:
        st.info("Historical approval pattern data is unavailable.")
    patterns = outputs["approval_patterns"]
    if patterns:
        titles = {
            "Property_Area": "Property area",
            "Credit_History": "Credit history",
            "Self_Employed": "Self-employment",
        }
        features = list(patterns)
        tabs = st.tabs([titles.get(feature, feature) for feature in features])
        for tab, feature in zip(tabs, features):
            with tab:
                data = patterns[feature]
                st.markdown(f"Historical approval by {titles.get(feature, feature)}")
                render_historical_approval_pattern(data, feature)
                st.dataframe(
                    data.rename(
                        columns={
                            feature: "Group",
                            "applications": "Applications",
                            "approvals": "Historical approvals",
                            "approval_rate": "Historical approval rate",
                        }
                    ).style.format(
                        {"Historical approval rate": "{:.1%}"},
                        na_rep="Not available",
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
    else:
        st.info("No saved labeled-data group summaries are available.")

    st.divider()
    st.subheader("Global predictive feature importance")
    st.caption(
        "Saved out-of-fold permutation importance on original inputs. "
        "It indicates predictive contribution within this dataset, not "
        "causation; a score decrease is model sensitivity."
    )
    importance = outputs["importance"]
    if importance is not None:
        display_importance = importance.rename(
            columns={
                "feature": "Original input feature",
                "importance_mean": "Mean CV score decrease",
                "importance_fold_std": "Fold standard deviation",
                "importance_repeat_std": "Permutation-repeat standard deviation",
            }
        )
        st.dataframe(
            display_importance.style.format(
                {
                    column: "{:.4f}"
                    for column in display_importance.columns
                    if column != "Original input feature"
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        if outputs["importance_figure"] is not None:
            st.image(
                str(outputs["importance_figure"]),
                caption="Saved training-fold permutation importance",
                use_container_width=True,
            )
        else:
            st.warning("The saved permutation-importance chart is missing.")
    else:
        st.info("Global feature-importance values are unavailable.")

    st.subheader("SHAP")
    if outputs["shap_figures"]:
        for figure in outputs["shap_figures"]:
            st.image(str(figure), caption=figure.stem, use_container_width=True)
    elif outputs["shap_status"]:
        st.markdown(outputs["shap_status"])
    else:
        st.info("No saved SHAP report or chart is available; no SHAP values are inferred.")

    st.subheader("Exploratory fairness analysis")
    st.caption(
        "Saved out-of-fold group results are descriptive, depend on available "
        "attributes and group sizes, and are not a legal fairness determination."
    )
    st.info(
        "Fairness and robustness checks are restricted to historical approval patterns; "
        "they do not establish fairness in repayment, default, or real-world lending decisions."
    )
    groups = outputs["groups"]
    if groups is not None:
        display_groups = groups.rename(
            columns={
                "attribute": "Attribute",
                "group": "Group",
                "n": "Applications",
                "eligible_for_comparison": "Meets minimum group size",
                "approval_rate": "Predicted historical approval-label rate",
                "observed_approval_rate": "Historical approval rate",
                "true_positive_rate": "Recall for historical approvals",
                "false_positive_rate": "False positive rate for historical rejections",
            }
        )
        st.dataframe(
            display_groups.style.format(
                {
                    "Predicted historical approval-label rate": "{:.1%}",
                    "Historical approval rate": "{:.1%}",
                    "Recall for historical approvals": "{:.1%}",
                    "False positive rate for historical rejections": "{:.1%}",
                },
                na_rep="Not estimable",
            ),
            use_container_width=True,
            hide_index=True,
        )
        if outputs["figure"] is not None:
            st.image(
                str(outputs["figure"]),
                caption=(
                    "Saved training out-of-fold subgroup rates; not default "
                    "or loss rates."
                ),
                use_container_width=True,
            )
        else:
            st.warning("The saved subgroup-rate chart is missing.")
    else:
        st.info("Saved fairness group metrics are unavailable.")
    with st.expander("Fairness report interpretation"):
        _show_optional_markdown(
            outputs["fairness_interpretation"],
            "Fairness interpretation",
        )

    st.subheader("Dataset limitations")
    with st.expander("Read limitations from the data-quality audit", expanded=True):
        _show_optional_markdown(
            outputs["quality_limitations"],
            "Data-quality limitations",
        )
    st.caption(
        "Source reports: "
        + ", ".join(
            name
            for name, available in (
                ("global permutation importance", importance is not None),
                ("fairness group metrics", groups is not None),
                ("explainability report", outputs["explainability_report"] is not None),
                ("data-quality report", outputs["quality_report"] is not None),
            )
            if available
        )
    )
    render_disclaimer(
        "Subgroup gaps may reflect sample size, missingness, historical policy, "
        "label quality, or omitted factors. This analysis does not prove "
        "discrimination or its absence and is not sufficient for credit decisions."
    )
