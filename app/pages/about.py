"""Project scope, model provenance, and limitations."""

from pathlib import Path

import streamlit as st

from app.components.theme import (
    render_disclaimer,
    render_historical_approval_disclosure,
    render_page_header,
)
from config.paths import get_project_paths
from config.settings import load_config


def render() -> None:
    """Render transparent project provenance and intended-use guidance."""
    render_page_header(
        "Project information",
        "About this prototype",
        "An academic decision-support project for historical loan approval labels.",
    )
    render_historical_approval_disclosure()
    config = load_config()
    paths = get_project_paths(config)
    st.markdown(
        """
        ### Project title and actual model scope

        The academic project title is **Intelligent Loan Approval & Credit
        Risk Prediction System**. Despite the title, the implemented model
        estimates historical `Loan_Status` labels only; it is not a default,
        repayment, credit-score, or real-world lending-decision model.

        ### Purpose

        This application demonstrates how a complete machine-learning pipeline
        can summarize a small historical loan-approval dataset, produce an
        experimental approval-label estimate, and expose model and data
        limitations. It is not a production lending system or autonomous
        decision engine.

        ### Prediction target

        The model target is `Loan_Status`: `Y` means the application was
        historically approved and `N` means it was historically rejected.
        There are no repayment/default outcomes in the model target. Approval
        No validated default or repayment labels are available. The live
        application withholds probabilities because calibration has not been
        validated.

        ### Data source

        The labeled data is the
        [Kaggle Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset).
        The unlabeled Kaggle test CSV is kept separate and is not used as
        labeled training or historical dashboard data.

        ### Model and limitations

        The predictor loads a registered, versioned pipeline with its
        preprocessing steps. The active application model is specified in
        `config.yaml`. Historical dashboard rates are computed from the
        labeled file; live predictions are counted only within this browser
        session. The evaluation holdout seen in Phase 5.1 was for baseline
        v1.0.0, not the tuned v1.1.0 model.

        The dataset is small, includes missing values, and does not establish
        current representativeness, policy context, or loan repayment results.
        Feature importance indicates predictive contribution within this
        dataset, not causation. Feature explanations and subgroup metrics are
        descriptive and not a legal fairness determination.
        """
    )
    st.caption(f"Labeled dataset: `{Path(paths.labeled_dataset).name}`")
    st.caption(
        f"Configured model: `{config['explainability']['model_name']}` "
        f"v{config['explainability']['model_version']}."
    )
    st.caption(
        "Business interpretation report: "
        f"`{Path(paths.business_insights_report).name}`"
    )
    st.link_button(
        "View Kaggle dataset source",
        config["dataset"]["source_url"],
    )
    render_disclaimer(
        "Do not use any prediction or explanation as the sole reason for a "
        "lending decision."
    )
