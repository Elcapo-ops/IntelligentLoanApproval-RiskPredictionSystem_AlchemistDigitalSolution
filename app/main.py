"""Streamlit application entry point and top-level page navigation."""
import logging
from config.settings import load_config
from utils.exceptions import LoanPredictionError
from utils.logging_config import configure_logging
from utils.reproducibility import set_random_seeds


def main() -> None:
    """Initialize configuration and render the application pages."""
    import streamlit as st

    try:
        config = load_config()
        logger = configure_logging(config)
        set_random_seeds(config["project"]["random_seed"])
    except (LoanPredictionError, OSError, ValueError, TypeError) as exc:
        st.set_page_config(page_title="Loan Approval Decision Support")
        logging.getLogger("loan_prediction").exception(
            "Unable to initialize the application"
        )
        st.error(f"Application setup failed: {exc}")
        return

    st.set_page_config(
        page_title=config["streamlit"]["page_title"],
        page_icon="🏦",
        layout=config["streamlit"]["layout"],
        initial_sidebar_state="expanded",
    )

    from app.components.theme import apply_theme, render_sidebar_brand
    from app.pages import (
        about,
        default_risk_predictor,
        dashboard,
        lending_decision_engine,
        loan_predictor,
        model_performance,
        risk_analysis,
    )

    apply_theme()
    pages = [
        st.Page(
            dashboard.render,
            title="Dashboard",
            icon=":material/dashboard:",
            url_path="dashboard",
            default=True,
        ),
        st.Page(
            loan_predictor.render,
            title="Loan Eligibility Predictor",
            icon=":material/assignment_turned_in:",
            url_path="loan-eligibility-predictor",
        ),
        st.Page(
            lending_decision_engine.render,
            title="Lending Decision Engine",
            icon=":material/rule:",
            url_path="lending-decision-engine",
        ),
        st.Page(
            default_risk_predictor.render,
            title="Loan Default Risk Prediction",
            icon=":material/health_and_safety:",
            url_path="loan-default-risk-prediction",
        ),
        st.Page(
            risk_analysis.render,
            title="Risk Analysis",
            icon=":material/monitoring:",
            url_path="risk-analysis",
        ),
        st.Page(
            model_performance.render,
            title="Model Performance",
            icon=":material/analytics:",
            url_path="model-performance",
        ),
        st.Page(
            about.render,
            title="About",
            icon=":material/info:",
            url_path="about",
        ),
    ]
    navigation = st.navigation(pages, position="sidebar", expanded=True)
    logger.info("Streamlit application initialized")
    render_sidebar_brand()
    navigation.run()

    


if __name__ == "__main__":
    main()
