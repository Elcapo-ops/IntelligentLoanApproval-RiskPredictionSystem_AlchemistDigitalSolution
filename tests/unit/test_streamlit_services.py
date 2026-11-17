from copy import deepcopy
import json

import pytest

from app.services.dashboard_service import (
    load_explainability_data,
    load_holdout_evaluation_data,
    load_historical_dashboard_data,
    load_model_performance_data,
)
from app.services.prediction_service import (
    build_historical_prediction_exports,
    explain_historical_approval_prediction,
    get_form_defaults,
    load_prediction_model,
    predict_historical_approval,
)
from config.settings import load_config
from data.splitting import get_model_feature_columns
from prediction.scope import HISTORICAL_LOAN_STATUS_DISCLOSURE
from utils.exceptions import DataValidationError


def test_dashboard_uses_observed_historical_records() -> None:
    data = load_historical_dashboard_data()

    assert data["application_count"] == 614
    assert data["approval_count"] == 422
    assert data["rejection_count"] == 192
    assert data["approval_rate"] == pytest.approx(422 / 614)
    assert data["rejection_rate"] == pytest.approx(192 / 614)
    assert int(data["approval_by_area"]["Y"].sum()) == 422
    assert int(data["approval_by_area"]["N"].sum()) == 192


def test_performance_service_returns_training_cv_only_metrics() -> None:
    data = load_model_performance_data()

    assert data["model_name"] == "random_forest"
    assert data["model_version"] == "1.1.0"
    assert data["training_rows"] == 491
    assert data["tuning"]["holdout_used"] is False
    assert len(data["candidate_cv"]) == 4
    assert data["tuning"]["tuned_balanced_accuracy_mean"] == pytest.approx(
        0.6927793669
    )
    assert {
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "roc_auc_mean",
    }.issubset(data["candidate_cv"].columns)
    assert data["cv_figure"].is_file()


def test_holdout_metrics_and_model_version_match_saved_evaluation_report() -> None:
    data = load_holdout_evaluation_data()
    confusion = data["confusion_matrix"]
    total = sum(confusion.values())

    assert data["model_name"] == "random_forest"
    assert data["model_version"] == "1.0.0"
    assert data["holdout_rows"] == total == 123
    assert data["report_date"]
    assert data["metrics"]["accuracy"] == pytest.approx(
        (confusion["true_negative"] + confusion["true_positive"]) / total,
        abs=0.00006,
    )
    assert data["metrics"]["precision"] == pytest.approx(
        confusion["true_positive"]
        / (confusion["true_positive"] + confusion["false_positive"]),
        abs=0.00006,
    )
    assert data["metrics"]["recall"] == pytest.approx(
        confusion["true_positive"]
        / (confusion["true_positive"] + confusion["false_negative"]),
        abs=0.00006,
    )
    assert data["metrics"]["f1"] == pytest.approx(
        2
        * data["metrics"]["precision"]
        * data["metrics"]["recall"]
        / (data["metrics"]["precision"] + data["metrics"]["recall"]),
        abs=0.00006,
    )
    assert all(
        figure is not None
        for figure in data["figures"].values()
    )


def test_explainability_service_loads_actual_saved_outputs() -> None:
    data = load_explainability_data()

    assert "Credit_History" in set(data["importance"]["feature"])
    assert {"Gender", "Education", "Married"}.issubset(
        set(data["groups"]["attribute"])
    )
    assert len(data["local"]) == 2
    assert data["figure"].is_file()
    assert data["importance_figure"].is_file()
    assert data["historical_application_count"] == 614
    assert {
        "Property_Area",
        "Credit_History",
        "Self_Employed",
    }.issubset(data["approval_patterns"])
    assert sum(
        int(group["applications"])
        for group in data["approval_patterns"]["Credit_History"].to_dict("records")
    ) == 614
    assert int(data["approval_patterns"]["Property_Area"]["approvals"].sum()) == 422
    assert set(data["importance"]["feature"]) == set(
        get_model_feature_columns(load_config())
    )
    assert set(data["groups"].groupby("attribute")["n"].sum()) == {491}
    assert data["quality_limitations"]
    assert data["fairness_interpretation"]
    assert "SHAP was not run" in data["shap_status"]
    assert data["shap_figures"] == []


def test_risk_analysis_gracefully_reports_missing_optional_artifacts(tmp_path) -> None:
    config = deepcopy(load_config())
    paths = config["paths"]
    paths["global_importance"] = str(tmp_path / "missing_importance.csv")
    paths["fairness_results"] = str(tmp_path / "missing_fairness.csv")
    paths["local_explanations"] = str(tmp_path / "missing_local.json")
    paths["quality_report"] = str(tmp_path / "missing_quality.md")
    paths["explainability_report"] = str(tmp_path / "missing_explainability.md")
    paths["figures_dir"] = str(tmp_path / "missing_figures")

    data = load_explainability_data(config)

    assert data["importance"] is None
    assert data["groups"] is None
    assert data["local"] == {}
    assert data["historical_application_count"] == 614
    assert data["approval_patterns"]
    assert len(data["warnings"]) >= 5


def test_registered_prediction_pipeline_scores_historical_approval() -> None:
    config = load_config()
    model, metadata = load_prediction_model(config)
    prediction = predict_historical_approval(
        model,
        get_form_defaults(config),
        config,
    )

    assert metadata["model_version"] == "1.1.0"
    assert prediction["predicted_loan_status"] in {"Y", "N"}
    assert "approval_probability" not in prediction
    assert prediction["target"] == "Loan_Status"
    assert prediction["prediction_type"] == "historical_approval_prediction"
    assert prediction["disclosure"] == HISTORICAL_LOAN_STATUS_DISCLOSURE


def test_prediction_exports_include_scope_and_no_applicant_inputs() -> None:
    config = load_config()
    model, metadata = load_prediction_model(config)
    prediction = predict_historical_approval(
        model,
        get_form_defaults(config),
        config,
    )

    exports = build_historical_prediction_exports(prediction, metadata)
    report = json.loads(exports["json"])

    assert report["predicted_loan_status"] == prediction["predicted_loan_status"]
    assert report["prediction_type"] == "historical_approval_prediction"
    assert report["disclosure"] == HISTORICAL_LOAN_STATUS_DISCLOSURE
    assert HISTORICAL_LOAN_STATUS_DISCLOSURE in exports["csv"]
    assert "ApplicantIncome" not in exports["json"]
    assert "ApplicantIncome" not in exports["csv"]


def test_prediction_uses_exact_schema_and_handles_optional_missing_values() -> None:
    config = load_config()
    model, _ = load_prediction_model(config)
    applicant = get_form_defaults(config)
    optional_values = {
        **applicant,
        "Gender": None,
        "LoanAmount": None,
        "Credit_History": None,
    }

    prediction = predict_historical_approval(model, optional_values, config)

    assert len(applicant) == 11
    assert set(applicant) == set(get_model_feature_columns(config))
    assert prediction["predicted_loan_status"] in {"Y", "N"}


def test_prediction_rejects_missing_extra_and_invalid_inputs() -> None:
    config = load_config()
    model, _ = load_prediction_model(config)
    valid = get_form_defaults(config)

    with pytest.raises(DataValidationError, match="schema mismatch"):
        predict_historical_approval(
            model,
            {key: value for key, value in valid.items() if key != "Gender"},
        )
    with pytest.raises(DataValidationError, match="extra"):
        predict_historical_approval(model, {**valid, "Loan_ID": "not-a-feature"})
    with pytest.raises(DataValidationError, match="cannot be negative"):
        predict_historical_approval(model, {**valid, "ApplicantIncome": -1})
    with pytest.raises(DataValidationError, match="greater than zero"):
        predict_historical_approval(model, {**valid, "LoanAmount": 0})
    with pytest.raises(DataValidationError, match="outside configured categories"):
        predict_historical_approval(model, {**valid, "Property_Area": "Unknown"})
    with pytest.raises(DataValidationError, match="must be finite"):
        predict_historical_approval(model, {**valid, "ApplicantIncome": float("inf")})


def test_prediction_does_not_read_uncalibrated_probability_scores() -> None:
    class LabelOnlyModel:
        classes_ = ["N", "Y"]

        def predict(self, _frame):
            assert len(_frame) == 1
            return ["Y"]

        def predict_proba(self, _frame):
            del _frame
            pytest.fail("Unvalidated probabilities must not be used")

    config = load_config()
    prediction = predict_historical_approval(
        LabelOnlyModel(),
        get_form_defaults(config),
        config,
    )

    assert prediction["predicted_loan_status"] == "Y"
    assert "approval_probability" not in prediction


def test_local_prediction_explanation_uses_training_only_class_sensitivity() -> None:
    config = load_config()
    model, _ = load_prediction_model(config)
    applicant = get_form_defaults(config)

    explanation = explain_historical_approval_prediction(model, applicant, config)

    assert explanation["predicted_loan_status"] in {"Y", "N"}
    assert explanation["reference_rows"] == 491
    assert len(explanation["features"]) == len(get_model_feature_columns(config))
    assert all(
        item["predicted_loan_status_after_replacement"] in {"Y", "N"}
        and isinstance(item["changed"], bool)
        for item in explanation["features"]
    )
    assert "do not establish causation" in explanation["warning"]


def _render_page_by_module(page_module: str) -> None:
    import importlib

    importlib.import_module(page_module).render()


@pytest.mark.parametrize(
    ("page_module", "expected_title"),
    [
        ("app.pages.dashboard", "Loan Approval Dashboard"),
        ("app.pages.loan_predictor", "Loan Eligibility Predictor"),
        ("app.pages.risk_analysis", "Risk Analysis"),
        ("app.pages.model_performance", "Model Performance"),
        ("app.pages.about", "About this prototype"),
    ],
)
def test_each_streamlit_page_renders_without_errors(
    page_module: str,
    expected_title: str,
) -> None:
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_function(
        _render_page_by_module,
        args=(page_module,),
    ).run(timeout=30)

    assert not app_test.exception
    assert expected_title in [title.value for title in app_test.title]
    assert any(
        item.value == HISTORICAL_LOAN_STATUS_DISCLOSURE
        for item in app_test.warning
    )


def test_model_performance_page_displays_saved_holdout_and_cv_outputs() -> None:
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_function(
        _render_page_by_module,
        args=("app.pages.model_performance",),
    ).run(timeout=30)
    metric_values = {item.label: item.value for item in app_test.metric}
    displayed_text = "\n".join(item.value for item in app_test.caption)

    assert not app_test.exception
    assert metric_values["Holdout accuracy"] == "0.8618"
    assert metric_values["Holdout precision (Y)"] == "0.9146"
    assert metric_values["Holdout recall (Y)"] == "0.8824"
    assert metric_values["Holdout F1 (Y)"] == "0.8982"
    assert metric_values["Holdout ROC-AUC"] == "0.8672"
    assert "random_forest v1.0.0" in " ".join(
        item.value for item in app_test.subheader
    )
    assert load_holdout_evaluation_data()["report_date"] in displayed_text


def test_model_performance_page_keeps_cv_when_holdout_report_is_missing(
    monkeypatch,
) -> None:
    from streamlit.testing.v1 import AppTest

    import app.pages.model_performance as performance_page

    def missing_evaluation_report():
        raise DataValidationError("evaluation report not found")

    monkeypatch.setattr(
        performance_page,
        "load_holdout_evaluation_data",
        missing_evaluation_report,
    )
    app_test = AppTest.from_function(
        _render_page_by_module,
        args=("app.pages.model_performance",),
    ).run(timeout=30)

    assert not app_test.exception
    assert any("evaluation report not found" in item.value for item in app_test.warning)
    assert any(
        item.label == "Tuned CV balanced accuracy"
        for item in app_test.metric
    )


def test_risk_page_displays_actual_importance_fairness_and_limitations() -> None:
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_function(
        _render_page_by_module,
        args=("app.pages.risk_analysis",),
    ).run(timeout=30)
    headings = [item.value for item in app_test.subheader]

    assert not app_test.exception
    assert "Historical Approval Patterns" in headings
    assert "Global predictive feature importance" in headings
    assert "Exploratory fairness analysis" in headings
    assert "Dataset limitations" in headings
    assert "SHAP" in headings
    assert any("SHAP was not run" in item.value for item in app_test.markdown)


def test_predictor_form_exposes_all_supported_schema_fields() -> None:
    from streamlit.testing.v1 import AppTest

    config = load_config()
    feature_columns = get_model_feature_columns(config)
    app_test = AppTest.from_function(
        _render_page_by_module,
        args=("app.pages.loan_predictor",),
    ).run(timeout=30)

    rendered_fields = {
        item.label for item in [*app_test.number_input, *app_test.selectbox]
    }

    assert rendered_fields == {column.replace("_", " ") for column in feature_columns}
    assert sum(item.label == "Not provided" for item in app_test.checkbox) == len(
        feature_columns
    )


def test_streamlit_entry_point_renders_dashboard_and_historical_metrics() -> None:
    from streamlit.testing.v1 import AppTest

    from config.settings import PROJECT_ROOT

    app_test = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)

    assert not app_test.exception
    assert "Loan Approval Dashboard" in [title.value for title in app_test.title]
    metrics = {metric.label: metric.value for metric in app_test.metric}
    assert metrics["Applications"] == "614"
    assert metrics["Approval rate"] == "68.7%"
    assert metrics["Predictions run"] == "0"
    assert any(
        item.value == HISTORICAL_LOAN_STATUS_DISCLOSURE
        for item in app_test.warning
    )


def test_streamlit_navigation_registers_all_configured_page_routes() -> None:
    from streamlit.testing.v1 import AppTest

    from config.settings import PROJECT_ROOT

    app_test = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
    routes = {
        page.get("url_pathname")
        for page in app_test._registered_pages.values()
    }

    assert not app_test.exception
    assert {
        "",
        "loan-eligibility-predictor",
        "risk-analysis",
        "model-performance",
        "about",
    }.issubset(routes)


def test_predictor_form_runs_pipeline_and_updates_live_session_counts() -> None:
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_function(
        _render_page_by_module,
        args=("app.pages.loan_predictor",),
    ).run(timeout=30)
    submit = next(
        button
        for button in app_test.button
        if button.label == "Estimate Historical Approval Status"
    )

    app_test = submit.click().run(timeout=30)

    assert not app_test.exception
    assert any(
        "Estimated Historical Approval Status" in item.value
        for item in app_test.subheader
    )
    assert any(
        item.label == "Estimated Historical Approval Status"
        for item in app_test.metric
    )
    assert next(
        item.value for item in app_test.metric if item.label == "Model version"
    ) == "v1.1.0"
    assert app_test.session_state["live_prediction_count"] == 1
    assert (
        app_test.session_state.get("live_prediction_approved", 0)
        + app_test.session_state.get("live_prediction_rejected", 0)
        == 1
    )
    assert sum(
        item.value == HISTORICAL_LOAN_STATUS_DISCLOSURE
        for item in app_test.warning
    ) >= 2
    assert {
        item.label for item in app_test.get("download_button")
    } == {
        "Download prediction summary (JSON)",
        "Download prediction summary (CSV)",
    }
    assert any(
        "probability calibration has not been validated" in item.value
        for item in app_test.info
    )
    assert any(
        "One-feature sensitivity checks" in item.value
        for item in app_test.subheader
    )


def test_predictor_form_accepts_all_optional_values_as_missing() -> None:
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_function(
        _render_page_by_module,
        args=("app.pages.loan_predictor",),
    ).run(timeout=30)
    assert len(app_test.checkbox) == 11
    for checkbox in app_test.checkbox:
        checkbox.set_value(True)
    app_test.run(timeout=30)
    submit = next(
        button
        for button in app_test.button
        if button.label == "Estimate Historical Approval Status"
    )

    app_test = submit.click().run(timeout=30)

    assert not app_test.exception
    assert any(
        "Estimated Historical Approval Status" in item.value
        for item in app_test.subheader
    )
    assert app_test.session_state["live_prediction_count"] == 1


def test_predictor_surfaces_registered_model_loading_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest

    import app.pages.loan_predictor as predictor_page
    from utils.exceptions import ArtifactError

    def fail_model_load():
        raise ArtifactError("registered model integrity check failed")

    monkeypatch.setattr(predictor_page, "_cached_model", fail_model_load)
    app_test = AppTest.from_function(
        _render_page_by_module,
        args=("app.pages.loan_predictor",),
    ).run(timeout=30)
    submit = next(
        button
        for button in app_test.button
        if button.label == "Estimate Historical Approval Status"
    )

    app_test = submit.click().run(timeout=30)

    assert not app_test.exception
    assert any(
        "Prediction could not be completed" in item.value
        and "integrity check failed" in item.value
        for item in app_test.error
    )
    assert app_test.session_state.get("live_prediction_count", 0) == 0
