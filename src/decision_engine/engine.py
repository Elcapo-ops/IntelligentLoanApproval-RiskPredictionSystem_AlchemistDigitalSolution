"""Core lending decision engine.

Evaluates loan applications against a configurable lending policy and produces
one of three outcomes: APPROVE, REJECT, or REFER FOR MANUAL REVIEW.

The engine is deterministic and explainable — every rule evaluated produces a
recorded result with inputs, thresholds, pass/fail status, and contribution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from decision_engine.affordability import evaluate_affordability
from decision_engine.eligibility import evaluate_eligibility
from decision_engine.risk_integration import DefaultRiskAdapter
from decision_engine.rules import (
    evaluate_mandatory_manual_review,
    evaluate_mandatory_rejection,
)
from decision_engine.schemas import (
    AffordabilityCalculation,
    ApplicationInput,
    DecisionReport,
    DecisionOutcome,
    FinancialAssessment,
    RiskAssessment,
    RuleResult,
)
from decision_engine.policy_loader import load_lending_policy
from decision_engine.policy_validator import validate_lending_policy


class DecisionEngine:
    """Rule-based lending decision engine."""

    def __init__(self, project_root: str | None = None, policy: dict | None = None) -> None:
        from pathlib import Path

        self.project_root = Path(project_root).resolve() if project_root else Path.cwd()
        self.policy = policy or load_lending_policy()
        validate_lending_policy(self.policy)
        self.policy_version = self.policy.get("policy_version", "unknown")
        self._risk_adapter: DefaultRiskAdapter | None = None
        self._risk_loaded: bool = False

    @property
    def risk_adapter(self) -> DefaultRiskAdapter:
        """Lazily initialize the default-risk model adapter."""
        if not self._risk_loaded:
            self._risk_adapter = DefaultRiskAdapter(self.project_root)
            self._risk_loaded = True
        return self._risk_adapter  # type: ignore[return-value]

    def _build_application(self, raw: dict[str, Any]) -> ApplicationInput:
        """Build and validate an ApplicationInput from raw input data."""
        missing_required: list[str] = []
        required_fields = self.policy.get("eligibility", {}).get("required_application_fields", [])

        for field in required_fields:
            if raw.get(field) is None:
                missing_required.append(field)

        return ApplicationInput(
            application_id=str(raw.get("application_id", "unknown")),
            applicant_age=raw.get("applicant_age"),
            annual_income=raw.get("annual_income"),
            monthly_income=raw.get("monthly_income"),
            monthly_debt_obligations=raw.get("monthly_debt_obligations"),
            requested_loan_amount=raw.get("requested_loan_amount"),
            loan_term_months=raw.get("loan_term_months"),
            interest_rate=raw.get("interest_rate"),
            employment_length_months=raw.get("employment_length_months"),
            home_ownership=raw.get("home_ownership"),
            verification_status=raw.get("verification_status"),
            loan_purpose=raw.get("loan_purpose"),
            residency_state=raw.get("residency_state"),
            default_probability=raw.get("default_probability"),
            risk_category=raw.get("risk_category"),
            model_version=raw.get("model_version"),
            model_used=raw.get("model_used", False),
            missing_required_fields=missing_required,
        )

    def _calculate_financials(
        self,
        application: ApplicationInput,
    ) -> FinancialAssessment:
        """Calculate DTI, loan-to-income, installment, and affordability."""
        from decision_engine.financial_calculations import (
            calculate_affordability,
            calculate_dti,
            calculate_loan_to_income,
            calculate_monthly_installment,
        )

        # Calculate installment
        installment = None
        if (
            application.requested_loan_amount
            and application.loan_term_months
            and application.interest_rate is not None
        ):
            try:
                installment_calc = calculate_monthly_installment(
                    principal=application.requested_loan_amount,
                    annual_interest_rate=application.interest_rate,
                    term_months=application.loan_term_months,
                )
                installment = installment_calc.monthly_payment
            except (ValueError, TypeError):
                installment = None

        # DTI
        dti_calc = calculate_dti(
            total_monthly_debt=application.monthly_debt_obligations,
            gross_monthly_income=application.monthly_income,
        )

        # Loan-to-income
        annual_income = application.annual_income
        if annual_income is None and application.monthly_income is not None:
            annual_income = application.monthly_income * 12
        lti_calc = calculate_loan_to_income(
            loan_amount=application.requested_loan_amount,
            annual_income=annual_income,
        )

        # Affordability
        afford_calc = calculate_affordability(
            gross_monthly_income=application.monthly_income,
            monthly_debt_obligations=application.monthly_debt_obligations,
            proposed_monthly_payment=installment,
            dti_percent=dti_calc.dti_percent,
        )

        return FinancialAssessment(
            monthly_income=application.monthly_income,
            monthly_debt_obligations=application.monthly_debt_obligations,
            proposed_monthly_payment=installment,
            dti_percent=dti_calc.dti_percent,
            loan_to_income=lti_calc.ratio,
            payment_to_income_percent=afford_calc.payment_to_income_percent,
            disposable_income=afford_calc.disposable_income,
            affordability_pass=afford_calc.available,
        )

    def _evaluate_risk(self, application: ApplicationInput) -> RiskAssessment:
        """Evaluate default-risk model if integration is configured."""
        from decision_engine.risk_integration import DefaultRiskAdapter

        adapter = DefaultRiskAdapter(self.project_root)
        return adapter.predict(application)

    def _determine_decision(
        self,
        rejection_results: list[RuleResult],
        review_results: list[RuleResult],
        eligibility_results: list[RuleResult],
        financial_results: list[RuleResult],
        risk_results: list[RuleResult],
        warning_results: list[RuleResult],
    ) -> tuple[str, list[str], list[str], list[str]]:
        """Determine the final decision using documented precedence.

        Precedence:
        1. Mandatory rejection — any fail = REJECT
        2. Mandatory manual review — any fail = REFER FOR MANUAL REVIEW
        3. Eligibility and financial — any fail = REJECT
        4. Credit risk — any fail = REJECT (or refer based on policy)
        5. Warnings — recorded but don't change decision
        6. APPROVE only when all required conditions pass
        """
        reasons: list[str] = []
        warnings: list[str] = []
        manual_review_reasons: list[str] = []

        # Step 1: Mandatory rejection
        for rule in rejection_results:
            if rule.result == "fail":
                reasons.append(f"{rule.rule_name}: {rule.explanation}")
                return "REJECT", reasons, warnings, manual_review_reasons

        # Step 2: Mandatory manual review
        for rule in review_results:
            if rule.result == "fail":
                manual_review_reasons.append(f"{rule.rule_name}: {rule.explanation}")
                return "REFER FOR MANUAL REVIEW", reasons, warnings, manual_review_reasons

        # Step 3: Eligibility failures
        for rule in eligibility_results:
            if rule.result == "fail":
                reasons.append(f"{rule.rule_name}: {rule.explanation}")
                return "REJECT", reasons, warnings, manual_review_reasons

        # Step 4: Financial failures
        for rule in financial_results:
            if rule.result == "fail":
                reasons.append(f"{rule.rule_name}: {rule.explanation}")
                return "REJECT", reasons, warnings, manual_review_reasons

        # Step 5: Credit risk failures
        credit_risk_cfg = self.policy.get("credit_risk", {})
        max_prob = credit_risk_cfg.get("maximum_default_probability")
        restricted_categories = credit_risk_cfg.get("risk_categories_restricted", [])

        for rule in risk_results:
            if rule.result == "fail":
                reasons.append(f"{rule.rule_name}: {rule.explanation}")
                return "REJECT", reasons, warnings, manual_review_reasons

        # Step 6: Warnings
        for rule in warning_results:
            if rule.result == "fail":
                warnings.append(f"{rule.rule_name}: {rule.explanation}")

        # Step 7: Approve if everything passes
        return "APPROVE", reasons, warnings, manual_review_reasons

    def evaluate(self, raw_application: dict[str, Any]) -> DecisionReport:
        """Evaluate a loan application and return a complete decision report.

        Args:
            raw_application: Dictionary of application data.

        Returns:
            DecisionReport with full rule evaluation, financial assessment,
            risk assessment, and final decision.
        """
        application = self._build_application(raw_application)
        timestamp = datetime.now(timezone.utc).isoformat()

        # Evaluate rules in precedence order
        rejection_results = evaluate_mandatory_rejection(application, self.policy)
        review_results = evaluate_mandatory_manual_review(
            application, self.policy
        )
        eligibility_results = evaluate_eligibility(application, self.policy)
        financial_results = evaluate_affordability(application, self.policy)

        # Risk assessment (optional)
        risk_assessment = self._evaluate_risk(application)

        # Risk rules
        risk_results: list[RuleResult] = []
        credit_risk_cfg = self.policy.get("credit_risk", {})
        max_prob = credit_risk_cfg.get("maximum_default_probability")
        restricted_categories = credit_risk_cfg.get("risk_categories_restricted", [])

        if risk_assessment.model_used and risk_assessment.estimated_default_probability is not None:
            prob = risk_assessment.estimated_default_probability
            if max_prob is not None and prob > max_prob:
                risk_results.append(
                    RuleResult(
                        rule_id="RISK-001",
                        rule_name="Maximum default probability",
                        category="risk",
                        input_value=prob,
                        threshold=max_prob,
                        result="fail",
                        explanation=(
                            f"Estimated default probability {prob:.1%} exceeds "
                            f"maximum {max_prob:.1%}."
                        ),
                        contribution="reject",
                    )
                )
            else:
                risk_results.append(
                    RuleResult(
                        rule_id="RISK-001",
                        rule_name="Maximum default probability",
                        category="risk",
                        input_value=prob,
                        threshold=max_prob,
                        result="pass",
                        explanation=(
                            f"Estimated default probability {prob:.1%} is within "
                            f"maximum {max_prob:.1%}."
                        ),
                        contribution="none",
                    )
                )

            if risk_assessment.risk_category and risk_assessment.risk_category in restricted_categories:
                risk_results.append(
                    RuleResult(
                        rule_id="RISK-002",
                        rule_name="Restricted risk category",
                        category="risk",
                        input_value=risk_assessment.risk_category,
                        threshold=restricted_categories,
                        result="fail",
                        explanation=(
                            f"Risk category '{risk_assessment.risk_category}' is "
                            "restricted by policy."
                        ),
                        contribution="reject",
                    )
                )
            elif risk_assessment.risk_category:
                risk_results.append(
                    RuleResult(
                        rule_id="RISK-002",
                        rule_name="Restricted risk category",
                        category="risk",
                        input_value=risk_assessment.risk_category,
                        threshold=restricted_categories,
                        result="pass",
                        explanation=(
                            f"Risk category '{risk_assessment.risk_category}' is not restricted."
                        ),
                        contribution="none",
                    )
                )
        elif not risk_assessment.model_used:
            # Model unavailable — recorded as not_applicable, not a failure
            risk_results.append(
                RuleResult(
                    rule_id="RISK-000",
                    rule_name="Default-risk model availability",
                    category="risk",
                    input_value=None,
                    threshold="model available",
                    result="not_applicable",
                    explanation=(
                        f"Default-risk model not used: {risk_assessment.error or 'not available'}."
                    ),
                    contribution="none",
                )
            )

        # Determine decision
        decision, reasons, warnings, manual_review_reasons = self._determine_decision(
            rejection_results,
            review_results,
            eligibility_results,
            financial_results,
            risk_results,
            [],
        )

        # Build financial assessment
        financial_assessment = self._calculate_financials(application)

        # Build report
        all_rules = (
            rejection_results
            + review_results
            + eligibility_results
            + financial_results
            + risk_results
        )
        passed = [r.rule_id for r in all_rules if r.result == "pass"]
        failed = [r.rule_id for r in all_rules if r.result == "fail"]

        # Generate explanation
        explanation = self._generate_explanation(
            decision, application, financial_assessment, risk_assessment,
            all_rules, reasons, warnings, manual_review_reasons,
        )

        return DecisionReport(
            application_id=application.application_id,
            decision_timestamp=timestamp,
            policy_version=self.policy_version,
            decision=decision,
            application=application,
            financial_assessment=financial_assessment,
            risk_assessment=risk_assessment,
            rules=all_rules,
            passed_rules=passed,
            failed_rules=failed,
            warnings=warnings,
            manual_review_reasons=manual_review_reasons,
            decision_explanation=explanation,
            model_used=risk_assessment.model_used,
        )

    def evaluate_outcome(self, raw_application: dict[str, Any]) -> DecisionOutcome:
        """Evaluate and return a simplified DecisionOutcome."""
        report = self.evaluate(raw_application)
        return DecisionOutcome(
            decision=report.decision,
            reasons=report.manual_review_reasons or [
                r.explanation for r in report.rules if r.result == "fail"
            ],
            warnings=report.warnings,
            manual_review_reasons=report.manual_review_reasons,
        )

    def _generate_explanation(
        self,
        decision: str,
        application: ApplicationInput,
        financial: FinancialAssessment,
        risk: RiskAssessment,
        rules: list[RuleResult],
        reasons: list[str],
        warnings: list[str],
        manual_review_reasons: list[str],
    ) -> str:
        """Generate a human-readable explanation of the decision."""
        lines: list[str] = []
        lines.append(f"Decision: {decision}")
        lines.append(f"Policy version: {self.policy_version}")
        lines.append(f"Application: {application.application_id}")
        lines.append("")

        if decision == "REJECT":
            lines.append("This application was rejected because:")
            for reason in reasons:
                lines.append(f"  - {reason}")
        elif decision == "REFER FOR MANUAL REVIEW":
            lines.append("This application requires manual review because:")
            for reason in manual_review_reasons:
                lines.append(f"  - {reason}")
        else:
            lines.append("This application meets all policy requirements and is approved.")

        lines.append("")
        lines.append("Financial summary:")
        if financial.monthly_income is not None:
            lines.append(f"  Monthly income: {financial.monthly_income:,.2f}")
        if financial.monthly_debt_obligations is not None:
            lines.append(f"  Monthly debt obligations: {financial.monthly_debt_obligations:,.2f}")
        if financial.proposed_monthly_payment is not None:
            lines.append(f"  Proposed monthly payment: {financial.proposed_monthly_payment:,.2f}")
        if financial.dti_percent is not None:
            lines.append(f"  Debt-to-income: {financial.dti_percent}%")
        if financial.payment_to_income_percent is not None:
            lines.append(f"  Payment-to-income: {financial.payment_to_income_percent}%")
        if financial.disposable_income is not None:
            lines.append(f"  Disposable income: {financial.disposable_income:,.2f}")

        lines.append("")
        if risk.model_used:
            lines.append("Risk assessment:")
            if risk.estimated_default_probability is not None:
                lines.append(f"  Estimated default probability: {risk.estimated_default_probability:.1%}")
            if risk.risk_category:
                lines.append(f"  Risk category: {risk.risk_category}")
            lines.append(f"  Model: {risk.model_name or 'unknown'} v{risk.model_version or 'unknown'}")
        else:
            lines.append("Risk assessment: Default-risk model was not used.")
            if risk.error:
                lines.append(f"  Reason: {risk.error}")

        if warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in warnings:
                lines.append(f"  - {warning}")

        lines.append("")
        lines.append(
            "Disclaimer: This is a decision-support prototype, not an autonomous "
            "lending decision. All decisions require human review and approval "
            "by authorized personnel."
        )

        return "\n".join(lines)
