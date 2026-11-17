"""Reproducible, class-imbalance-aware classifier factories."""

from typing import Any

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from config.settings import load_config, validate_config
from utils.exceptions import ConfigurationError


class BalancedGradientBoostingClassifier(ClassifierMixin, BaseEstimator):
    """Gradient boosting classifier with fold-local balanced sample weights."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 2,
        subsample: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "BalancedGradientBoostingClassifier":
        """Fit with weights computed solely from this fit call's target rows."""
        self.estimator_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        self.estimator_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        return self

    def predict(self, X: Any) -> Any:
        """Predict labels with the fitted gradient boosting model."""
        return self.estimator_.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Return fitted gradient boosting class probabilities."""
        return self.estimator_.predict_proba(X)


def build_classifiers(
    config: dict[str, Any] | None = None,
) -> dict[str, BaseEstimator]:
    """Construct configured classifier candidates with a shared seed."""
    settings = validate_config(config if config is not None else load_config())
    seed = settings["project"]["random_seed"]
    parameters = settings["models"]["parameters"]
    candidates = settings["models"]["candidates"]
    factories: dict[str, BaseEstimator] = {
        "logistic_regression": LogisticRegression(
            max_iter=parameters["logistic_regression"]["max_iter"],
            solver=parameters["logistic_regression"]["solver"],
            class_weight=parameters["logistic_regression"]["class_weight"],
            random_state=seed,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=parameters["decision_tree"]["max_depth"],
            min_samples_leaf=parameters["decision_tree"]["min_samples_leaf"],
            class_weight=parameters["decision_tree"]["class_weight"],
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=parameters["random_forest"]["n_estimators"],
            max_features=parameters["random_forest"]["max_features"],
            min_samples_leaf=parameters["random_forest"]["min_samples_leaf"],
            class_weight=parameters["random_forest"]["class_weight"],
            n_jobs=parameters["random_forest"]["n_jobs"],
            random_state=seed,
        ),
        "gradient_boosting": BalancedGradientBoostingClassifier(
            n_estimators=parameters["gradient_boosting"]["n_estimators"],
            learning_rate=parameters["gradient_boosting"]["learning_rate"],
            max_depth=parameters["gradient_boosting"]["max_depth"],
            subsample=parameters["gradient_boosting"]["subsample"],
            random_state=seed,
        ),
    }
    unknown_candidates = sorted(set(candidates).difference(factories))
    if unknown_candidates:
        raise ConfigurationError(
            f"Unsupported configured model candidates: {unknown_candidates}"
        )
    return {name: factories[name] for name in candidates}
