"""
Baseline classification models for Human Activity Recognition.

Implements standard benchmark classifiers:
- Random Forest: Non-linear decision tree ensemble with parallelization.
- Logistic Regression: Linear baseline wrapped in a StandardScaler pipeline with L2 regularization.
- Dummy Classifier: Most-frequent baseline to establish the absolute empirical floor.
"""

from typing import Any, Dict, Optional, Union
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_baseline_model(
    model_type: str = "random_forest",
    random_state: int = 42,
    class_weight: Optional[Union[str, Dict]] = "balanced",
    **kwargs: Any,
) -> BaseEstimator:
    """
    Factory function to instantiate baseline classifiers with standardized defaults.

    Provides three foundational models:
    1. 'random_forest' (or 'rf'):
       - Uses `sklearn.ensemble.RandomForestClassifier`.
       - Defaults: 100 estimators, all CPU cores (`n_jobs=-1`), `class_weight='balanced'`.
       - High non-linear expressive power, handles feature interactions, and measures feature importance.
    2. 'logistic_regression' (or 'lr'):
       - Uses a `sklearn.pipeline.Pipeline` with `StandardScaler` + `LogisticRegression`.
       - Defaults: L-BFGS solver, L2 penalty, max 1,000 iterations, `class_weight='balanced'`.
       - Fast, linear baseline model serving as a benchmark against complex tree and neural methods.
    3. 'dummy':
       - Uses `sklearn.dummy.DummyClassifier`.
       - Strategy: Defaults to 'most_frequent'. Provides the theoretical lower performance bound.

    Parameters
    ----------
    model_type : str, default="random_forest"
        'random_forest', 'rf', 'logistic_regression', 'lr', or 'dummy'.
    random_state : int, default=42
        Random seed for model reproducibility across folds.
    class_weight : Optional[Union[str, Dict]], default="balanced"
        Weighting strategy for class imbalance. 'balanced' automatically adjusts
        weights inversely proportional to class frequencies, crucial for rare classes
        like class 11 ('jump_front_back').
    **kwargs : Any
        Additional hyperparameter arguments passed directly to the underlying estimator.

    Returns
    -------
    BaseEstimator
        Scikit-learn estimator or Pipeline object ready for `fit()` and `predict()`.

    Raises
    ------
    ValueError
        If an unrecognized `model_type` string is provided.

    Examples
    --------
    >>> rf_model = get_baseline_model("random_forest", n_estimators=200)
    >>> lr_model = get_baseline_model("logistic_regression", C=0.1)
    """
    m_type = model_type.lower().strip()

    if m_type in ("random_forest", "rf"):
        rf_params = {
            "n_estimators": 100,
            "max_depth": kwargs.pop("max_depth", None),
            "min_samples_split": kwargs.pop("min_samples_split", 2),
            "min_samples_leaf": kwargs.pop("min_samples_leaf", 1),
            "class_weight": class_weight,
            "random_state": random_state,
            "n_jobs": kwargs.pop("n_jobs", -1),
        }
        rf_params.update(kwargs)
        return RandomForestClassifier(**rf_params)

    elif m_type in ("logistic_regression", "lr"):
        lr_params = {
            "C": kwargs.pop("C", 1.0),
            "max_iter": kwargs.pop("max_iter", 1000),
            "class_weight": class_weight,
            "random_state": random_state,
            "solver": kwargs.pop("solver", "lbfgs"),
        }
        lr_params.update(kwargs)
        # Scaler is required before Logistic Regression to normalize diverse feature magnitudes
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(**lr_params)),
        ])

    elif m_type == "dummy":
        strategy = kwargs.pop("strategy", "most_frequent")
        return DummyClassifier(strategy=strategy, random_state=random_state, **kwargs)

    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. Choose 'random_forest', 'logistic_regression', or 'dummy'."
        )
