"""
Baseline classification models for Human Activity Recognition.
Implements Logistic Regression, Random Forest, and Dummy baselines.
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

    Parameters:
    -----------
    model_type : str
        'random_forest', 'rf':
            RandomForestClassifier with 100 trees, balanced class weights, parallelized.
        'logistic_regression', 'lr':
            StandardScaler + LogisticRegression pipeline with L2 regularization and balanced weights.
        'dummy':
            Most-frequent dummy classifier for absolute floor baseline.
    random_state : int
        Seed for reproducibility.
    class_weight : str or dict
        Handling class imbalance (default: 'balanced' to protect class 11 'jump_front_back').
    kwargs : Any
        Additional keyword arguments passed to the underlying estimator.

    Returns:
    --------
    BaseEstimator
        Scikit-learn compatible estimator ready for fit/predict.
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
