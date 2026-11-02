"""
Training and Cross-Validation pipelines for Human Activity Recognition.
Includes Stratified K-Fold CV, Out-Of-Fold (OOF) tracking, and Kaggle submission generators.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import StratifiedKFold

from iml_assignment.config import (
    ACTIVITY_CLASSES,
    ID2LABEL,
    LABEL2ID,
    SUBMISSIONS_DIR,
)
from iml_assignment.evaluate import compute_classification_metrics
from iml_assignment.models.baseline import get_baseline_model


def cross_validate_model(
    model: Union[str, BaseEstimator],
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray, List],
    n_splits: int = 5,
    random_state: int = 42,
    test_X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Execute a leak-free Stratified K-Fold Cross-Validation pipeline.

    Features:
    ---------
    1. Stratified Splits: Preserves identical 12-activity class proportions in every fold,
       guaranteeing balanced exposure for the rare class 11 ('jump_front_back').
    2. Out-Of-Fold (OOF) Tracking: Collects unbiased OOF predictions and probability distributions
       for the full dataset, which can also be used as meta-features for ensembling.
    3. Fold Averaging for Test: If test_X is provided, predictions from all 5 fold models
       are averaged to maximize test generalization.

    Parameters:
    -----------
    model : str or BaseEstimator
        Model identifier ('random_forest', 'logistic_regression', 'dummy') or an estimator instance.
    X : DataFrame or ndarray
        Feature matrix (N, n_features).
    y : Series, ndarray, or List
        Target labels.
    n_splits : int
        Number of cross-validation folds (default 5).
    random_state : int
        Seed for stratified fold splitting.
    test_X : Optional[DataFrame or ndarray]
        Test set features to generate averaged fold predictions for Kaggle.
    model_kwargs : Optional[dict]
        Additional arguments passed to get_baseline_model if model is a string.
    verbose : bool
        Whether to print fold-by-fold results.

    Returns:
    --------
    dict with:
      - 'model_name': str
      - 'fold_summary': pd.DataFrame with per-fold metrics and Mean +/- Std
      - 'oof_preds': np.ndarray of predicted labels across all samples
      - 'oof_probs': np.ndarray of predicted probabilities (N, 12)
      - 'test_preds': np.ndarray of final test predictions (if test_X provided)
      - 'test_probs': np.ndarray of final test probabilities
      - 'overall_metrics': Dict of OOF accuracy, macro_f1, per-class metrics
      - 'models': List of fitted estimators for each fold
    """
    model_kwargs = model_kwargs or {}

    # Identify model name and base estimator
    if isinstance(model, str):
        model_name = model.replace("_", " ").title()
        base_estimator = get_baseline_model(model_type=model, random_state=random_state, **model_kwargs)
    elif isinstance(model, BaseEstimator):
        model_name = model.__class__.__name__
        base_estimator = model
    else:
        raise TypeError(f"Expected model string or BaseEstimator, got {type(model)}")

    # Format arrays
    X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_vec = y.to_numpy() if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)

    n_samples = len(X_mat)
    classes = ACTIVITY_CLASSES
    n_classes = len(classes)

    # Initialize OOF storage
    oof_preds = np.empty(n_samples, dtype=object)
    oof_probs = np.zeros((n_samples, n_classes), dtype=float)

    # Initialize test accumulator
    test_mat = None
    accum_test_probs = None
    if test_X is not None:
        test_mat = test_X.to_numpy() if isinstance(test_X, pd.DataFrame) else np.asarray(test_X)
        accum_test_probs = np.zeros((len(test_mat), n_classes), dtype=float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_records = []
    fitted_models = []

    if verbose:
        print(f"\n=======================================================")
        print(f"  Starting {n_splits}-Fold Stratified CV: {model_name}")
        print(f"  Samples: {n_samples} | Features: {X_mat.shape[1]}")
        print(f"=======================================================")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_mat, y_vec), start=1):
        X_tr, y_tr = X_mat[train_idx], y_vec[train_idx]
        X_va, y_va = X_mat[val_idx], y_vec[val_idx]

        # Clone estimator for this fold
        fold_model = clone(base_estimator)
        fold_model.fit(X_tr, y_tr)
        fitted_models.append(fold_model)

        # Train and Validation predictions
        tr_preds = fold_model.predict(X_tr)
        va_preds = fold_model.predict(X_va)

        oof_preds[val_idx] = va_preds

        # Predict probabilities if supported
        if hasattr(fold_model, "predict_proba"):
            va_probs = fold_model.predict_proba(X_va)
            # Align columns if classes order differs
            model_classes = list(fold_model.classes_)
            for col_idx, cls_name in enumerate(model_classes):
                if cls_name in classes:
                    c_id = classes.index(cls_name)
                    oof_probs[val_idx, c_id] = va_probs[:, col_idx]

            # Accumulate test predictions
            if test_mat is not None:
                te_probs = fold_model.predict_proba(test_mat)
                for col_idx, cls_name in enumerate(model_classes):
                    if cls_name in classes:
                        c_id = classes.index(cls_name)
                        accum_test_probs[:, c_id] += te_probs[:, col_idx] / n_splits

        # Metrics for this fold
        tr_acc = np.mean(tr_preds == y_tr)
        va_acc = np.mean(va_preds == y_va)
        va_metrics = compute_classification_metrics(y_va, va_preds, class_names=classes)

        fold_records.append({
            "Fold": fold_idx,
            "Train Accuracy": round(tr_acc * 100, 2),
            "Val Accuracy": round(va_acc * 100, 2),
            "Val Macro F1": round(va_metrics["macro_f1"], 4),
            "Val Weighted F1": round(va_metrics["weighted_f1"], 4),
        })

        if verbose:
            print(f"  Fold {fold_idx}: Val Acc = {va_acc*100:.2f}% | Val Macro F1 = {va_metrics['macro_f1']:.4f} (Train Acc = {tr_acc*100:.2f}%)")

    # Overall Out-Of-Fold metrics
    overall_metrics = compute_classification_metrics(
        y_vec, oof_preds, y_prob=oof_probs if np.any(oof_probs) else None, class_names=classes
    )

    # Summary DataFrame
    summary_df = pd.DataFrame(fold_records)
    mean_row = {
        "Fold": "Mean",
        "Train Accuracy": round(summary_df["Train Accuracy"].mean(), 2),
        "Val Accuracy": round(summary_df["Val Accuracy"].mean(), 2),
        "Val Macro F1": round(summary_df["Val Macro F1"].mean(), 4),
        "Val Weighted F1": round(summary_df["Val Weighted F1"].mean(), 4),
    }
    std_row = {
        "Fold": "Std (+/-)",
        "Train Accuracy": round(summary_df["Train Accuracy"].std(), 2),
        "Val Accuracy": round(summary_df["Val Accuracy"].std(), 2),
        "Val Macro F1": round(summary_df["Val Macro F1"].std(), 4),
        "Val Weighted F1": round(summary_df["Val Weighted F1"].std(), 4),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    if verbose:
        print(f"-------------------------------------------------------")
        print(f"  Overall OOF Accuracy: {overall_metrics['accuracy_pct']}% | Macro F1: {overall_metrics['macro_f1']}")
        print(f"=======================================================\n")

    # Final test predictions
    test_preds = None
    if accum_test_probs is not None:
        best_class_idx = np.argmax(accum_test_probs, axis=1)
        test_preds = np.array([classes[i] for i in best_class_idx])

    return {
        "model_name": model_name,
        "fold_summary": summary_df,
        "oof_preds": oof_preds,
        "oof_probs": oof_probs,
        "test_preds": test_preds,
        "test_probs": accum_test_probs,
        "overall_metrics": overall_metrics,
        "models": fitted_models,
    }


def create_submission_file(
    test_df: pd.DataFrame,
    predictions: Union[pd.Series, np.ndarray, List[str]],
    save_name: str = "submission.csv",
    verbose: bool = True,
) -> Path:
    """
    Generate and validate a Kaggle-compliant submission CSV file.

    Parameters:
    -----------
    test_df : pd.DataFrame
        Test dataset containing the 'id' column.
    predictions : Series, ndarray, or list
        Predicted activity labels for each test ID.
    save_name : str
        Filename saved inside reports/submissions/.

    Returns:
    --------
    Path
        Path to the verified submission file.
    """
    if "id" not in test_df.columns:
        raise ValueError("test_df must contain an 'id' column.")

    if len(test_df) != len(predictions):
        raise ValueError(
            f"Length mismatch: test_df has {len(test_df)} rows, but predictions has {len(predictions)} elements."
        )

    preds_list = list(predictions)
    # If predictions are numeric IDs, convert to string labels
    if len(preds_list) > 0 and isinstance(preds_list[0], (int, np.integer)):
        preds_list = [ID2LABEL[p] for p in preds_list]

    sub_df = pd.DataFrame({
        "id": test_df["id"].values,
        "target": preds_list,
    })

    # Validate Kaggle formatting
    assert list(sub_df.columns) == ["id", "target"], "Columns must be exactly ['id', 'target']"
    assert len(sub_df) == 403, f"Expected 403 test rows, got {len(sub_df)}"
    assert not sub_df["target"].isna().any(), "Submission contains null predictions!"

    out_path = SUBMISSIONS_DIR / (save_name if save_name.endswith(".csv") else f"{save_name}.csv")
    sub_df.to_csv(out_path, index=False)

    if verbose:
        print(f"Kaggle submission successfully saved to: {out_path}")
        print(f"  - Total Rows: {len(sub_df)}")
        print(f"  - First 3 Predictions:\n{sub_df.head(3).to_string(index=False)}")
        print(f"  - Predicted Class Distribution:")
        print(sub_df["target"].value_counts().to_string())

    return out_path
