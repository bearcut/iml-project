"""
Training and Cross-Validation pipelines for Human Activity Recognition.

Includes:
- Stratified 5-Fold Cross Validation preserving minority class ratios.
- Out-Of-Fold (OOF) prediction and probability tracking.
- Test set fold-ensemble averaging.
- Automated Kaggle submission generation and validation.
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
    1. Stratified Splits: Preserves identical 12-activity class proportions across every fold,
       guaranteeing balanced exposure for rare activities like class 11 ('jump_front_back').
    2. Out-Of-Fold (OOF) Tracking: Collects unbiased OOF predictions and probability distributions
       for the full training dataset, useful for threshold tuning and ensembling.
    3. Fold Averaging for Test: If `test_X` is provided, predicted probability distributions
       from all 5 fold models are averaged to maximize test generalization on Kaggle.

    Parameters
    ----------
    model : Union[str, BaseEstimator]
        Model identifier ('random_forest', 'rf', 'logistic_regression', 'lr', 'dummy')
        or an existing scikit-learn estimator / pipeline instance.
    X : Union[pd.DataFrame, np.ndarray]
        Training feature matrix of shape (N, n_features).
    y : Union[pd.Series, np.ndarray, List]
        Target labels of length N.
    n_splits : int, default=5
        Number of cross-validation folds.
    random_state : int, default=42
        Seed for deterministic stratified splitting.
    test_X : Optional[Union[pd.DataFrame, np.ndarray]], default=None
        Test set feature matrix. If provided, generates fold-averaged predictions.
    model_kwargs : Optional[Dict[str, Any]], default=None
        Hyperparameter overrides passed to `get_baseline_model` if `model` is a string.
    verbose : bool, default=True
        Whether to print fold-by-fold validation progress to console.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'model_name': str
        - 'fold_summary': pd.DataFrame with per-fold metrics and Mean +/- Std rows.
        - 'oof_preds': np.ndarray of predicted labels across all training samples.
        - 'oof_probs': np.ndarray of predicted probability distributions (N, 12).
        - 'test_preds': np.ndarray of final test predictions (if test_X provided).
        - 'test_probs': np.ndarray of final averaged test probabilities (N_test, 12).
        - 'overall_metrics': Dict of overall OOF accuracy, macro_f1, and per-class metrics.
        - 'models': List of fitted estimators across all folds.

    Examples
    --------
    >>> results = cross_validate_model("random_forest", X_train, y_train, test_X=X_test)
    >>> display(results["fold_summary"])
    >>> print("OOF Accuracy:", results["overall_metrics"]["accuracy_pct"])
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
    predictions: Union[pd.Series, np.ndarray, List[Any], pd.DataFrame],
    save_name: str = "submission.csv",
    class_order: Optional[Union[List[str], str]] = None,
    verbose: bool = True,
) -> Path:
    """
    Generate, format, and validate a Kaggle-compliant submission CSV file.

    Performs automated validation checks:
    - Verifies 'id' column exists and test length matches exactly 403 rows.
    - Converts integer class IDs or probability matrices to canonical string activity names.
    - Confirms zero NaN/null values exist in predictions.
    - Writes CSV formatted as 'id,target' to `reports/submissions/<save_name>`.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test dataset DataFrame containing the 'id' column (403 rows).
    predictions : Union[pd.Series, np.ndarray, List[Any], pd.DataFrame]
        Predicted activity labels for each test ID. Can be:
        1. 1D array/list of string labels (RECOMMENDED, e.g. `model.predict(X_test)` or `results["test_preds"]`).
        2. 1D array/list of integer IDs (0 to 11).
        3. 2D array or DataFrame of probabilities (N, 12).
    save_name : str, default="submission.csv"
        Output filename saved inside `reports/submissions/`.
    class_order : Optional[Union[List[str], str]], default=None
        Column order if passing raw 2D probabilities:
        - None: defaults to canonical `ACTIVITY_CLASSES` (matches `results["test_probs"]`).
        - 'alphabetical': sorted alphabetical order (matches raw `sklearn.model.predict_proba()`).
        - list of strings: explicit custom class ordering (e.g., `model.classes_`).
    verbose : bool, default=True
        Whether to print submission summary and class frequency breakdown.

    Returns
    -------
    Path
        Path to the verified output submission file.

    Raises
    ------
    ValueError
        If test_df is missing 'id' or row counts do not match predictions.
    AssertionError
        If columns, row count (403), or null checks fail.

    Examples
    --------
    >>> # Method 1: Using 5-Fold Cross-Validation test predictions
    >>> sub_path = create_submission_file(test_df, rf_results["test_preds"], "sub_rf_cv.csv")
    >>>
    >>> # Method 2: Using full model .predict()
    >>> test_preds = rf_model.predict(X_test)
    >>> sub_path = create_submission_file(test_df, test_preds, "sub_rf_full.csv")
    """
    if "id" not in test_df.columns:
        raise ValueError("test_df must contain an 'id' column.")

    if len(test_df) != len(predictions):
        raise ValueError(
            f"Length mismatch: test_df has {len(test_df)} rows, but predictions has {len(predictions)} elements."
        )

    # Determine class order for probability mapping if needed
    if class_order is None:
        target_classes = ACTIVITY_CLASSES
    elif isinstance(class_order, str):
        if class_order.lower() in ("alphabetical", "alpha", "sorted"):
            target_classes = sorted(ACTIVITY_CLASSES)
        elif class_order.lower() in ("canonical", "default"):
            target_classes = ACTIVITY_CLASSES
        else:
            raise ValueError(f"Unknown class_order shortcut '{class_order}'. Use 'canonical', 'alphabetical', or pass a list.")
    else:
        target_classes = list(class_order)

    # Normalize predictions input to a clean 1D list of string activity labels
    if isinstance(predictions, pd.DataFrame):
        # If columns match activity classes, take idxmax directly
        matching_cols = [c for c in target_classes if c in predictions.columns]
        if len(matching_cols) == len(target_classes):
            preds_list = list(predictions[target_classes].idxmax(axis=1))
        else:
            preds_list = [target_classes[i] for i in np.argmax(predictions.values, axis=1)]
    elif isinstance(predictions, np.ndarray) and predictions.ndim == 2:
        # 2D probability matrix (N, 12) -> convert via argmax to class names
        best_indices = np.argmax(predictions, axis=1)
        preds_list = [target_classes[i] for i in best_indices]
    else:
        preds_list = list(predictions)

    if len(preds_list) > 0:
        first_elem = preds_list[0]
        # Case: list of arrays/lists of probabilities
        if isinstance(first_elem, (list, tuple, np.ndarray)):
            preds_list = [target_classes[int(np.argmax(p))] for p in preds_list]
        # Case: integer class indices (0 to 11)
        elif isinstance(first_elem, (int, np.integer)):
            preds_list = [ID2LABEL[int(p)] for p in preds_list]
        # Case: string labels (verify they are strings)
        else:
            preds_list = [str(p) for p in preds_list]

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
