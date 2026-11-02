"""
Evaluation metrics, test matrices, and reporting utilities for multiclass HAR classification.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from iml_assignment.config import (
    ACTIVITY_CLASSES,
    BENCHMARK_SUBMISSION_PATH,
    FIGURES_DIR,
    ID2LABEL,
    METRICS_DIR,
)


def compute_classification_metrics(
    y_true: Union[pd.Series, np.ndarray, List],
    y_pred: Union[pd.Series, np.ndarray, List],
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Union[float, Dict]]:
    """
    Compute comprehensive classification metrics for multiclass predictions.

    Calculates:
    - Overall Accuracy (primary Kaggle competition metric).
    - Macro-Averaged F1 Score (unweighted mean across all 12 classes; vital for evaluating rare activities).
    - Weighted F1 Score (weighted by class support).
    - Balanced Accuracy (unweighted mean of recalls across all classes).
    - Log Loss / Cross-Entropy (if predicted probability distribution `y_prob` is provided).
    - Per-Class Breakdown: Precision, Recall, F1, and Support for every activity class.

    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray, List]
        Ground-truth activity labels.
    y_pred : Union[pd.Series, np.ndarray, List]
        Predicted activity labels.
    y_prob : Optional[np.ndarray], default=None
        Predicted probability array of shape (N, n_classes).
    class_names : Optional[List[str]], default=None
        List of target class names. Defaults to canonical `ACTIVITY_CLASSES`.

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing:
        - 'accuracy': float
        - 'accuracy_pct': float (accuracy as percentage 0-100)
        - 'macro_f1': float
        - 'weighted_f1': float
        - 'balanced_accuracy': float
        - 'log_loss': float (if y_prob provided)
        - 'per_class': Dict mapping each activity to its precision, recall, f1, support.

    Examples
    --------
    >>> metrics = compute_classification_metrics(y_val, y_pred)
    >>> print(f"Val Accuracy: {metrics['accuracy_pct']}% | Macro F1: {metrics['macro_f1']}")
    """
    labels = class_names if class_names is not None else ACTIVITY_CLASSES

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class_dict = {}
    for i, cls in enumerate(labels):
        per_class_dict[cls] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(sup[i]),
        }

    results = {
        "accuracy": round(acc, 4),
        "accuracy_pct": round(acc * 100, 2),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "per_class": per_class_dict,
    }

    if y_prob is not None:
        try:
            results["log_loss"] = round(float(log_loss(y_true, y_prob, labels=sorted(labels))), 4)
        except Exception:
            pass

    return results


def get_classification_report_df(
    y_true: Union[pd.Series, np.ndarray, List],
    y_pred: Union[pd.Series, np.ndarray, List],
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate a clean pandas DataFrame representing the classification report.

    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray, List]
        True activity class labels.
    y_pred : Union[pd.Series, np.ndarray, List]
        Predicted activity class labels.
    class_names : Optional[List[str]], default=None
        Ordered list of class labels. Defaults to canonical `ACTIVITY_CLASSES`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['precision', 'recall', 'f1-score', 'support'],
        indexed by activity class name plus overall accuracy and macro/weighted averages.

    Examples
    --------
    >>> report_df = get_classification_report_df(y_val, y_pred)
    >>> display(report_df)
    """
    labels = class_names if class_names is not None else ACTIVITY_CLASSES
    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report_dict).transpose()
    return df


def plot_confusion_matrix(
    y_true: Union[pd.Series, np.ndarray, List],
    y_pred: Union[pd.Series, np.ndarray, List],
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a publication-grade annotated heatmap confusion matrix for the 12 activity classes.

    Each cell shows both the normalized recall percentage and the raw sample count.

    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray, List]
        Ground-truth labels.
    y_pred : Union[pd.Series, np.ndarray, List]
        Predicted labels.
    class_names : Optional[List[str]], default=None
        List of activity names. Defaults to canonical `ACTIVITY_CLASSES`.
    normalize : bool, default=True
        Whether to normalize confusion matrix rows by class support (True = recall rates).
    title : Optional[str], default=None
        Chart title. If None, displays default title showing overall accuracy.
    save_name : Optional[str], default=None
        Filename to save the figure inside reports/figures/.
    figsize : Tuple[int, int], default=(10, 8)
        Dimensions of figure.
    cmap : str, default="Blues"
        Matplotlib/Seaborn colormap.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes objects.

    Examples
    --------
    >>> fig, ax = plot_confusion_matrix(y_true, oof_preds, save_name="fig_cm.png")
    >>> plt.show()
    """
    labels = class_names if class_names is not None else ACTIVITY_CLASSES

    # Compute raw counts and normalized matrices
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(all="ignore"):
        cm_norm = cm_counts.astype(float) / cm_counts.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    # Format cell annotations: percentage + (count)
    annot = np.empty_like(cm_counts, dtype=object)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if normalize:
                annot[i, j] = f"{cm_norm[i, j]*100:.1f}%\n({cm_counts[i, j]})"
            else:
                annot[i, j] = f"{cm_counts[i, j]}"

    fig, ax = plt.subplots(figsize=figsize)
    plot_data = cm_norm if normalize else cm_counts

    sns.heatmap(
        plot_data,
        annot=annot,
        fmt="",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Normalized Recall" if normalize else "Count"},
        linewidths=0.5,
        ax=ax,
    )

    ax.set_xlabel("Predicted Activity Class", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Activity Class (Ground Truth)", fontweight="bold", labelpad=10)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, rotation=0, fontsize=9)

    overall_acc = np.trace(cm_counts) / np.sum(cm_counts)
    default_title = f"Confusion Matrix (Overall Accuracy: {overall_acc*100:.2f}%)"
    ax.set_title(title or default_title, pad=15, fontweight="bold", fontsize=12)

    plt.tight_layout()

    if save_name:
        out_path = FIGURES_DIR / (save_name if save_name.endswith((".png", ".pdf")) else f"{save_name}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to: {out_path}")

    return fig, ax


def save_metrics(metrics: Dict, filename: str) -> Path:
    """
    Save an evaluation metrics dictionary to `reports/metrics/` as formatted JSON.

    Parameters
    ----------
    metrics : Dict
        Metrics dictionary produced by `compute_classification_metrics`.
    filename : str
        Filename (e.g. 'rf_baseline_metrics.json').

    Returns
    -------
    Path
        Path to the saved JSON file.

    Examples
    --------
    >>> path = save_metrics(metrics, "rf_baseline_metrics.json")
    """
    fname = filename if filename.endswith(".json") else f"{filename}.json"
    out_path = METRICS_DIR / fname
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {out_path}")
    return out_path


def evaluate_against_benchmark(
    predictions: Union[pd.DataFrame, pd.Series, np.ndarray, List[Any], str, Path, Dict[str, Any]],
    benchmark_path: Optional[Union[str, Path]] = None,
    plot_cm: bool = False,
    save_cm: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate test set predictions against the 100% Kaggle-scoring SOTA benchmark.

    Enables instant, offline test set evaluation and error diagnosis without requiring
    frequent Kaggle submissions:
    - Calculates test accuracy matching the exact Kaggle leaderboard score.
    - Computes macro and weighted F1-scores across all 12 activities.
    - Identifies exact sample IDs where predictions disagree with the benchmark.
    - Summarizes top confusion pairs (e.g., jogging vs running).
    - Optionally displays or saves a publication-grade confusion matrix heatmap.

    Parameters
    ----------
    predictions : Union[pd.DataFrame, pd.Series, np.ndarray, List[Any], str, Path, Dict[str, Any]]
        The test predictions to evaluate. Can be:
        - A results dict from `cross_validate_model` containing `'test_preds'`.
        - A file path to a submission CSV (e.g. `'reports/submissions/submission.csv'`).
        - A DataFrame with columns `['id', 'target']` or `['target']`.
        - A Series or 1D list/array of 403 predicted activity label strings or integer IDs.
        - A 2D probability matrix of shape (403, 12).
    benchmark_path : Optional[Union[str, Path]], default=None
        Path to the reference benchmark CSV. Defaults to `benchmark_sota/benchmark_submission.csv`.
    plot_cm : bool, default=False
        Whether to display a confusion matrix comparing predictions to benchmark.
    save_cm : Optional[str], default=None
        Filename to save the confusion matrix figure in `reports/figures/`.
    verbose : bool, default=True
        Whether to print a formatted evaluation summary to the console.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'accuracy': float (0.0 to 1.0)
        - 'accuracy_pct': float (0.0 to 100.0)
        - 'macro_f1': float
        - 'weighted_f1': float
        - 'matches': int (number of correct predictions out of 403)
        - 'mismatches': int (number of disagreements)
        - 'mismatches_df': pd.DataFrame of disagreeing samples (`['id', 'predicted', 'benchmark']`)
        - 'confusion_pairs': pd.Series counting the most frequent prediction errors
        - 'report_df': pd.DataFrame per-class classification report
        - 'fig': Optional matplotlib Figure if plot_cm or save_cm is set.

    Examples
    --------
    >>> # Evaluate cross-validation test predictions directly:
    >>> eval_res = evaluate_against_benchmark(rf_results)
    >>>
    >>> # Evaluate a submission CSV:
    >>> eval_res = evaluate_against_benchmark("reports/submissions/sub_rf.csv")
    >>> print("Benchmark Accuracy:", eval_res["accuracy_pct"], "%")
    >>> display(eval_res["mismatches_df"].head())
    """
    bench_p = Path(benchmark_path) if benchmark_path is not None else BENCHMARK_SUBMISSION_PATH
    if not bench_p.exists():
        raise FileNotFoundError(
            f"Benchmark submission file not found at: {bench_p}. "
            "Please ensure 'benchmark_sota/benchmark_submission.csv' exists."
        )

    bench_df = pd.read_csv(bench_p)
    if "target" not in bench_df.columns:
        raise ValueError(f"Benchmark CSV must contain a 'target' column. Got: {bench_df.columns.tolist()}")

    test_ids = bench_df["id"].values if "id" in bench_df.columns else None

    # Case 1: Dictionary from cross_validate_model
    if isinstance(predictions, dict):
        if "test_preds" in predictions and predictions["test_preds"] is not None:
            raw_preds = predictions["test_preds"]
        elif "test_probs" in predictions and predictions["test_probs"] is not None:
            raw_preds = predictions["test_probs"]
        else:
            raise KeyError("Dictionary passed to evaluate_against_benchmark must contain 'test_preds' or 'test_probs'.")
    # Case 2: File path to submission CSV
    elif isinstance(predictions, (str, Path)) and str(predictions).endswith(".csv"):
        pred_p = Path(predictions)
        if not pred_p.exists():
            from iml_assignment.config import PROJECT_ROOT
            if (PROJECT_ROOT / pred_p).exists():
                pred_p = PROJECT_ROOT / pred_p
            else:
                raise FileNotFoundError(f"Prediction CSV file not found: {pred_p}")
        pdf = pd.read_csv(pred_p)
        if "id" in pdf.columns and "target" in pdf.columns and test_ids is not None:
            merged = pd.merge(bench_df[["id", "target"]], pdf[["id", "target"]], on="id", suffixes=("_bench", "_pred"))
            raw_preds = merged["target_pred"].values
            bench_df = merged[["id", "target_bench"]].rename(columns={"target_bench": "target"})
        elif "target" in pdf.columns:
            raw_preds = pdf["target"].values
        else:
            raw_preds = pdf.iloc[:, -1].values
    # Case 3: DataFrame
    elif isinstance(predictions, pd.DataFrame):
        if "id" in predictions.columns and "target" in predictions.columns and test_ids is not None:
            merged = pd.merge(bench_df[["id", "target"]], predictions[["id", "target"]], on="id", suffixes=("_bench", "_pred"))
            raw_preds = merged["target_pred"].values
            bench_df = merged[["id", "target_bench"]].rename(columns={"target_bench": "target"})
        elif "target" in predictions.columns:
            raw_preds = predictions["target"].values
        else:
            raw_preds = predictions.iloc[:, 0].values
    else:
        raw_preds = predictions

    # Format into 1D list of string activity names
    if isinstance(raw_preds, np.ndarray) and raw_preds.ndim == 2:
        best_idx = np.argmax(raw_preds, axis=1)
        preds_list = [ACTIVITY_CLASSES[i] for i in best_idx]
    else:
        preds_list = list(raw_preds)

    if len(preds_list) > 0 and isinstance(preds_list[0], (int, np.integer)):
        preds_list = [ID2LABEL[int(p)] for p in preds_list]
    else:
        preds_list = [str(p) for p in preds_list]

    if len(preds_list) != len(bench_df):
        raise ValueError(
            f"Length mismatch: predictions has {len(preds_list)} elements, but benchmark has {len(bench_df)} rows."
        )

    y_true = bench_df["target"].values
    y_pred = np.array(preds_list)

    n_total = len(y_true)
    matches_mask = (y_pred == y_true)
    n_matches = int(np.sum(matches_mask))
    n_mismatches = n_total - n_matches
    acc_pct = round(n_matches / n_total * 100, 2)
    acc = round(n_matches / n_total, 4)

    macro_f1 = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
    weighted_f1 = round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4)
    report_df = get_classification_report_df(y_true, y_pred, class_names=ACTIVITY_CLASSES)

    # Disagreements DataFrame
    ids = bench_df["id"].values if "id" in bench_df.columns else np.arange(n_total)
    mismatches_df = pd.DataFrame({
        "id": ids[~matches_mask],
        "predicted": y_pred[~matches_mask],
        "benchmark": y_true[~matches_mask],
    })

    confusion_pairs = mismatches_df.groupby(["predicted", "benchmark"]).size().sort_values(ascending=False)

    if verbose:
        print(f"\n=======================================================")
        print(f"  Test Benchmark Evaluation (SOTA Pseudo-Ground Truth)")
        print(f"=======================================================")
        print(f"  Benchmark Reference: {bench_p.name} (403 samples)")
        print(f"  Matches:     {n_matches} / {n_total} ({acc_pct:.2f}%)")
        print(f"  Mismatches:  {n_mismatches} / {n_total} ({100 - acc_pct:.2f}%)")
        print(f"  Macro F1:    {macro_f1:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")
        print(f"-------------------------------------------------------")
        if n_mismatches > 0:
            print(f"  Top Disagreement Pairs (Predicted -> Benchmark):")
            for (p_cls, b_cls), count in confusion_pairs.head(5).items():
                print(f"    - '{p_cls}' predicted as '{b_cls}': {count} sample(s)")
            print(f"=======================================================\n")
        else:
            print(f"  PERFECT 100% MATCH WITH SOTA BENCHMARK!")
            print(f"=======================================================\n")

    fig = None
    if plot_cm or save_cm:
        title = f"Test Predictions vs SOTA Benchmark (Accuracy: {acc_pct:.2f}%)"
        fig, _ = plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=ACTIVITY_CLASSES,
            title=title,
            save_name=save_cm,
        )

    return {
        "accuracy": acc,
        "accuracy_pct": acc_pct,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "matches": n_matches,
        "mismatches": n_mismatches,
        "mismatches_df": mismatches_df,
        "confusion_pairs": confusion_pairs,
        "report_df": report_df,
        "fig": fig,
    }


# Convenient Aliases
test_against_benchmark = evaluate_against_benchmark
evaluate_test_predictions = evaluate_against_benchmark
