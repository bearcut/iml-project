"""
Evaluation metrics, test matrices, and reporting utilities for multiclass HAR classification.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
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
    FIGURES_DIR,
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

    Returns:
    --------
    dict with:
      - accuracy: Overall accuracy (Kaggle competition metric)
      - macro_f1: Unweighted mean F1 across all 12 classes (critical for minority class 11)
      - weighted_f1: Frequency-weighted mean F1
      - balanced_accuracy: Mean recall across classes
      - log_loss: Multiclass cross-entropy (if y_prob provided)
      - per_class: Dictionary of precision, recall, F1, and support for each class
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
    Generate a clean pandas DataFrame of precision, recall, f1-score, and support.
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
    Plot an annotated heatmap confusion matrix with normalized percentages and sample counts.
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
    """Save metrics dictionary to reports/metrics/ as JSON."""
    fname = filename if filename.endswith(".json") else f"{filename}.json"
    out_path = METRICS_DIR / fname
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {out_path}")
    return out_path
