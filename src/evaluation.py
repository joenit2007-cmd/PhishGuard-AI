"""
Model Evaluation and Visualization Module for Phishing Email Detection System.

Computes classification metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
and generates high-resolution evaluation charts.
"""

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

# Ensure a writable matplotlib cache directory
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Classifier",
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics for a binary classifier.

    Args:
        y_true: Ground truth binary labels (0 = SAFE, 1 = PHISHING).
        y_pred: Predicted binary labels.
        y_proba: Predicted probability of positive class (PHISHING).
        model_name: Name of algorithm.

    Returns:
        Dictionary of formatted evaluation metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred).tolist()

    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba))
        except Exception:
            roc_auc = None

    report = classification_report(
        y_true,
        y_pred,
        target_names=["SAFE", "PHISHING"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def save_confusion_matrix_plot(
    cm: List[List[int]],
    model_name: str,
    output_path: str,
) -> str:
    """
    Renders and saves a seaborn confusion matrix heatmap.

    Args:
        cm: 2x2 confusion matrix list [[TN, FP], [FN, TP]].
        model_name: Model name for plot title.
        output_path: Path to save PNG image.

    Returns:
        Path of saved image.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm_array = np.array(cm)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["SAFE (0)", "PHISHING (1)"],
        yticklabels=["SAFE (0)", "PHISHING (1)"],
        linewidths=1.5,
        linecolor="#E2E8F0",
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=14, weight="bold", pad=15, color="#0B2545")
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10, color="#134074")
    ax.set_ylabel("True Label", fontsize=12, labelpad=10, color="#134074")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_model_comparison_plot(
    all_metrics: Dict[str, Dict[str, Any]],
    output_path: str,
) -> str:
    """
    Generates a multi-bar chart comparing Accuracy, Precision, Recall, and F1
    across all trained candidate models.

    Args:
        all_metrics: Dictionary mapping model names to their evaluation dicts.
        output_path: Output file path.

    Returns:
        Path of saved image.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    models = list(all_metrics.keys())
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]

    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    # Professional palette
    colors = ["#134074", "#0066CC", "#0284C7", "#38BDF8"]

    for i, (m_name, m_key) in enumerate(zip(metrics_names, metric_keys)):
        values = [all_metrics[m][m_key] for m in models]
        rects = ax.bar(x + i * width, values, width, label=m_name, color=colors[i], edgecolor="#0B2545", linewidth=0.8)
        # Add labels on top of bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                weight="bold",
                color="#0B2545",
            )

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11, color="#134074", weight="bold")
    ax.set_title("Algorithm Performance Comparison", fontsize=14, weight="bold", pad=15, color="#0B2545")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=10, weight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
