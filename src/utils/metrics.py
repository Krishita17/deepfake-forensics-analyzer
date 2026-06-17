import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, average_precision_score,
    precision_recall_curve, log_loss,
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import os


def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        metrics["average_precision"] = average_precision_score(y_true, y_prob)
        metrics["log_loss"] = log_loss(y_true, y_prob)
        metrics["eer"] = compute_eer(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0

    return metrics


def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer)


def plot_roc_curve(y_true, y_prob, save_path="outputs/figures/roc_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    eer = compute_eer(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#1A1A2E")

    ax = axes[0]
    ax.set_facecolor("#1A1A2E")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#5B8DEF")
    ax.plot(fpr, tpr, color="#5B8DEF", linewidth=2.5, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="#666", linewidth=1)
    ax.plot(eer, 1 - eer, "o", markersize=10, color="#FFA502", label=f"EER = {eer:.4f}")
    ax.set_xlabel("False Positive Rate", color="#E8E8E8", fontsize=12)
    ax.set_ylabel("True Positive Rate", color="#E8E8E8", fontsize=12)
    ax.set_title("ROC Curve", color="#E8E8E8", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#1A1A2E", edgecolor="#2D2D44", labelcolor="#E8E8E8")
    ax.tick_params(colors="#E8E8E8")

    ax2 = axes[1]
    ax2.set_facecolor("#1A1A2E")
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax2.fill_between(recall, precision, alpha=0.15, color="#00D4AA")
    ax2.plot(recall, precision, color="#00D4AA", linewidth=2.5, label=f"PR (AP = {ap:.4f})")
    ax2.set_xlabel("Recall", color="#E8E8E8", fontsize=12)
    ax2.set_ylabel("Precision", color="#E8E8E8", fontsize=12)
    ax2.set_title("Precision-Recall Curve", color="#E8E8E8", fontsize=14, fontweight="bold")
    ax2.legend(facecolor="#1A1A2E", edgecolor="#2D2D44", labelcolor="#E8E8E8")
    ax2.tick_params(colors="#E8E8E8")

    fig.suptitle("Model Performance Evaluation", color="#E8E8E8", fontsize=16, fontweight="bold")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1A1A2E")
    plt.close(fig)
    return save_path


def plot_confusion_matrix(y_true, y_pred, save_path="outputs/figures/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1A1A2E")

    for ax, data, title, fmt in [
        (axes[0], cm, "Confusion Matrix (Counts)", "d"),
        (axes[1], cm_normalized, "Confusion Matrix (Normalized)", ".2%"),
    ]:
        ax.set_facecolor("#1A1A2E")
        im = ax.imshow(data, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Real", "Fake"], color="#E8E8E8")
        ax.set_yticklabels(["Real", "Fake"], color="#E8E8E8")
        ax.set_xlabel("Predicted", color="#E8E8E8", fontsize=12)
        ax.set_ylabel("Actual", color="#E8E8E8", fontsize=12)
        ax.set_title(title, color="#E8E8E8", fontsize=13, fontweight="bold")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(data[i, j], fmt),
                        ha="center", va="center", color="white", fontsize=16, fontweight="bold")

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1A1A2E")
    plt.close(fig)
    return save_path
