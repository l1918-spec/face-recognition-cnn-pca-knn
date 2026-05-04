"""Metrics + plots for model comparison."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {"accuracy": float(acc), "precision": float(p),
            "recall": float(r), "f1": float(f1)}


def plot_confusion(y_true, y_pred, title, save_path, max_labels=20):
    cm = confusion_matrix(y_true, y_pred)
    n = cm.shape[0]
    fig_size = max(6, min(12, n * 0.3))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    annot = n <= max_labels
    sns.heatmap(cm, annot=annot, fmt="d", cmap="Blues", cbar=False, ax=ax,
                square=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(results, save_path):
    """results: dict[name] -> metrics dict (accuracy, train_time, infer_time)."""
    names = list(results.keys())
    acc = [results[n]["accuracy"] for n in names]
    f1 = [results[n]["f1"] for n in names]
    train_t = [results[n].get("train_time", 0) for n in names]
    infer_t = [results[n].get("infer_time_ms", 0) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(len(names))
    w = 0.35
    axes[0].bar(x - w/2, acc, w, label="Accuracy", color="#4C72B0")
    axes[0].bar(x + w/2, f1, w, label="F1 (macro)", color="#55A868")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Accuracy / F1 by model")
    axes[0].legend()
    for i, v in enumerate(acc):
        axes[0].text(i - w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)

    ax2 = axes[1]
    ax3 = ax2.twinx()
    ax2.bar(x - w/2, train_t, w, label="Train (s)", color="#C44E52")
    ax3.bar(x + w/2, infer_t, w, label="Infer (ms/img)", color="#8172B2")
    ax2.set_xticks(x); ax2.set_xticklabels(names)
    ax2.set_ylabel("Train time (s)")
    ax3.set_ylabel("Inference time (ms/img)")
    ax2.set_title("Compute cost")
    lines = ax2.get_legend_handles_labels()[0] + ax3.get_legend_handles_labels()[0]
    labels = ax2.get_legend_handles_labels()[1] + ax3.get_legend_handles_labels()[1]
    ax2.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_report(results, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)


def text_report(y_true, y_pred, label_names=None):
    return classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0
    )
