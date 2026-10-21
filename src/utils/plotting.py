"""
Plotting helpers for the 20-class ProposedTrafficCLIP results.

Pure matplotlib / seaborn (no torch), so figures can be regenerated from
output/results_summary.json on any machine, e.g. via
scripts/04_generate_figures.py, without the model or dataset.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, class_names, save_path, normalize=False):
    """Confusion matrix heatmap sized for 20 classes.

    normalize=True shows each row as a fraction of its true class (recall
    per cell), which is easier to read when classes have unequal support.
    """
    cm = np.asarray(cm, dtype=float)
    n = len(class_names)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        data = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        fmt, vmax = ".2f", 1.0
        title = f"Normalized Confusion Matrix -- ProposedTrafficCLIP (closed-set, {n} classes)"
    else:
        data = cm
        fmt, vmax = "d", None
        title = f"Confusion Matrix -- ProposedTrafficCLIP (closed-set, {n} classes)"

    plt.figure(figsize=(14, 11.5))
    sns.heatmap(
        data if normalize else data.astype(int),
        annot=True, fmt=fmt, cmap="Blues", vmin=0, vmax=vmax,
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"size": 8}, linewidths=0.3, linecolor="white",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=60, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def plot_per_class_bars(class_names, precision, recall, f1, save_path):
    x = np.arange(len(class_names))
    width = 0.27

    plt.figure(figsize=(16, 6.5))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1")
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(
        f"Per-Class Precision / Recall / F1 -- ProposedTrafficCLIP ({len(class_names)} classes)"
    )
    # Legend outside the axes so it never hides the last classes' bars.
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Per-class bar chart saved to: {save_path}")


def plot_training_curves(closed_log, open_log, save_path):
    """Train/val loss and accuracy per epoch, closed-set vs open-set run."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for log, label, style in [
        (closed_log, "Closed-set (20 classes)", "-"),
        (open_log, "Open-set (18 known classes)", "--"),
    ]:
        if not log:
            continue
        epochs = [r["epoch"] for r in log]
        axes[0].plot(epochs, [r["train_loss"] for r in log], style, marker="o",
                     label=f"{label} - train")
        axes[0].plot(epochs, [r["val_loss"] for r in log], style, marker="s",
                     label=f"{label} - val")
        axes[1].plot(epochs, [r["train_acc"] for r in log], style, marker="o",
                     label=f"{label} - train")
        axes[1].plot(epochs, [r["val_acc"] for r in log], style, marker="s",
                     label=f"{label} - val")

    axes[0].set_title("Loss per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].set_title("Accuracy per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("ProposedTrafficCLIP training curves (USTC-TFC2016, 20 classes)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Training curves saved to: {save_path}")


def plot_qat_comparison(qat, save_path):
    """Bar chart of test accuracy: FP32 vs naive fake-quant vs after QAT."""
    names = ["FP32 baseline", "Fake-quant\n(no QAT)", "Fake-quant\nafter QAT"]
    values = [qat["fp32_baseline_accuracy"], qat["naive_fake_quant_accuracy"],
              qat["qat_finetuned_accuracy"]]

    plt.figure(figsize=(6.5, 5))
    bars = plt.bar(names, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.4f}",
                 ha="center", va="bottom")
    plt.axhline(0.85, color="gray", linestyle="--", linewidth=1,
                label="0.85 target (teacher's Macro-F1 goal)")
    lo = min(values) - 0.05
    plt.ylim(max(lo, 0), 1.0)
    plt.ylabel("Test accuracy")
    plt.title("Fusion-head QAT (20 classes, test set)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"QAT comparison saved to: {save_path}")
