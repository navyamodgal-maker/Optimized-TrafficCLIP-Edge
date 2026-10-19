import os
import json
import logging
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no display needed on Kaggle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from src.datasets.traffic_dataset import get_dataloader, TrafficDataset
from src.models.proposed_traffic_clip import ProposedTrafficCLIP
from src.train_proposed import OPEN_SET_HELD_OUT_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

DATA_PATH = "data/traffic_data.npz"
OUTPUT_DIR = "output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path, num_classes):
    model = ProposedTrafficCLIP(num_classes=num_classes, stats_input_dim=8)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def run_standard_evaluation(model, test_loader, class_names):
    all_predictions = []
    all_labels = []

    for batch in test_loader:
        images = batch["image"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        stats = batch["stats"].to(DEVICE)
        labels = batch["label"]

        logits = model(images, input_ids, attention_mask, stats)
        predictions = torch.argmax(logits, dim=1).cpu()

        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average="macro", zero_division=0
    )

    # Per-class breakdown, needed for the bar chart and for sanity-checking
    # whether the near-ceiling accuracy is uniform across classes or hiding
    # a weak spot in one of them.
    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            all_labels, all_predictions, labels=list(range(len(class_names))),
            average=None, zero_division=0
        )
    )

    print()
    print("=" * 60)
    print("Standard Evaluation (known classes)")
    print("=" * 60)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Macro F1  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print()
    print("Per-class breakdown:")
    for i, name in enumerate(class_names):
        print(
            f"  {name:12s} | Precision: {per_class_precision[i]:.4f} | "
            f"Recall: {per_class_recall[i]:.4f} | F1: {per_class_f1[i]:.4f} | "
            f"Support: {per_class_support[i]}"
        )

    cm = confusion_matrix(
        all_labels, all_predictions, labels=list(range(len(class_names)))
    )

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "per_class_support": per_class_support.tolist(),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix -- ProposedTrafficCLIP (closed-set, 10 classes)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def plot_per_class_bars(class_names, precision, recall, f1, save_path):
    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(11, 6))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1")
    plt.xticks(x, class_names, rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Per-Class Precision / Recall / F1 -- ProposedTrafficCLIP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Per-class bar chart saved to: {save_path}")


@torch.no_grad()
def run_open_set_evaluation(model, npz_path, held_out_class_names):
    """
    Feed BOTH known-class test samples and held-out (unknown) class
    samples through the model, use the OpenSetDetector's confidence/
    entropy scoring to separate them, and report ROC-AUC for the
    known-vs-unknown binary detection task.
    """
    dataset = TrafficDataset(npz_path=npz_path, use_dynamic_prompts=True)

    labels_list = dataset.class_names
    held_out_indices = {
        labels_list.index(name) for name in held_out_class_names if name in labels_list
    }

    if not held_out_indices:
        logging.warning("No held-out classes found in dataset labels; skipping open-set eval.")
        return None

    is_unknown_true = []
    unknown_scores = []  # higher score = model thinks it's MORE likely unknown

    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    for batch in loader:
        images = batch["image"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        stats = batch["stats"].to(DEVICE)
        labels = batch["label"]

        result = model.predict_open_set(images, input_ids, attention_mask, stats)

        # Use entropy as the "unknown-ness" score: higher entropy ->
        # more likely to be flagged unknown / zero-day.
        entropy = result["entropy"].cpu().tolist()

        for label, entropy_value in zip(labels.tolist(), entropy):
            is_unknown_true.append(1 if label in held_out_indices else 0)
            unknown_scores.append(entropy_value)

    if len(set(is_unknown_true)) < 2:
        logging.warning(
            "Open-set eval needs both known and unknown samples in the "
            "dataset passed here; got only one class of ground truth."
        )
        return None

    auc = roc_auc_score(is_unknown_true, unknown_scores)
    fpr, tpr, thresholds = roc_curve(is_unknown_true, unknown_scores)

    print()
    print("=" * 60)
    print("Open-Set / Zero-Day Detection Evaluation")
    print("=" * 60)
    print(f"Held-out (unknown) classes : {held_out_class_names}")
    print(f"Known samples   : {is_unknown_true.count(0)}")
    print(f"Unknown samples : {is_unknown_true.count(1)}")
    print(f"ROC-AUC (known vs unknown) : {auc:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    roc_path = os.path.join(OUTPUT_DIR, "open_set_roc_curve.npz")
    np.savez(roc_path, fpr=fpr, tpr=tpr, thresholds=thresholds, auc=auc)
    print(f"ROC curve data saved to: {roc_path}")

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Open-Set / Zero-Day Detection ROC\nHeld-out: {held_out_class_names}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_plot_path = os.path.join(OUTPUT_DIR, "open_set_roc_curve.png")
    plt.savefig(roc_plot_path, dpi=200)
    plt.close()
    print(f"ROC curve plot saved to: {roc_plot_path}")

    return {"auc": auc, "fpr": fpr.tolist(), "tpr": tpr.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--closed-checkpoint",
        default=os.path.join(OUTPUT_DIR, "proposed_traffic_clip_best.pth"),
        help="Path to the checkpoint trained WITHOUT --open-set (all 10 classes). "
             "Used for the standard accuracy / confusion matrix / per-class report."
    )
    parser.add_argument(
        "--open-checkpoint",
        default=os.path.join(OUTPUT_DIR, "proposed_traffic_clip_open_set_best.pth"),
        help="Path to the checkpoint trained WITH --open-set. "
             "Used only for the ROC-AUC zero-day detection evaluation."
    )
    parser.add_argument("--data", default=DATA_PATH)
    args = parser.parse_args()

    full_data = np.load(args.data, allow_pickle=True)
    class_names = full_data["labels"].tolist()
    num_classes = len(class_names)

    results_summary = {}

    # ---- Standard evaluation: closed-set checkpoint, all 10 classes ----
    if not os.path.exists(args.closed_checkpoint):
        print(f"Checkpoint not found: {args.closed_checkpoint}")
        print("Run: python -m src.train_proposed")
    else:
        model, _ = load_model(args.closed_checkpoint, num_classes)

        _, _, test_loader = get_dataloader(
            npz_path=args.data,
            tokenizer="google-bert/bert-base-uncased",
            batch_size=16
        )

        standard_results = run_standard_evaluation(model, test_loader, class_names)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plot_confusion_matrix(
            np.array(standard_results["confusion_matrix"]),
            class_names,
            os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        )
        plot_per_class_bars(
            class_names,
            standard_results["per_class_precision"],
            standard_results["per_class_recall"],
            standard_results["per_class_f1"],
            os.path.join(OUTPUT_DIR, "per_class_metrics.png")
        )

        results_summary["standard_evaluation"] = standard_results

    # ---- Open-set evaluation: open-set checkpoint, held-out classes ----
    if not os.path.exists(args.open_checkpoint):
        print(f"Checkpoint not found: {args.open_checkpoint}")
        print("Run: python -m src.train_proposed --open-set")
    else:
        open_model, _ = load_model(args.open_checkpoint, num_classes)
        open_set_results = run_open_set_evaluation(
            open_model, args.data, OPEN_SET_HELD_OUT_CLASSES
        )
        if open_set_results is not None:
            results_summary["open_set_evaluation"] = {
                "auc": open_set_results["auc"],
                "held_out_classes": OPEN_SET_HELD_OUT_CLASSES,
            }

    summary_path = os.path.join(OUTPUT_DIR, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nFull results summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
