import os
import logging
import argparse

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
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

    print()
    print("=" * 60)
    print("Standard Evaluation (known classes)")
    print("=" * 60)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Macro F1  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


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

    return {"auc": auc, "fpr": fpr, "tpr": tpr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(OUTPUT_DIR, "proposed_traffic_clip_open_set_best.pth"),
        help="Path to the checkpoint trained with --open-set"
    )
    parser.add_argument("--data", default=DATA_PATH)
    args = parser.parse_args()

    full_data = np.load(args.data, allow_pickle=True)
    class_names = full_data["labels"].tolist()
    num_classes = len(class_names)

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run: python -m src.train_proposed --open-set")
        return

    model, checkpoint = load_model(args.checkpoint, num_classes)

    _, _, test_loader = get_dataloader(
        npz_path=args.data,
        tokenizer="google-bert/bert-base-uncased",
        batch_size=16
    )

    run_standard_evaluation(model, test_loader, class_names)

    run_open_set_evaluation(model, args.data, OPEN_SET_HELD_OUT_CLASSES)


if __name__ == "__main__":
    main()
