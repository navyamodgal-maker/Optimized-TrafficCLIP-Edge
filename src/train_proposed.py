import os
import logging
import argparse

import torch
import torch.nn as nn

from src.datasets.traffic_dataset import get_dataloader
from src.models.proposed_traffic_clip import ProposedTrafficCLIP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = "data/traffic_data.npz"
OUTPUT_DIR = "output"

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Classes held out entirely from training to simulate zero-day /
# unknown traffic. These indices are looked up by name at runtime.
OPEN_SET_HELD_OUT_CLASSES = ["Zeus", "BitTorrent"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        stats = batch["stats"].to(device)
        labels = batch["label"].to(device)

        logits = model(images, input_ids, attention_mask, stats)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / max(total, 1)

    return average_loss, accuracy


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        stats = batch["stats"].to(device)
        labels = batch["label"].to(device)

        logits = model(images, input_ids, attention_mask, stats)
        loss = criterion(logits, labels)

        total_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / max(total, 1)

    return average_loss, accuracy


def filter_out_classes(dataset_npz_path, held_out_class_names, filtered_path):
    """
    Build a filtered copy of the npz that excludes the held-out
    classes entirely, for open-set training. Held-out samples are
    kept separately (unfiltered) so they can be used at eval time
    as 'unknown' traffic.
    """
    import numpy as np

    data = np.load(dataset_npz_path, allow_pickle=True)

    labels = data["labels"].tolist()
    held_out_indices = {labels.index(name) for name in held_out_class_names if name in labels}

    if not held_out_indices:
        logging.warning("None of the held-out class names were found in the dataset labels.")

    y = data["y"]
    keep_mask = ~np.isin(y, list(held_out_indices))

    np.savez_compressed(
        filtered_path,
        x=data["x"][keep_mask],
        y=data["y"][keep_mask],
        labels=data["labels"],
        m=data["m"][keep_mask]
    )

    logging.info(
        f"Open-set filtering: kept {keep_mask.sum()} / {len(y)} samples "
        f"(excluded classes: {held_out_class_names})"
    )

    return filtered_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--open-set",
        action="store_true",
        help="Train with held-out classes excluded, for zero-day/open-set evaluation later"
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("Proposed TrafficCLIP - Training")
    print("=" * 60)
    print()
    print("Device:", DEVICE)

    if not os.path.exists(DATA_PATH):
        print()
        print("Dataset file not found.")
        print("Expected dataset:", DATA_PATH)
        print()
        print("Run scripts/01_download_pcap_data.py then scripts/02_build_dataset.py first.")
        return

    data_path_to_use = DATA_PATH

    if args.open_set:
        filtered_path = "data/traffic_data_open_set_train.npz"
        data_path_to_use = filter_out_classes(
            DATA_PATH,
            OPEN_SET_HELD_OUT_CLASSES,
            filtered_path
        )

    print()
    print("Loading dataset...")

    train_loader, val_loader, _ = get_dataloader(
        npz_path=data_path_to_use,
        tokenizer="google-bert/bert-base-uncased",
        batch_size=BATCH_SIZE
    )

    import numpy as np
    full_data = np.load(DATA_PATH, allow_pickle=True)
    num_classes_full = len(full_data["labels"].tolist())

    print()
    print("Creating ProposedTrafficCLIP model...")

    # NOTE: we always size the classifier head for the FULL class set
    # so that, in open-set mode, held-out classes simply never receive
    # training signal but the architecture stays consistent for eval.
    model = ProposedTrafficCLIP(
        num_classes=num_classes_full,
        stats_input_dim=8
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    best_validation_accuracy = 0.0
    checkpoint_name = "proposed_traffic_clip_open_set_best.pth" if args.open_set else "proposed_traffic_clip_best.pth"

    print()
    print("Starting training...")
    print()

    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        validation_loss, validation_accuracy = validate(
            model, val_loader, criterion, DEVICE
        )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Val Loss: {validation_loss:.4f} "
            f"| Val Acc: {validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            checkpoint_path = os.path.join(OUTPUT_DIR, checkpoint_name)

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "open_set": args.open_set,
                    "held_out_classes": OPEN_SET_HELD_OUT_CLASSES if args.open_set else []
                },
                checkpoint_path
            )

            print(f"Best model saved to: {checkpoint_path}")

    print()
    print("=" * 60)
    print("Training completed.")
    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
