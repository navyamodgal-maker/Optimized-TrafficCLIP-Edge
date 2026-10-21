
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

import matplotlib.pyplot as plt


# ---------------- CONFIGURATION ----------------

SEED = 42
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3

DATA_DIR = Path(
    "/content/YaTC_data/YaTC_datasets/CICIoT2022_MFR"
)
OUTPUT_DIR = Path("output/image_only")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ---------------- DATASET ----------------

class TrafficImageDataset(Dataset):

    def __init__(self, root, class_names):
        self.root = Path(root)
        self.classes = class_names
        self.class_to_idx = {
            name: idx for idx, name in enumerate(class_names)
        }

        self.samples = []

        for class_name in self.classes:
            folder = self.root / class_name

            for file in folder.rglob("*"):
                if file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    self.samples.append(
                        (file, self.class_to_idx[class_name])
                    )

        self.transform = transforms.Compose([
            transforms.Resize((40, 40)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as im:
            image = self.transform(im.convert("L"))

        return image, label


classes = sorted([
    p.name for p in (DATA_DIR / "train").iterdir()
    if p.is_dir()
])

train_dataset = TrafficImageDataset(
    DATA_DIR / "train", classes
)

test_dataset = TrafficImageDataset(
    DATA_DIR / "test", classes
)

assert len(train_dataset) > 0, "No training images found"
assert len(test_dataset) > 0, "No test images found"

print("Classes:", classes)
print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))


# ---------------- TRAIN / VALIDATION SPLIT ----------------

labels = np.array([
    label for _, label in train_dataset.samples
])

train_indices, val_indices = train_test_split(
    np.arange(len(labels)),
    test_size=0.15,
    random_state=SEED,
    stratify=labels
)

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(train_dataset, val_indices)

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

print("Training split:", len(train_subset))
print("Validation split:", len(val_subset))


# ---------------- MODEL ----------------

# ResNet-18 initialized from scratch.
# This is an image-only adaptation, not the full multimodal model.

model = models.resnet18(weights=None)

model.conv1 = nn.Conv2d(
    3, 64, kernel_size=3, stride=1, padding=1, bias=False
)

model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, len(classes))

model = model.to(DEVICE)


# Class-weighted loss helps with class imbalance.

train_labels = labels[train_indices]

class_counts = np.bincount(
    train_labels, minlength=len(classes)
)

class_weights = len(train_labels) / (
    len(classes) * np.maximum(class_counts, 1)
)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(
        class_weights, dtype=torch.float32, device=DEVICE
    )
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ---------------- TRAINING AND VALIDATION ----------------

best_val_accuracy = -1.0
history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}


def run_epoch(loader, training):
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, targets in loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            if training:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)

    return total_loss / total, correct / total


for epoch in range(EPOCHS):

    train_loss, train_acc = run_epoch(train_loader, True)
    val_loss, val_acc = run_epoch(val_loader, False)

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "epoch": epoch + 1,
            "validation_accuracy": val_acc,
            "architecture": "ResNet18-image-only",
            "seed": SEED
        }, OUTPUT_DIR / "best_model.pth")

        print("Saved best checkpoint.")


# ---------------- TRAINING GRAPHS ----------------

epochs = range(1, EPOCHS + 1)

plt.figure()
plt.plot(epochs, history["train_loss"], label="Train loss")
plt.plot(epochs, history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=200)
plt.show()
plt.close()

plt.figure()
plt.plot(epochs, history["train_accuracy"], label="Train accuracy")
plt.plot(epochs, history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=200)
plt.show()
plt.close()


# ---------------- FINAL TEST EVALUATION ----------------

checkpoint = torch.load(
    OUTPUT_DIR / "best_model.pth",
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_predictions = []
all_targets = []

with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(DEVICE)

        predictions = model(images).argmax(1).cpu().numpy()

        all_predictions.extend(predictions.tolist())
        all_targets.extend(targets.numpy().tolist())

accuracy = accuracy_score(all_targets, all_predictions)
macro_f1 = f1_score(
    all_targets, all_predictions,
    average="macro", zero_division=0
)

print("\n===== FINAL TEST RESULTS =====")
print(f"Test accuracy: {accuracy * 100:.2f}%")
print(f"Macro F1:      {macro_f1 * 100:.2f}%")
print("\nClassification report:")

report = classification_report(
    all_targets,
    all_predictions,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0
)

print(report)

with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(
    all_targets,
    all_predictions,
    labels=list(range(len(classes)))
)

plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation="nearest")
plt.title("CICIoT2022 YaTC — Image-only Confusion Matrix")
plt.colorbar()
plt.xticks(range(len(classes)), classes, rotation=90)
plt.yticks(range(len(classes)), classes)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
plt.show()
plt.close()

np.savez(
    OUTPUT_DIR / "test_predictions.npz",
    y_true=np.array(all_targets),
    y_pred=np.array(all_predictions),
    classes=np.array(classes)
)

print(f"\nBest validation accuracy: {best_val_accuracy * 100:.2f}%")
print(f"Test accuracy: {accuracy * 100:.2f}%")
print(f"Macro F1: {macro_f1 * 100:.2f}%")
print("Results saved to:", OUTPUT_DIR.resolve())
