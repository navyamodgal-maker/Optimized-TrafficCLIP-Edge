import os
import random
import csv

# =========================
# SETTINGS
# =========================

DATASET_DIR = "data/processed/USTC-TFC2016"
SPLIT_DIR = "splits"

SAMPLES_PER_CLASS = 5000

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42


# =========================
# SETUP
# =========================

random.seed(SEED)

os.makedirs(SPLIT_DIR, exist_ok=True)


# =========================
# GET 20 CLASSES
# =========================

classes = sorted(
    [
        folder
        for folder in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, folder))
    ]
)

print("Classes found:")
for class_name in classes:
    print(" -", class_name)

print(f"\nTotal classes: {len(classes)}")


# =========================
# CHECK CLASSES
# =========================

if len(classes) != 20:
    raise ValueError(
        f"Expected 20 classes, but found {len(classes)}"
    )


# =========================
# CREATE SPLITS
# =========================

train_rows = []
val_rows = []
test_rows = []


for class_name in classes:

    class_dir = os.path.join(
        DATASET_DIR,
        class_name
    )

    images = [
        file
        for file in os.listdir(class_dir)
        if file.lower().endswith(".png")
    ]

    print(
        f"\n{class_name}: "
        f"{len(images)} available"
    )

    if len(images) < SAMPLES_PER_CLASS:
        raise ValueError(
            f"{class_name} has only "
            f"{len(images)} images."
        )

    # Same random seed + same procedure
    # gives reproducible selection
    random.shuffle(images)

    selected = images[:SAMPLES_PER_CLASS]

    train_count = int(
        SAMPLES_PER_CLASS * TRAIN_RATIO
    )

    val_count = int(
        SAMPLES_PER_CLASS * VAL_RATIO
    )

    train_images = selected[
        :train_count
    ]

    val_images = selected[
        train_count:
        train_count + val_count
    ]

    test_images = selected[
        train_count + val_count:
    ]

    # -------------------------
    # TRAIN
    # -------------------------

    for image in train_images:

        path = os.path.join(
            DATASET_DIR,
            class_name,
            image
        ).replace("\\", "/")

        train_rows.append(
            [path, class_name]
        )

    # -------------------------
    # VALIDATION
    # -------------------------

    for image in val_images:

        path = os.path.join(
            DATASET_DIR,
            class_name,
            image
        ).replace("\\", "/")

        val_rows.append(
            [path, class_name]
        )

    # -------------------------
    # TEST
    # -------------------------

    for image in test_images:

        path = os.path.join(
            DATASET_DIR,
            class_name,
            image
        ).replace("\\", "/")

        test_rows.append(
            [path, class_name]
        )


# =========================
# SHUFFLE FINAL SPLITS
# =========================

random.shuffle(train_rows)
random.shuffle(val_rows)
random.shuffle(test_rows)


# =========================
# WRITE CSV FILES
# =========================

def write_csv(filename, rows):

    filepath = os.path.join(
        SPLIT_DIR,
        filename
    )

    with open(
        filepath,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["image_path", "label"]
        )

        writer.writerows(rows)


write_csv(
    "train.csv",
    train_rows
)

write_csv(
    "val.csv",
    val_rows
)

write_csv(
    "test.csv",
    test_rows
)


# =========================
# FINAL INFORMATION
# =========================

print("\n" + "=" * 50)
print("FAIR SPLIT CREATION COMPLETE")
print("=" * 50)

print(
    f"Training images   : {len(train_rows)}"
)

print(
    f"Validation images : {len(val_rows)}"
)

print(
    f"Test images       : {len(test_rows)}"
)

print(
    f"Total images      : "
    f"{len(train_rows) + len(val_rows) + len(test_rows)}"
)

print("\nFiles created:")
print("splits/train.csv")
print("splits/val.csv")
print("splits/test.csv")

print("\nRandom seed:", SEED)
print(
    "Samples per class:",
    SAMPLES_PER_CLASS
)