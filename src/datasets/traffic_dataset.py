import logging

import numpy as np
import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset
)

from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)


class TrafficDataset(Dataset):
    """
    Dataset for the Proposed TrafficCLIP model.

    Each sample contains:

        Image
        Text prompt
        8 statistical traffic features
        Label

    The 8 statistics are:

        1. Mean IAT
        2. IAT variance
        3. Jitter
        4. Entropy
        5. Mean packet length
        6. Packet-length variance
        7. Flow duration
        8. Burstiness
    """

    def __init__(
        self,
        npz_path,
        tokenizer_name="google-bert/bert-base-uncased",
        max_length=64,
        use_dynamic_prompts=True
    ):

        # --------------------------------------------------
        # Load processed dataset
        # --------------------------------------------------

        data = np.load(
            npz_path,
            allow_pickle=True
        )

        # --------------------------------------------------
        # Tokenizer
        # --------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name
        )

        self.max_length = max_length

        self.use_dynamic_prompts = (
            use_dynamic_prompts
        )

        # --------------------------------------------------
        # Images
        # --------------------------------------------------

        self.images = torch.from_numpy(
            data["x"]
        ).float()

        # --------------------------------------------------
        # Labels
        # --------------------------------------------------

        self.labels = torch.from_numpy(
            data["y"]
        ).long()

        # --------------------------------------------------
        # Class names
        # --------------------------------------------------

        self.class_names = data[
            "labels"
        ].tolist()

        # --------------------------------------------------
        # Statistical metadata
        # --------------------------------------------------

        self.metadata = data.get(
            "m",
            None
        )

        if self.metadata is None:

            raise ValueError(
                "The NPZ file does not contain "
                "statistical metadata 'm'."
            )

        # Convert metadata to numpy array
        self.metadata = np.asarray(
            self.metadata,
            dtype=np.float32
        )

        # --------------------------------------------------
        # Check statistics
        # --------------------------------------------------

        if self.metadata.ndim != 2:

            raise ValueError(
                "Statistics metadata must be a "
                "2-dimensional array."
            )

        if self.metadata.shape[1] != 8:

            raise ValueError(
                f"Expected 8 statistical features, "
                f"but found {self.metadata.shape[1]}."
            )

        logging.info(
            "Loaded dataset successfully."
        )

        logging.info(
            f"Number of samples: {len(self.images)}"
        )

        logging.info(
            f"Number of statistics: "
            f"{self.metadata.shape[1]}"
        )

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        # --------------------------------------------------
        # Image
        # --------------------------------------------------

        image = self.images[idx]

        # --------------------------------------------------
        # Label
        # --------------------------------------------------

        label = self.labels[idx]

        class_name = self.class_names[
            label.item()
        ]

        # --------------------------------------------------
        # Get 8 statistics
        # --------------------------------------------------

        stats_values = self.metadata[idx]

        (
            mean_iat,
            iat_variance,
            jitter,
            entropy,
            mean_packet_length,
            packet_length_variance,
            flow_duration,
            burstiness
        ) = stats_values

        # --------------------------------------------------
        # Dynamic text prompt
        # --------------------------------------------------

        if (
            self.use_dynamic_prompts
            and self.metadata is not None
        ):

            text_description = (
                f"A network traffic gray photo "
                f"of class {class_name} with "
                f"{mean_iat:.2f}ms mean IAT, "
                f"{iat_variance:.2f} IAT variance, "
                f"{jitter:.2f}ms jitter, "
                f"{entropy:.2f} byte entropy, "
                f"{mean_packet_length:.2f} byte mean "
                f"packet length, "
                f"{packet_length_variance:.2f} packet "
                f"length variance, "
                f"{flow_duration:.2f}ms flow duration, "
                f"and {burstiness:.4f} burstiness."
            )

        else:

            text_description = (
                f"A network traffic gray photo "
                f"of class {class_name}."
            )

        # --------------------------------------------------
        # Tokenize text
        # --------------------------------------------------

        tokens = self.tokenizer(
            text_description,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # --------------------------------------------------
        # Create 8-dimensional statistics vector
        # --------------------------------------------------

        stats_vector = torch.tensor(
            stats_values,
            dtype=torch.float32
        )

        # --------------------------------------------------
        # Return sample
        # --------------------------------------------------

        return {
            "image": image,

            "input_ids": (
                tokens["input_ids"]
                .squeeze(0)
            ),

            "attention_mask": (
                tokens["attention_mask"]
                .squeeze(0)
            ),

            "stats": stats_vector,

            "label": label,

            "raw_text": text_description,

            "class_name": class_name
        }


def get_dataloader(
    npz_path,
    tokenizer,
    batch_size=64,
    max_length=64,
    seed=42,
    use_dynamic_prompts=True
):
    """
    Creates stratified Train, Validation and Test
    DataLoaders.

    Split:

        70% Training
        15% Validation
        15% Testing
    """

    # --------------------------------------------------
    # Create complete dataset
    # --------------------------------------------------

    full_dataset = TrafficDataset(
        npz_path=npz_path,
        tokenizer_name=tokenizer,
        max_length=max_length,
        use_dynamic_prompts=use_dynamic_prompts
    )

    # --------------------------------------------------
    # Extract labels
    # --------------------------------------------------

    targets = (
        full_dataset.labels
        .numpy()
        .tolist()
    )

    indices = np.arange(
        len(full_dataset)
    )

    # --------------------------------------------------
    # First split:
    #
    # 70% Train
    # 30% Temporary
    # --------------------------------------------------

    train_indices, temp_indices = (
        train_test_split(
            indices,
            test_size=0.30,
            stratify=targets,
            random_state=seed
        )
    )

    # --------------------------------------------------
    # Second split:
    #
    # 15% Validation
    # 15% Test
    # --------------------------------------------------

    temp_targets = [
        targets[i]
        for i in temp_indices
    ]

    val_indices, test_indices = (
        train_test_split(
            temp_indices,
            test_size=0.50,
            stratify=temp_targets,
            random_state=seed
        )
    )

    # --------------------------------------------------
    # Create subsets
    # --------------------------------------------------

    train_set = Subset(
        full_dataset,
        train_indices
    )

    val_set = Subset(
        full_dataset,
        val_indices
    )

    test_set = Subset(
        full_dataset,
        test_indices
    )

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------

    logging.info(
        f"Total Samples: {len(full_dataset)}"
    )

    logging.info(
        f"Training Samples: {len(train_set)}"
    )

    logging.info(
        f"Validation Samples: {len(val_set)}"
    )

    logging.info(
        f"Testing Samples: {len(test_set)}"
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


def check_distribution(
    loader,
    name
):
    """
    Display class distribution.
    """

    all_labels = []

    for batch in loader:

        all_labels.extend(
            batch["label"].tolist()
        )

    unique, counts = np.unique(
        all_labels,
        return_counts=True
    )

    distribution = dict(
        zip(
            unique.tolist(),
            counts.tolist()
        )
    )

    logging.info(
        f"{name} distribution: "
        f"{distribution}"
    )


# ------------------------------------------------------
# Simple test
# ------------------------------------------------------

if __name__ == "__main__":

    print(
        "TrafficDataset module loaded successfully."
    )

    print(
        "Expected statistics per sample: 8"
    )