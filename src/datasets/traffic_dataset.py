import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TrafficDataset(Dataset):
    def __init__(
        self,
        csv_file,
        class_to_idx=None,
        image_size=28,
        statistics_csv=None,
        stats_mean=None,
        stats_std=None
    ):
        self.data = pd.read_csv(csv_file)

        required = [
            "image_path",
            "label",
            "mean_iat",
            "jitter",
            "entropy"
        ]

        missing = [c for c in required if c not in self.data.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if class_to_idx is None:
            classes = sorted(self.data["label"].unique())
            self.class_to_idx = {
                name: idx for idx, name in enumerate(classes)
            }
        else:
            self.class_to_idx = class_to_idx

        raw = self.data[
            ["mean_iat", "jitter", "entropy"]
        ].astype("float32").values

        if stats_mean is None:
            stats_mean = raw.mean(axis=0)

        if stats_std is None:
            stats_std = raw.std(axis=0)

        stats_std = stats_std.copy()
        stats_std[stats_std < 1e-6] = 1.0

        self.stats_mean = torch.tensor(
            stats_mean,
            dtype=torch.float32
        )

        self.stats_std = torch.tensor(
            stats_std,
            dtype=torch.float32
        )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image = Image.open(
            row["image_path"]
        ).convert("L")

        image = self.transform(image)

        label = self.class_to_idx[row["label"]]

        raw_stats = torch.tensor(
            [
                row["mean_iat"],
                row["jitter"],
                row["entropy"]
            ],
            dtype=torch.float32
        )

        stats = (
            raw_stats - self.stats_mean
        ) / self.stats_std

        return {
            "image": image,
            "label": torch.tensor(
                label,
                dtype=torch.long
            ),
            "class_name": row["label"],
            "stats": stats
        }