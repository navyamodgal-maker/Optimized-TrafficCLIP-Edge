import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.models.traffic_clip import TrafficCLIP
from src.losses.cross_entropy import TrafficCrossEntropy
from src.losses.contrastive import SupConLoss
from src.training.scheduler import get_scheduler
from src.utils.seed import set_seed


class Trainer:

    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        class_names,
        batch_size=32,
        lr=0.002,
        epochs=20,
        device=None
    ):

        set_seed(42)

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.class_names = class_names
        self.epochs = epochs

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        print(f"Device: {self.device}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Classes: {class_names}")

        self.model = TrafficCLIP(
            embed_dim=1024
        ).to(self.device)

        self.ce_loss = TrafficCrossEntropy()
        self.cl_loss = SupConLoss()

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )

        self.scheduler = get_scheduler(
            self.optimizer,
            total_epochs=epochs,
            base_lr=lr
        )

        self.text_prompts = [
            f"a network traffic photo of {c}"
            for c in class_names
        ]

        os.makedirs(
            "outputs/checkpoints",
            exist_ok=True
        )

    def calculate_loss(self, batch):

        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)

        logits_per_image, _ = self.model(
            images,
            self.text_prompts
        )

        loss_ce = self.ce_loss(
            logits_per_image,
            labels
        )

        detail_feat = self.model.detail_encoder(images)

        semantic_feat = self.model.adapter(
            self.model.semantic_encoder(images)
        )

        visual_feat = self.model.fusion(
            detail_feat,
            semantic_feat
        )

        loss_cl = self.cl_loss(
            visual_feat,
            labels
        )

        loss = loss_ce + loss_cl

        return loss, logits_per_image, labels

    def train_one_epoch(self, epoch):

        self.model.train()

        total_loss = 0.0

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs} [TRAIN]",
            unit="batch"
        )

        for batch in progress:

            loss, _, _ = self.calculate_loss(batch)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        return total_loss / len(self.train_loader)

    def validate(self, epoch):

        self.model.eval()

        total_loss = 0.0

        progress = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch}/{self.epochs} [VAL]",
            unit="batch"
        )

        with torch.no_grad():

            for batch in progress:

                loss, _, _ = self.calculate_loss(batch)

                total_loss += loss.item()

                progress.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

        return total_loss / len(self.val_loader)

    def test(self):

        self.model.eval()

        predictions = []
        targets = []

        print("\nTesting model...")

        with torch.no_grad():

            for batch in tqdm(
                self.test_loader,
                desc="Testing",
                unit="batch"
            ):

                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                logits_per_image, _ = self.model(
                    images,
                    self.text_prompts
                )

                preds = torch.argmax(
                    logits_per_image,
                    dim=1
                )

                predictions.extend(
                    preds.cpu().numpy()
                )

                targets.extend(
                    labels.cpu().numpy()
                )

        accuracy = accuracy_score(
            targets,
            predictions
        )

        macro_f1 = f1_score(
            targets,
            predictions,
            average="macro"
        )

        return accuracy, macro_f1

    def save_checkpoint(self, epoch):

        checkpoint_path = (
            f"outputs/checkpoints/"
            f"epoch_{epoch}.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path
        )

        print(
            f"Checkpoint saved: {checkpoint_path}"
        )

    def train(self):

        train_losses = []
        val_losses = []

        for epoch in range(1, self.epochs + 1):

            print()
            print("=" * 50)
            print(f"Epoch {epoch}/{self.epochs}")
            print("=" * 50)

            train_loss = self.train_one_epoch(
                epoch
            )

            val_loss = self.validate(
                epoch
            )

            self.scheduler.step()

            train_losses.append(
                train_loss
            )

            val_losses.append(
                val_loss
            )

            print()
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")

            self.save_checkpoint(
                epoch
            )

        accuracy, macro_f1 = self.test()

        print()
        print("=" * 50)
        print("FINAL TEST RESULTS")
        print("=" * 50)

        print(
            f"Test Accuracy: {accuracy * 100:.2f}%"
        )

        print(
            f"Macro F1:      {macro_f1 * 100:.2f}%"
        )

        return (
            train_losses,
            val_losses,
            accuracy,
            macro_f1
        )
        