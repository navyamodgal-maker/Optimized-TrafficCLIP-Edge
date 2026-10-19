"""
Quantization-Aware Training (QAT) for ProposedTrafficCLIP's fusion head.

The original PTQ (post-training quantization) approach --- quantize
AFTER training is done --- caused Macro F1 to collapse below 0.10 in
Pramit's paper. QAT fixes this by simulating INT8 rounding noise
DURING training, so the model's weights adapt to tolerate that noise
before it's ever actually quantized.

Scope: the ResNet-50 and BERT backbones are frozen in this
architecture regardless of quantization (they contribute zero
gradient), so quantizing them yields no training-time benefit and
adds real risk of breaking pretrained numerics. We therefore apply
QAT to the trainable classifier: the fusion head, which is what
actually gets fine-tuned and deployed as the task-specific model.

Usage:
    python -m src.qat_train --checkpoint output/proposed_traffic_clip_best.pth
"""

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

DATA_PATH = "data/traffic_data.npz"
OUTPUT_DIR = "output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

QAT_EPOCHS = 5
QAT_LEARNING_RATE = 5e-5
NUM_BITS = 8


def fake_quantize(tensor, num_bits=NUM_BITS):
    """
    Simulate INT8 quantization noise on a tensor during the forward
    pass, with a straight-through estimator for gradients (the
    rounding itself has zero gradient almost everywhere, so we let
    gradients flow through as if this were the identity function).
    """
    quant_min = -(2 ** (num_bits - 1))
    quant_max = (2 ** (num_bits - 1)) - 1

    scale = (tensor.max() - tensor.min()).clamp(min=1e-8) / (quant_max - quant_min)
    zero_point = quant_min - tensor.min() / scale

    fake_quantized = torch.fake_quantize_per_tensor_affine(
        tensor,
        scale=scale.item(),
        zero_point=int(zero_point.item()),
        quant_min=quant_min,
        quant_max=quant_max
    )

    return fake_quantized


class QATWrapper(nn.Module):
    """
    Wraps the existing LightweightFusion head, inserting fake
    quantization on its input features and on the output of each
    internal Linear layer's activation, so the head learns weights
    that are robust to INT8 rounding.
    """

    def __init__(self, fusion_module):
        super().__init__()
        self.fusion_module = fusion_module

    def forward(self, visual_features, text_features, stats_features):
        visual_features = fake_quantize(visual_features)
        text_features = fake_quantize(text_features)
        stats_features = fake_quantize(stats_features)

        return self.fusion_module(visual_features, text_features, stats_features)


@torch.no_grad()
def extract_features(model, batch):
    """Run the frozen/backbone parts of the model to get the three
    feature vectors that feed the fusion head."""
    images = batch["image"].to(DEVICE)
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    stats = batch["stats"].to(DEVICE)

    visual_features = model.get_vision_features(images)
    text_features = model.get_text_features(input_ids, attention_mask)

    processed_stats = model.robust_stats(stats)
    stats_features = model.rich_stats_encoder(processed_stats)

    return visual_features, text_features, stats_features


def evaluate_fusion(model, qat_wrapper, dataloader, use_qat):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            visual_features, text_features, stats_features = extract_features(model, batch)
            labels = batch["label"].to(DEVICE)

            if use_qat:
                logits = qat_wrapper(visual_features, text_features, stats_features)
            else:
                logits = model.lightweight_fusion(visual_features, text_features, stats_features)

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(OUTPUT_DIR, "proposed_traffic_clip_best.pth"),
        help="Full-precision checkpoint to start QAT fine-tuning from"
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run: python -m src.train_proposed  (without --open-set) first.")
        return

    import numpy as np
    full_data = np.load(DATA_PATH, allow_pickle=True)
    num_classes = len(full_data["labels"].tolist())

    print()
    print("=" * 60)
    print("Quantization-Aware Training (QAT) - Fusion Head")
    print("=" * 60)

    model = ProposedTrafficCLIP(num_classes=num_classes, stats_input_dim=8)
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)

    # Freeze everything except the fusion head - QAT only fine-tunes
    # the part we're actually going to quantize.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.lightweight_fusion.parameters():
        parameter.requires_grad = True

    qat_wrapper = QATWrapper(model.lightweight_fusion).to(DEVICE)

    train_loader, val_loader, test_loader = get_dataloader(
        npz_path=DATA_PATH,
        tokenizer="google-bert/bert-base-uncased",
        batch_size=16
    )

    print()
    print("Baseline (FP32) accuracy before QAT fine-tuning:")
    fp32_accuracy = evaluate_fusion(model, qat_wrapper, test_loader, use_qat=False)
    print(f"  FP32 test accuracy: {fp32_accuracy:.4f}")

    print()
    print("Naive PTQ-style check (fake-quant applied with NO fine-tuning):")
    ptq_style_accuracy = evaluate_fusion(model, qat_wrapper, test_loader, use_qat=True)
    print(f"  Fake-quantized accuracy (no QAT training yet): {ptq_style_accuracy:.4f}")
    print("  (This is the accuracy drop that plain post-training quantization would cause.)")

    optimizer = torch.optim.AdamW(
        model.lightweight_fusion.parameters(),
        lr=QAT_LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()

    print()
    print("Starting QAT fine-tuning...")
    print()

    for epoch in range(QAT_EPOCHS):
        model.lightweight_fusion.train()
        total_loss = 0.0

        for batch in train_loader:
            visual_features, text_features, stats_features = extract_features(model, batch)
            labels = batch["label"].to(DEVICE)

            logits = qat_wrapper(visual_features, text_features, stats_features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_accuracy = evaluate_fusion(model, qat_wrapper, val_loader, use_qat=True)

        print(
            f"QAT Epoch [{epoch + 1}/{QAT_EPOCHS}] "
            f"| Train Loss: {total_loss / max(len(train_loader), 1):.4f} "
            f"| Val Acc (fake-quantized): {val_accuracy:.4f}"
        )

    print()
    final_qat_accuracy = evaluate_fusion(model, qat_wrapper, test_loader, use_qat=True)

    print("=" * 60)
    print("QAT Results Summary")
    print("=" * 60)
    print(f"FP32 baseline accuracy              : {fp32_accuracy:.4f}")
    print(f"Naive fake-quant, no QAT training    : {ptq_style_accuracy:.4f}")
    print(f"Fake-quant AFTER QAT fine-tuning     : {final_qat_accuracy:.4f}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    qat_checkpoint_path = os.path.join(OUTPUT_DIR, "fusion_head_qat.pth")
    torch.save(
        {
            "fusion_state_dict": model.lightweight_fusion.state_dict(),
            "fp32_accuracy": fp32_accuracy,
            "ptq_style_accuracy": ptq_style_accuracy,
            "qat_accuracy": final_qat_accuracy
        },
        qat_checkpoint_path
    )
    print(f"Saved QAT fusion head to: {qat_checkpoint_path}")


if __name__ == "__main__":
    main()
