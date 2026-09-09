import logging

import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel

from src.models.robust_stats import RobustStatisticsProcessor
from src.models.rich_stats_encoder import RichStatisticsEncoder
from src.models.lightweight_fusion import LightweightFusion
from src.open_set.detector import OpenSetDetector


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)


# ============================================================
# DETAIL ENCODER
# ============================================================

class DetailAwareEncoder(nn.Module):
    """
    Lightweight CNN for extracting traffic-image details.

    Input:
        [B, 1, H, W]

    Output:
        [B, 512]
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.projection = nn.Linear(256, 512)

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, 1)

        x = self.projection(x)

        return x


# ============================================================
# SEMANTIC ENCODER
# ============================================================

class SemanticEncoder(nn.Module):
    """
    ResNet-50 semantic encoder.

    Original RGB input is changed to one-channel input
    because traffic images are grayscale.

    Output:
        [B, 2048]
    """

    def __init__(self, pretrained=True):

        super().__init__()

        try:

            if pretrained:
                weights = models.ResNet50_Weights.DEFAULT
                backbone = models.resnet50(weights=weights)
            else:
                backbone = models.resnet50(weights=None)

        except Exception as error:

            logging.warning(
                "Could not load pretrained ResNet-50 weights."
            )

            logging.warning(
                f"Reason: {error}"
            )

            logging.warning(
                "Using ResNet-50 without pretrained weights."
            )

            backbone = models.resnet50(weights=None)

        # ----------------------------------------------------
        # Convert first convolution from 3 channels to 1
        # ----------------------------------------------------

        old_conv = backbone.conv1

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        if old_conv.weight.shape[1] == 3:

            with torch.no_grad():

                new_conv.weight[:] = (
                    old_conv.weight.mean(dim=1, keepdim=True)
                )

        backbone.conv1 = new_conv

        # Remove classification layer
        backbone.fc = nn.Identity()

        self.backbone = backbone

        # Freeze semantic encoder
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def forward(self, x):

        with torch.no_grad():

            features = self.backbone(x)

        return features


# ============================================================
# TRAFFIC ADAPTER
# ============================================================

class TrafficAdapter(nn.Module):
    """
    Adapts ResNet semantic features:

        2048 -> 512
    """

    def __init__(
        self,
        input_dim=2048,
        adapter_dim=256,
        output_dim=512,
        alpha=0.9
    ):

        super().__init__()

        self.down = nn.Linear(
            input_dim,
            adapter_dim
        )

        self.relu = nn.ReLU()

        self.up = nn.Linear(
            adapter_dim,
            output_dim
        )

        self.original_projection = nn.Linear(
            input_dim,
            output_dim
        )

        self.alpha = alpha

    def forward(self, x):

        adapted = self.down(x)

        adapted = self.relu(adapted)

        adapted = self.up(adapted)

        original = self.original_projection(x)

        output = (
            self.alpha * adapted
            +
            (1.0 - self.alpha) * original
        )

        return output


# ============================================================
# TEXT ENCODER
# ============================================================

class TextEncoder(nn.Module):
    """
    BERT text encoder.

    BERT:
        768 dimensions

    Projection:
        768 -> 1024
    """

    def __init__(
        self,
        output_dim=1024
    ):

        super().__init__()

        try:

            self.bert = BertModel.from_pretrained(
                "bert-base-uncased"
            )

        except Exception as error:

            logging.warning(
                "Could not load pretrained BERT weights."
            )

            logging.warning(
                f"Reason: {error}"
            )

            logging.warning(
                "Creating BERT model from configuration."
            )

            from transformers import BertConfig

            config = BertConfig()

            self.bert = BertModel(config)

        # Freeze BERT
        for parameter in self.bert.parameters():
            parameter.requires_grad = False

        self.projection = nn.Linear(
            768,
            output_dim
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        with torch.no_grad():

            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            cls_embedding = (
                outputs.last_hidden_state[:, 0, :]
            )

        features = self.projection(
            cls_embedding
        )

        return features


# ============================================================
# PROPOSED TRAFFIC CLIP
# ============================================================

class ProposedTrafficCLIP(nn.Module):
    """
    Proposed TrafficCLIP architecture.

    Main contributions:

    1. Eight traffic statistics
    2. Robust statistics processing
    3. Rich statistics encoder
    4. Visual + text + statistics fusion
    5. Lightweight fusion network
    6. Open-set unknown detection

    Architecture:

        IMAGE
          |
          +--> Detail Encoder ------+
          |                         |
          +--> ResNet Semantic -----+--> Visual 1024
                                    |
        TEXT --> BERT 768 --> 1024 --+
                                    |
        8 STATISTICS
          |
          +--> Robust Processor
          |
          +--> Rich Statistics Encoder
          |
          +--> Stats 128
                                    |
                                    v
                         Lightweight Fusion
                                    |
                                    v
                              Classifier
                                    |
                                    v
                            Known / UNKNOWN
    """

    def __init__(
        self,
        num_classes,
        stats_input_dim=8,
        visual_dim=1024,
        text_dim=1024,
        stats_dim=128,
        fusion_dim=256
    ):

        super().__init__()

        self.num_classes = num_classes

        # ----------------------------------------------------
        # Visual encoders
        # ----------------------------------------------------

        self.detail_encoder = DetailAwareEncoder()

        self.semantic_encoder = SemanticEncoder(
            pretrained=True
        )

        self.adapter = TrafficAdapter(
            input_dim=2048,
            adapter_dim=256,
            output_dim=512
        )

        # ----------------------------------------------------
        # Text encoder
        # ----------------------------------------------------

        self.text_encoder = TextEncoder(
            output_dim=text_dim
        )

        # ----------------------------------------------------
        # Robust statistics processor
        # ----------------------------------------------------

        self.robust_stats = RobustStatisticsProcessor(
            num_features=stats_input_dim
        )

        # ----------------------------------------------------
        # Rich statistics encoder
        # ----------------------------------------------------

        self.rich_stats_encoder = RichStatisticsEncoder(
            input_dim=stats_input_dim,
            hidden_dim=64,
            output_dim=stats_dim,
            dropout=0.2
        )

        # ----------------------------------------------------
        # Lightweight fusion
        # ----------------------------------------------------

        self.lightweight_fusion = LightweightFusion(
            visual_dim=visual_dim,
            text_dim=text_dim,
            stats_dim=stats_dim,
            projected_dim=256,
            fusion_dim=fusion_dim,
            num_classes=num_classes,
            dropout=0.3
        )

        # ----------------------------------------------------
        # Open-set detector
        # ----------------------------------------------------

        self.open_set_detector = OpenSetDetector(
            confidence_threshold=0.60,
            entropy_threshold=1.50
        )

        logging.info(
            "ProposedTrafficCLIP initialized successfully."
        )

        logging.info(
            f"Number of classes: {num_classes}"
        )

        logging.info(
            f"Statistics features: {stats_input_dim}"
        )


    # ========================================================
    # VISUAL FEATURES
    # ========================================================

    def get_vision_features(self, images):

        # Detail features
        detail_features = self.detail_encoder(
            images
        )

        # Semantic features
        semantic_features = self.semantic_encoder(
            images
        )

        # Adapt 2048 -> 512
        semantic_features = self.adapter(
            semantic_features
        )

        # 512 + 512 = 1024
        visual_features = torch.cat(
            [
                detail_features,
                semantic_features
            ],
            dim=1
        )

        return visual_features


    # ========================================================
    # TEXT FEATURES
    # ========================================================

    def get_text_features(
        self,
        input_ids,
        attention_mask
    ):

        text_features = self.text_encoder(
            input_ids,
            attention_mask
        )

        return text_features


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        images,
        input_ids,
        attention_mask,
        stats_vector
    ):

        # ----------------------------------------------------
        # Check statistics
        # ----------------------------------------------------

        if stats_vector is None:

            raise ValueError(
                "stats_vector cannot be None. "
                "The proposed model requires "
                "8 statistical features."
            )

        if stats_vector.dim() != 2:

            raise ValueError(
                "stats_vector must have shape "
                "[batch_size, 8]."
            )

        if stats_vector.size(1) != 8:

            raise ValueError(
                f"Expected 8 statistics, "
                f"but received "
                f"{stats_vector.size(1)}."
            )

        # ----------------------------------------------------
        # 1. Visual features
        # ----------------------------------------------------

        visual_features = self.get_vision_features(
            images
        )

        # ----------------------------------------------------
        # 2. Text features
        # ----------------------------------------------------

        text_features = self.get_text_features(
            input_ids,
            attention_mask
        )

        # ----------------------------------------------------
        # 3. Robust statistics
        # ----------------------------------------------------

        processed_stats = self.robust_stats(
            stats_vector
        )

        # ----------------------------------------------------
        # 4. Rich statistics
        # ----------------------------------------------------

        stats_features = self.rich_stats_encoder(
            processed_stats
        )

        # ----------------------------------------------------
        # 5. Multimodal fusion
        # ----------------------------------------------------

        logits = self.lightweight_fusion(
            visual_features,
            text_features,
            stats_features
        )

        return logits


    # ========================================================
    # OPEN-SET PREDICTION
    # ========================================================

    @torch.no_grad()
    def predict_open_set(
        self,
        images,
        input_ids,
        attention_mask,
        stats_vector
    ):

        logits = self.forward(
            images,
            input_ids,
            attention_mask,
            stats_vector
        )

        result = self.open_set_detector.predict(
            logits
        )

        result["logits"] = logits

        return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("Testing ProposedTrafficCLIP")
    print("=" * 60)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    num_classes = 10

    batch_size = 2

    image_size = 224

    sequence_length = 64

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print()
    print("Creating model...")

    model = ProposedTrafficCLIP(
        num_classes=num_classes,
        stats_input_dim=8
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        f"Total parameters     : {total_parameters:,}"
    )

    print(
        f"Trainable parameters : {trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    print()
    print("Creating test inputs...")

    images = torch.randn(
        batch_size,
        1,
        image_size,
        image_size
    )

    input_ids = torch.randint(
        low=0,
        high=30522,
        size=(
            batch_size,
            sequence_length
        )
    )

    attention_mask = torch.ones(
        batch_size,
        sequence_length,
        dtype=torch.long
    )

    stats = torch.randn(
        batch_size,
        8
    )

    # --------------------------------------------------------
    # Model evaluation
    # --------------------------------------------------------

    model.eval()

    print()
    print("Running forward pass...")

    with torch.no_grad():

        logits = model(
            images,
            input_ids,
            attention_mask,
            stats
        )

    # --------------------------------------------------------
    # Output shapes
    # --------------------------------------------------------

    print()
    print("Input image shape    :", images.shape)

    print(
        "Statistics shape     :",
        stats.shape
    )

    print(
        "Logits shape         :",
        logits.shape
    )

    # --------------------------------------------------------
    # Open-set prediction
    # --------------------------------------------------------

    print()
    print("Running open-set detection...")

    result = model.predict_open_set(
        images,
        input_ids,
        attention_mask,
        stats
    )

    print(
        "Predictions          :",
        result["prediction"]
    )

    print(
        "Confidence           :",
        result["confidence"]
    )

    print(
        "Entropy              :",
        result["entropy"]
    )

    print(
        "Known                :",
        result["known"]
    )

    print()
    print("=" * 60)
    print("ProposedTrafficCLIP test completed.")
    print("=" * 60)