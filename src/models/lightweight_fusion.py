import torch
import torch.nn as nn


class LightweightFusion(nn.Module):
    """
    Lightweight multimodal fusion module.

    Visual features:
        1024 -> 256

    Text features:
        1024 -> 256

    Rich statistics:
        128 -> 128

    Concatenation:
        256 + 256 + 128 = 640

    Final fusion:
        640 -> 256

    Classification:
        256 -> number of classes
    """

    def __init__(
        self,
        visual_dim=1024,
        text_dim=1024,
        stats_dim=128,
        projected_dim=256,
        fusion_dim=256,
        num_classes=10,
        dropout=0.3
    ):
        super().__init__()

        # --------------------------------------------------
        # Visual projection
        # --------------------------------------------------

        self.visual_projection = nn.Sequential(

            nn.Linear(
                visual_dim,
                projected_dim
            ),

            nn.LayerNorm(
                projected_dim
            ),

            nn.GELU()
        )

        # --------------------------------------------------
        # Text projection
        # --------------------------------------------------

        self.text_projection = nn.Sequential(

            nn.Linear(
                text_dim,
                projected_dim
            ),

            nn.LayerNorm(
                projected_dim
            ),

            nn.GELU()
        )

        # --------------------------------------------------
        # Statistics projection
        # --------------------------------------------------

        self.stats_projection = nn.Sequential(

            nn.Linear(
                stats_dim,
                128
            ),

            nn.LayerNorm(
                128
            ),

            nn.GELU()
        )

        # --------------------------------------------------
        # Fusion
        # --------------------------------------------------

        fusion_input_dim = (
            projected_dim
            + projected_dim
            + 128
        )

        self.fusion = nn.Sequential(

            nn.Linear(
                fusion_input_dim,
                fusion_dim
            ),

            nn.LayerNorm(
                fusion_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            )
        )

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        self.classifier = nn.Linear(
            fusion_dim,
            num_classes
        )

    def forward(
        self,
        visual_features,
        text_features,
        stats_features
    ):

        visual = self.visual_projection(
            visual_features
        )

        text = self.text_projection(
            text_features
        )

        stats = self.stats_projection(
            stats_features
        )

        combined = torch.cat(
            [
                visual,
                text,
                stats
            ],
            dim=1
        )

        fused = self.fusion(
            combined
        )

        logits = self.classifier(
            fused
        )

        return logits


if __name__ == "__main__":

    model = LightweightFusion(
        num_classes=10
    )

    visual = torch.randn(
        4,
        1024
    )

    text = torch.randn(
        4,
        1024
    )

    stats = torch.randn(
        4,
        128
    )

    output = model(
        visual,
        text,
        stats
    )

    print("Visual shape :", visual.shape)
    print("Text shape   :", text.shape)
    print("Stats shape  :", stats.shape)
    print("Output shape :", output.shape)