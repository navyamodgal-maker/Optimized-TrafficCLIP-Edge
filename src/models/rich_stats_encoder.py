import torch
import torch.nn as nn


class RichStatisticsEncoder(nn.Module):
    """
    Encodes 8 traffic statistics into a compact
    behavioral representation.

    Input:
        [Batch, 8]

    Output:
        [Batch, 128]
    """

    def __init__(
        self,
        input_dim=8,
        hidden_dim=64,
        output_dim=128,
        dropout=0.2
    ):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                output_dim
            ),

            nn.LayerNorm(
                output_dim
            ),

            nn.GELU()
        )

    def forward(self, x):
        return self.encoder(x)


if __name__ == "__main__":

    model = RichStatisticsEncoder()

    sample = torch.randn(
        4,
        8
    )

    output = model(sample)

    print("Input shape :", sample.shape)
    print("Output shape:", output.shape)