import torch
import torch.nn as nn


class TrafficAdapter(nn.Module):

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

        adapted = self.relu(
            adapted
        )

        adapted = self.up(
            adapted
        )

        original = self.original_projection(
            x
        )

        output = (
            self.alpha * adapted
            +
            (1 - self.alpha) * original
        )

        return output