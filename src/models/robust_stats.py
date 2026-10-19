import torch
import torch.nn as nn


class RobustStatisticsProcessor(nn.Module):
    """
    Robust preprocessing module for traffic statistics.

    The proposed model uses 8 statistical traffic features:

    1. Mean IAT
    2. IAT variance
    3. Jitter
    4. Entropy
    5. Mean packet length
    6. Packet-length variance
    7. Flow duration
    8. Burstiness

    This module:
    - Handles NaN values
    - Handles infinite values
    - Clips extreme values
    - Performs robust normalization
    """

    def __init__(
        self,
        num_features=8,
        clip_value=5.0,
        eps=1e-6
    ):
        super().__init__()

        self.num_features = num_features
        self.clip_value = clip_value
        self.eps = eps

    def forward(
        self,
        x,
        median=None,
        iqr=None
    ):
        """
        Args:
            x:
                Tensor of shape [batch_size, 8]

            median:
                Optional median values for each feature.

            iqr:
                Optional interquartile range values.

        Returns:
            Processed statistics tensor.
        """

        x = x.float()

        # --------------------------------------------------
        # 1. Replace invalid values
        # --------------------------------------------------

        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # --------------------------------------------------
        # 2. Robust normalization
        # --------------------------------------------------

        if median is not None and iqr is not None:

            median = median.to(
                device=x.device,
                dtype=x.dtype
            )

            iqr = iqr.to(
                device=x.device,
                dtype=x.dtype
            )

            iqr = torch.clamp(
                iqr,
                min=self.eps
            )

            x = (x - median) / iqr

        # --------------------------------------------------
        # 3. Outlier clipping
        # --------------------------------------------------

        x = torch.clamp(
            x,
            min=-self.clip_value,
            max=self.clip_value
        )

        return x


if __name__ == "__main__":

    # Simple test
    processor = RobustStatisticsProcessor(
        num_features=8
    )

    sample = torch.tensor(
        [
            [
                10.0,
                5.0,
                2.0,
                7.5,
                500.0,
                10000.0,
                1000.0,
                0.3
            ]
        ],
        dtype=torch.float32
    )

    output = processor(sample)

    print("Input shape :", sample.shape)
    print("Output shape:", output.shape)
    print("Output      :", output)