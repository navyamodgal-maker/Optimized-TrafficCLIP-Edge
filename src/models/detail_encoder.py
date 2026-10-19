import torch
import torch.nn as nn


class BottleneckBlock(nn.Module):

    def __init__(self, channels=64):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1
            ),

            nn.BatchNorm2d(channels),

            nn.ReLU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(channels),

            nn.ReLU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1
            ),

            nn.BatchNorm2d(channels)
        )

        self.relu = nn.ReLU()

    def forward(self, x):

        identity = x

        x = self.block(x)

        x = x + identity

        return self.relu(x)


class DetailAwareEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Conv2d(
            1,
            64,
            kernel_size=3,
            padding=1
        )

        self.bn = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()

        self.bottleneck = BottleneckBlock(
            64
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.projection = nn.Linear(
            64,
            512
        )

    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        x = self.relu(x)

        x = self.bottleneck(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            1
        )

        x = self.projection(x)

        return x