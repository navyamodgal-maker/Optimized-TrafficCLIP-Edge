import torch
import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)


class SemanticEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = resnet50(
            weights=ResNet50_Weights.DEFAULT
        )

        old_conv = self.model.conv1

        new_conv = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        with torch.no_grad():

            new_conv.weight[:] = (
                old_conv.weight.mean(
                    dim=1,
                    keepdim=True
                )
            )

        self.model.conv1 = new_conv

        self.model.fc = nn.Identity()

        # Freeze ResNet-50 backbone
        for param in self.model.parameters():

            param.requires_grad = False

    def forward(self, x):

        return self.model(x)