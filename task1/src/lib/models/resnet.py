"""Implementation of a Pytorch Module to build different ResNet versions"""
import torch
import torch.nn as nn
from torchvision.models import (resnet18, resnet34, resnet50, resnet101,
                                resnet152)


class ResNetModel(nn.Module):
    def __init__(
        self,
        resnet_version: int = 18,
        pretrained: bool = True,
        reduction_factor: int = 2,
        n_classes: int = 10,
        in_channels: int = 3,
    ):
        """
        Download the ResNet model and prepare it for the new task changing
        the Fully-Connected layers

        Args:

            resnet_version (int): String of the ResNet version number to use.
                                  Options: `18`, `34`, `50`, `101`, `152`
            pretrained (bool): To use Imagenet pretrained weights or not
            reduction_factor (int): Divisor to reduce the number of features
                                    coming from the feature extractor
            n_classes (int): Number of units in the output layer
            in_channels (int): Number of channels of the input tensor. Used to
                               replicate the channels (if needed, up to 3)
        """
        super().__init__()
        self.pretrained = pretrained
        # Pre-compute the number of channels to add as zero padding
        self.n_pad_channels = 3 - in_channels
        if self.n_pad_channels < 0:
            raise ValueError("The number on 'in_channels' can't be more than 3")

        # Use "DEFAULT" pretrained weights if enabled
        weights = "DEFAULT" if self.pretrained else None

        if resnet_version == 18:
            base_resnet = resnet18(weights=weights)
        elif resnet_version == 34:
            base_resnet = resnet34(weights=weights)
        elif resnet_version == 50:
            base_resnet = resnet50(weights=weights)
        elif resnet_version == 101:
            base_resnet = resnet101(weights=weights)
        elif resnet_version == 152:
            base_resnet = resnet152(weights=weights)
        else:
            raise Exception(f"Invalid ResNet version ('{resnet_version}')")

        # Take only the convolutional block (remove fully-connected layers)
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-1])

        # Add the channel padding if needed
        # Note: As we are using the original resnet model, we need to input
        #       a tensor with 3 channels
        if self.n_pad_channels:
            # padding: (left, right, top, bottom, front, back)
            padding = (0, 0, 0, 0, 0, self.n_pad_channels)
            self.backbone = nn.Sequential(
                nn.ConstantPad3d(padding, 0),
                self.backbone,
            )

        # Replace the old Fully-Connected layers
        n_features = base_resnet.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(n_features, n_features // reduction_factor),
            nn.ReLU(),
            nn.Linear(n_features // reduction_factor, n_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        return self.classifier(features)
