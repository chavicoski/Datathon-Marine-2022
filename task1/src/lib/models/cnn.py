import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 2,
        pool_size: int = 2,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size),
        )

    def forward(self, x):
        return self.conv(x)


class CNNModel(nn.Module):
    def __init__(self, n_conv_blocks: int = 4, n_classes: int = 10):
        super().__init__()
        conv_blocks = [ConvBlock(in_channels=1, out_channels=32)]
        in_channels = 32
        for _ in range(n_conv_blocks - 1):
            out_channels = in_channels * 2
            conv_blocks.append(
                ConvBlock(in_channels=in_channels, out_channels=out_channels)
            )
            in_channels = out_channels

        self.conv = nn.Sequential(
            *conv_blocks,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels, n_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        preds = self.linear(x)
        return preds
