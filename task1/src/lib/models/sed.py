import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_size: int = (2, 1),
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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size),
        )

    def forward(self, x):
        return self.conv(x)


class RecurrentCNNModel(nn.Module):
    def __init__(
        self,
        n_conv_blocks: int = 5,
        n_classes: int = 10,
        input_height: int = 128,  # 128 mel bands
        start_n_filters: int = 128,
        filters_factor: int = 1,
        lstm_h_size: int = 32,
    ):
        super().__init__()
        # Start with a bigger pool_size (4, 1) (in the frequency axis)
        out_channels = start_n_filters
        conv_blocks = [
            ConvBlock(in_channels=1, out_channels=out_channels, pool_size=(4, 1)),
        ]
        in_channels = out_channels
        out_height = int(input_height / 4)

        # Add the following blocks with a smaller pool_size (2, 1)
        for _ in range(n_conv_blocks - 1):
            out_channels = in_channels * filters_factor
            conv_blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    pool_size=(2, 1),
                )
            )
            in_channels = out_channels
            out_height = int(out_height / 2)

        self.conv = nn.Sequential(*conv_blocks)

        self.recurrent = nn.LSTM(
            input_size=out_height * out_channels,
            hidden_size=lstm_h_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.linear = nn.Sequential(
            nn.Linear(lstm_h_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        # Stack the channels comming from the conv block
        # From (batch, channels, freq, frames) to (batch, channels * freq, frames)
        x = torch.cat([x[:, i, :, :] for i in range(x.size(1))], dim=1)
        # From (batch, channels * freq, frames) to (batch, frames, channels * freq)
        x = torch.permute(x, (0, 2, 1))
        x, _ = self.recurrent(x)
        preds = self.linear(x)
        # From (batch, frames, labels) to (batch, labels, frames)
        preds = torch.permute(preds, (0, 2, 1))
        return preds
