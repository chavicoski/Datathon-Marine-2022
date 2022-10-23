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
        drop_factor: float = 0.2,
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
            nn.Dropout(drop_factor),
            nn.MaxPool2d(kernel_size=pool_size),
        )

    def forward(self, x):
        return self.conv(x)


class RecurrentCNNModel(nn.Module):
    """Recurrent Convolutionan Neural Network

    Uses first a set of convolutional layers to extract features. Then uses
    a recurrent part to process the sequence of features. And finally predicts
    the class for each frame using a fully-connected network. Note that the frames
    dimension is always maintained, from input (batch, channels, freq_bins, frames)
    to ouputut (batch, n_classes, frames)
    """

    def __init__(
        self,
        n_conv_blocks: int = 3,
        n_classes: int = 10,
        input_height: int = 128,  # 128 mel bands
        start_n_filters: int = 128,
        filters_factor: int = 1,
        lstm_h_size: int = 64,
        pool_factor: int = 4,
        drop_factor: float = 0.2,
    ):
        super().__init__()
        # Create the first conv block for the 1 channel input
        conv_blocks = [
            ConvBlock(
                in_channels=1,  # Mono audio
                out_channels=start_n_filters,
                pool_size=(pool_factor, 1),
                drop_factor=drop_factor,
            ),
        ]

        in_channels = out_channels
        out_channels = start_n_filters
        out_height = int(input_height / pool_factor)  # For LSTM input shape
        for _ in range(n_conv_blocks - 1):
            out_channels = in_channels * filters_factor
            conv_blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    pool_size=(pool_factor, 1),
                    drop_factor=drop_factor,
                )
            )
            in_channels = out_channels
            out_height = int(out_height / pool_factor)

        self.conv = nn.Sequential(*conv_blocks)

        self.recurrent = nn.LSTM(
            input_size=out_height * out_channels,  # We stack all the channels
            hidden_size=lstm_h_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=drop_factor,
        )

        self.linear = nn.Sequential(
            # lstm_h_size * 2 because the LSTM is bidirectional
            nn.Linear(lstm_h_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(drop_factor),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        # x: (batch, channels=1, freq_bins, frames)
        x = self.conv(x)
        # x: (batch, channels, new_freq_bins, frames)
        # Stack the channels for the LSTM
        x = torch.cat([x[:, i, :, :] for i in range(x.size(1))], dim=1)
        # x: (batch, channels * new_freq_bins, frames)
        x = torch.permute(x, (0, 2, 1))
        # x: (batch, frames, channels * new_freq_bins)
        x, _ = self.recurrent(x)
        # x: (batch, frames, lstm_hidden_size * 2)
        preds = self.linear(x)
        # preds: (batch, frames, n_classes)
        preds = torch.permute(preds, (0, 2, 1))
        # preds: (batch, n_classes, frames)
        return preds
