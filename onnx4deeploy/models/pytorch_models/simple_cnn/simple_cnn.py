# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Simple CNN for image classification - Deploy Version (no MaxPool, strided convolutions)."""

import torch.nn as nn
import torch.nn.init as init


class SimpleCNN(nn.Module):
    """
    Simple CNN for image classification (Deploy Version).

    Uses strided convolutions instead of MaxPool to simplify the backward pass
    and keep the ONNX graph free of optional-output operators.

    Architecture (default 16×16 grayscale input):
    - Input:  (B, 1, 16, 16)
    - Conv1:  1  →  8 ch, k=3×3, stride=2, pad=1 → (B,  8, 8, 8), ReLU
    - Conv2:  8  → 16 ch, k=3×3, stride=2, pad=1 → (B, 16, 4, 4), ReLU
    - Flatten:                                     → (B, 256)
    - FC:     256 → num_classes
    - Output: logits (B, num_classes)

    Total trainable parameters (default config): ~3.8 K (≈15 KB float32).

    Args:
        input_channels: Number of input channels (default 1 = grayscale).
        input_height:   Spatial height of the input image (default 16).
        input_width:    Spatial width  of the input image (default 16).
        hidden_channels: Number of channels produced by Conv2 (default 16).
        num_classes:    Number of output classes (default 10).
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_height: int = 16,
        input_width: int = 16,
        hidden_channels: int = 16,
        num_classes: int = 10,
    ):
        super(SimpleCNN, self).__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes

        mid_channels = hidden_channels // 2  # 8 by default

        # Conv block 1: stride-2 strided conv replaces Conv + MaxPool
        # (B, in_ch, H, W) → (B, mid_ch, H/2, W/2)
        self.conv1 = nn.Conv2d(input_channels, mid_channels, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()

        # Conv block 2: stride-2 strided conv
        # (B, mid_ch, H/2, W/2) → (B, hidden_ch, H/4, W/4)
        self.conv2 = nn.Conv2d(mid_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()

        # Flatten then FC
        fc_in = hidden_channels * (input_height // 4) * (input_width // 4)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(fc_in, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x
