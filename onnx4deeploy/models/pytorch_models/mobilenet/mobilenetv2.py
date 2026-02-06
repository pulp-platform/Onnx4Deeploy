# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MobileNetV2 Model for ONNX Export.

Based on "MobileNetV2: Inverted Residuals and Linear Bottlenecks" - Sandler et al. (2018)

This implementation is optimized for ONNX export with clean computation graphs.
MobileNetV2 is widely used in MLPerf Mobile benchmarks.
"""

import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block (MobileNetV2 building block).

    Structure:
    - 1x1 conv (expand) -> BN -> ReLU6
    - 3x3 depthwise conv -> BN -> ReLU6
    - 1x1 conv (project) -> BN
    - Skip connection (if stride=1 and input_channels == output_channels)
    """

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        """
        Initialize Inverted Residual block.

        Args:
            inp: Number of input channels
            oup: Number of output channels
            stride: Stride for depthwise convolution
            expand_ratio: Channel expansion ratio
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion (1x1 conv)
            layers.extend(
                [
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=False),  # inplace=False for clean ONNX export
                ]
            )

        # Depthwise convolution (3x3)
        layers.extend(
            [
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False),
                # Pointwise projection (1x1 conv, no activation)
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection."""
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 architecture.

    Efficient mobile network using inverted residuals and linear bottlenecks.
    Standard configuration for ImageNet classification.
    """

    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0, input_channels: int = 3):
        """
        Initialize MobileNetV2.

        Args:
            num_classes: Number of output classes
            width_mult: Width multiplier for controlling model size (0.5, 1.0, 1.5, etc.)
            input_channels: Number of input channels (3 for RGB)
        """
        super(MobileNetV2, self).__init__()

        # Building first layer
        input_channel = 32
        last_channel = 1280

        # Inverted residual block settings
        # t: expansion factor, c: output channels, n: number of blocks, s: stride
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Apply width multiplier
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))

        # First conv layer
        features = [
            nn.Conv2d(input_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=False),
        ]

        # Build inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel

        # Last conv layer
        features.extend(
            [
                nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.last_channel),
                nn.ReLU6(inplace=False),
            ]
        )

        # Make it a sequential module
        self.features = nn.Sequential(*features)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for convolutions and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MobileNetV2."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_v2(
    num_classes: int = 1000, width_mult: float = 1.0, input_channels: int = 3
) -> MobileNetV2:
    """
    MobileNetV2 model (MLPerf Mobile benchmark model).

    Args:
        num_classes: Number of output classes
        width_mult: Width multiplier (0.5, 0.75, 1.0, 1.5, 2.0)
        input_channels: Number of input channels

    Returns:
        MobileNetV2 model
    """
    return MobileNetV2(
        num_classes=num_classes, width_mult=width_mult, input_channels=input_channels
    )
