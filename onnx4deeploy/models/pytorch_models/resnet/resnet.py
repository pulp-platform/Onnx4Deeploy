# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""ResNet Models for ONNX Export.

Based on the original ResNet paper:
"Deep Residual Learning for Image Recognition" - He et al. (2015)

This implementation is optimized for ONNX export with clean computation graphs.
"""

from typing import List, Type, Union

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18 and ResNet-34.

    Structure:
    - 3x3 conv -> BN -> ReLU
    - 3x3 conv -> BN
    - Add residual connection
    - ReLU
    """

    expansion = 1

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None
    ):
        """
        Initialize BasicBlock.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for first convolution
            downsample: Optional downsampling layer for residual connection
        """
        super(BasicBlock, self).__init__()

        # First convolution block
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)  # inplace=False for clean ONNX export

        # Second convolution block
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsampling for residual connection (if needed)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck residual block for ResNet-50, ResNet-101, and ResNet-152.

    Structure:
    - 1x1 conv -> BN -> ReLU (reduce dimensions)
    - 3x3 conv -> BN -> ReLU
    - 1x1 conv -> BN (expand dimensions)
    - Add residual connection
    - ReLU
    """

    expansion = 4

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None
    ):
        """
        Initialize Bottleneck block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of intermediate channels
            stride: Stride for 3x3 convolution
            downsample: Optional downsampling layer for residual connection
        """
        super(Bottleneck, self).__init__()

        # 1x1 conv (dimension reduction)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 conv
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv (dimension expansion)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=False)  # inplace=False for clean ONNX export
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = x

        # 1x1 conv (reduce)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3 conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1 conv (expand)
        out = self.conv3(out)
        out = self.bn3(out)

        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet architecture.

    Standard ResNet for ImageNet-style classification.
    Input: (N, 3, H, W) - typically H=W=224
    Output: (N, num_classes)
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        input_channels: int = 3,
    ):
        """
        Initialize ResNet.

        Args:
            block: Type of residual block (BasicBlock or Bottleneck)
            layers: Number of blocks in each layer (e.g., [2, 2, 2, 2] for ResNet-18)
            num_classes: Number of output classes
            input_channels: Number of input channels (3 for RGB)
        """
        super(ResNet, self).__init__()

        self.in_channels = 64

        # Initial convolution layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """
        Create a residual layer with multiple blocks.

        Args:
            block: Type of block to use
            out_channels: Number of output channels
            blocks: Number of blocks in this layer
            stride: Stride for first block (for downsampling)

        Returns:
            Sequential module containing all blocks
        """
        downsample = None

        # Create downsampling layer if needed
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # First block (may have downsampling)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet."""
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling and classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """
    ResNet-18 model.

    Architecture: [2, 2, 2, 2] BasicBlocks
    Parameters: ~11.7M

    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels

    Returns:
        ResNet-18 model
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, input_channels)


def resnet34(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """
    ResNet-34 model.

    Architecture: [3, 4, 6, 3] BasicBlocks
    Parameters: ~21.8M

    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels

    Returns:
        ResNet-34 model
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, input_channels)


def resnet50(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """
    ResNet-50 model (MLPerf benchmark model).

    Architecture: [3, 4, 6, 3] Bottleneck blocks
    Parameters: ~25.6M

    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels

    Returns:
        ResNet-50 model
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, input_channels)
