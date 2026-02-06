"""MobileViT Model for ONNX Export.

Based on "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer" (ICLR 2022)
https://github.com/apple/ml-cvnets

This is a simplified implementation optimized for ONNX export.
"""

from typing import Tuple

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ):
        """
        Initialize Conv-BN-Act block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size
            stride: Stride
            groups: Number of groups for grouped convolution
        """
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=False)  # SiLU (Swish) activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: int = 4):
        """
        Initialize inverted residual block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride
            expand_ratio: Channel expansion ratio
        """
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(ConvBNAct(in_channels, hidden_dim, kernel_size=1))

        # Depthwise convolution
        layers.extend(
            [
                ConvBNAct(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim),
                # Pointwise projection (no activation)
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class TransformerBlock(nn.Module):
    """Transformer block for MobileViT."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0):
        """
        Initialize transformer block.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class MobileViTBlock(nn.Module):
    """MobileViT block: Local representations (Conv) + Global representations (Transformer)."""

    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        num_heads: int = 4,
        num_transformer_blocks: int = 2,
    ):
        """
        Initialize MobileViT block.

        Args:
            in_channels: Number of input channels
            transformer_dim: Transformer dimension
            num_heads: Number of attention heads
            num_transformer_blocks: Number of transformer blocks
        """
        super().__init__()
        self.transformer_dim = transformer_dim

        # Local representation
        self.local_rep = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            ConvBNAct(in_channels, transformer_dim, kernel_size=1),
        )

        # Global representation (Transformer)
        self.global_rep = nn.Sequential(
            *[
                TransformerBlock(transformer_dim, num_heads=num_heads)
                for _ in range(num_transformer_blocks)
            ]
        )

        # Fusion
        self.fusion = nn.Sequential(
            ConvBNAct(transformer_dim, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Local representation
        y = self.local_rep(x)  # (B, transformer_dim, H, W)

        # Unfold patches for transformer
        B, C, H, W = y.shape
        y = y.flatten(2).transpose(1, 2)  # (B, H*W, transformer_dim)

        # Global representation (Transformer)
        y = self.global_rep(y)

        # Fold back to spatial
        y = y.transpose(1, 2).reshape(B, C, H, W)

        # Fusion
        y = self.fusion(y)

        # Residual
        return x + y


class MobileViT(nn.Module):
    """
    MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer.

    Hybrid CNN-Transformer architecture.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        num_classes: int = 1000,
        dims: list = [64, 80, 96],
        channels: list = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
    ):
        """
        Initialize MobileViT.

        Args:
            image_size: Input image size (H, W)
            num_classes: Number of output classes
            dims: Transformer dimensions for each MobileViT block
            channels: Channel configuration
        """
        super().__init__()

        # Initial convolution
        self.conv1 = ConvBNAct(3, channels[0], kernel_size=3, stride=2)

        # MV2 blocks
        self.mv2_1 = InvertedResidual(channels[0], channels[1], stride=1)
        self.mv2_2 = InvertedResidual(channels[1], channels[2], stride=2)
        self.mv2_3 = InvertedResidual(channels[2], channels[3], stride=1)
        self.mv2_4 = InvertedResidual(channels[3], channels[4], stride=2)

        # MobileViT block 1
        self.mvit1 = MobileViTBlock(channels[4], dims[0], num_heads=4)
        self.mv2_5 = InvertedResidual(channels[4], channels[5], stride=1)

        # MobileViT block 2
        self.mv2_6 = InvertedResidual(channels[5], channels[6], stride=2)
        self.mvit2 = MobileViTBlock(channels[6], dims[1], num_heads=4)
        self.mv2_7 = InvertedResidual(channels[6], channels[7], stride=1)

        # MobileViT block 3
        self.mv2_8 = InvertedResidual(channels[7], channels[8], stride=2)
        self.mvit3 = MobileViTBlock(channels[8], dims[2], num_heads=4)
        self.conv2 = ConvBNAct(channels[8], channels[9], kernel_size=1)

        # Classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[9], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Stem
        x = self.conv1(x)

        # MV2 + MobileViT blocks
        x = self.mv2_1(x)
        x = self.mv2_2(x)
        x = self.mv2_3(x)
        x = self.mv2_4(x)

        x = self.mvit1(x)
        x = self.mv2_5(x)

        x = self.mv2_6(x)
        x = self.mvit2(x)
        x = self.mv2_7(x)

        x = self.mv2_8(x)
        x = self.mvit3(x)
        x = self.conv2(x)

        # Classifier
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def mobile_vit_xxs(num_classes: int = 1000) -> MobileViT:
    """
    MobileViT-XXS: Extra-extra-small variant (~1.3M parameters).

    Args:
        num_classes: Number of output classes

    Returns:
        MobileViT-XXS model
    """
    return MobileViT(
        image_size=(256, 256),
        num_classes=num_classes,
        dims=[64, 80, 96],
        channels=[16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320],
    )


def mobile_vit_xs(num_classes: int = 1000) -> MobileViT:
    """
    MobileViT-XS: Extra-small variant (~2.3M parameters).

    Args:
        num_classes: Number of output classes

    Returns:
        MobileViT-XS model
    """
    return MobileViT(
        image_size=(256, 256),
        num_classes=num_classes,
        dims=[96, 120, 144],
        channels=[16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
    )


def mobile_vit_s(num_classes: int = 1000) -> MobileViT:
    """
    MobileViT-S: Small variant (~5.6M parameters).

    Args:
        num_classes: Number of output classes

    Returns:
        MobileViT-S model
    """
    return MobileViT(
        image_size=(256, 256),
        num_classes=num_classes,
        dims=[144, 192, 240],
        channels=[16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
    )
