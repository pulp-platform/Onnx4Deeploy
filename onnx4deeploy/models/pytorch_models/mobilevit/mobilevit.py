# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MobileViT Model for ONNX Export.

Based on "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer" (ICLR 2022)
https://github.com/apple/ml-cvnets

This is a Deploy Version optimized for ONNX export with:
- No dynamic shape operations (no x.size(), x.shape queries)
- No dynamic padding or reshaping
- All dimensions fixed at initialization
- No dropout
- inplace=False for all activations
- Canonical PyTorch style

Architecture:
    Input [B, 3, H, W]
    → Stem Conv (stride=2) [B, C0, H/2, W/2]
    → MV2 Blocks (downsampling) [B, C4, H/32, W/32]
    → MobileViT Blocks (CNN + Transformer fusion)
    → Global Average Pooling [B, C_final, 1, 1]
    → Classifier [B, num_classes]
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.init as init


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation (Deploy Version).

    Standard conv block with fixed padding, batch norm, and SiLU activation.
    No dynamic operations - all dimensions known at init.
    """

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
            kernel_size: Kernel size (must be constant)
            stride: Stride (must be constant)
            groups: Number of groups for grouped convolution
        """
        super().__init__()
        # Fixed padding - computed at init time, not runtime
        self.padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=self.padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=False)  # inplace=False for ONNX compatibility

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, in_channels, H, W]

        Returns:
            Output tensor [B, out_channels, H/stride, W/stride]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block (Deploy Version).

    Standard inverted residual: 1x1 expand → 3x3 depthwise → 1x1 project
    Optional residual connection when stride=1 and in_channels == out_channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 4,
    ):
        """
        Initialize inverted residual block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride (1 or 2)
            expand_ratio: Channel expansion ratio
        """
        super().__init__()
        self.stride = stride
        self.hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(ConvBNAct(in_channels, self.hidden_dim, kernel_size=1))

        # Depthwise convolution
        layers.extend(
            [
                ConvBNAct(
                    self.hidden_dim,
                    self.hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    groups=self.hidden_dim,
                ),
                # Pointwise projection (no activation)
                nn.Conv2d(self.hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional residual connection.

        Args:
            x: Input tensor [B, in_channels, H, W]

        Returns:
            Output tensor [B, out_channels, H/stride, W/stride]
        """
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MultiheadSelfAttention(nn.Module):
    """
    Multi-head Self-Attention with FIXED dimensions for clean ONNX export.

    CRITICAL: All dimensions (batch_size, seq_len, num_heads, head_dim) are
    defined at initialization and used in forward pass - NO dynamic operations!

    This avoids Shape/Gather nodes that make ONNX graphs ugly.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        batch_size: int = 1,
        seq_len: int = 1024,
    ):
        """
        Initialize multi-head self-attention with FIXED dimensions.

        Args:
            dim: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            batch_size: Fixed batch size
            seq_len: Fixed sequence length (num_patches)
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # Fixed dimensions - defined at init, never queried at runtime
        self.D = dim
        self.H = num_heads
        self.B = batch_size
        self.T = seq_len
        self.Hd = dim // num_heads  # Head dimension
        self.scale = self.Hd**-0.5

        # Projection layers
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FIXED reshaping dimensions.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Output tensor [B, T, D]
        """
        # Project to Q, K, V using FIXED dimensions
        # [B, T, D] -> [B, T, H, Hd] -> [B, H, T, Hd]
        q = self.q_proj(x).reshape(self.B, self.T, self.H, self.Hd).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(self.B, self.T, self.H, self.Hd).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(self.B, self.T, self.H, self.Hd).permute(0, 2, 1, 3)

        # Compute attention scores: [B, H, T, T]
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        # Apply softmax (ONNX native operator)
        attn = torch.nn.functional.softmax(attn_scores, dim=-1)

        # Apply attention to values: [B, H, T, Hd] -> [B, T, H, Hd] -> [B, T, D]
        out = (attn @ v).transpose(1, 2).reshape(self.B, self.T, self.D)
        out = self.out_proj(out)

        return out


class TransformerBlock(nn.Module):
    """Transformer block for MobileViT (Deploy Version).

    Standard transformer block: LayerNorm → Multi-head Self-Attention → MLP
    Uses pre-norm architecture with residual connections.
    No dropout for deployment.

    CRITICAL: Uses custom MultiheadSelfAttention with fixed dimensions instead
    of nn.MultiheadAttention to avoid dynamic operations in ONNX.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        batch_size: int = 1,
        seq_len: int = 1024,
    ):
        """
        Initialize transformer block with FIXED dimensions.

        Args:
            dim: Input/output dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            batch_size: Fixed batch size
            seq_len: Fixed sequence length (num_patches)
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        # Use custom attention with fixed dimensions
        self.attn = MultiheadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, self.mlp_hidden_dim),
            nn.SiLU(inplace=False),  # inplace=False for ONNX
            nn.Linear(self.mlp_hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Output tensor [B, T, D]
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm)
        x = x + attn_out  # [B, T, D]

        # MLP with residual
        x = x + self.mlp(self.norm2(x))  # [B, T, D]

        return x


class MobileViTBlock(nn.Module):
    """MobileViT block: Local (Conv) + Global (Transformer) representations (Deploy Version).

    Architecture:
        Input [B, in_channels, H, W]
        → Local Rep: Conv3x3 + Conv1x1 [B, transformer_dim, H, W]
        → Unfold to patches [B, H*W, transformer_dim]
        → Transformer blocks [B, H*W, transformer_dim]
        → Fold back [B, transformer_dim, H, W]
        → Fusion: Conv1x1 [B, in_channels, H, W]
        → Residual connection + Input [B, in_channels, H, W]

    CRITICAL: All dimensions (B, H, W, num_patches) are fixed at initialization.
    No dynamic shape operations allowed.
    """

    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        num_heads: int = 4,
        num_transformer_blocks: int = 2,
        patch_h: int = 8,  # Fixed patch height
        patch_w: int = 8,  # Fixed patch width
        batch_size: int = 1,  # Fixed batch size for ONNX
    ):
        """
        Initialize MobileViT block with FIXED dimensions.

        Args:
            in_channels: Number of input channels
            transformer_dim: Transformer dimension
            num_heads: Number of attention heads
            num_transformer_blocks: Number of transformer blocks
            patch_h: Fixed spatial height after local_rep
            patch_w: Fixed spatial width after local_rep
            batch_size: Fixed batch size for ONNX export
        """
        super().__init__()
        self.in_channels = in_channels
        self.transformer_dim = transformer_dim
        self.num_heads = num_heads
        self.batch_size = batch_size

        # FIXED dimensions - computed at init, never at runtime
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.num_patches = patch_h * patch_w  # Fixed number of patches

        # Local representation: Conv3x3 + Conv1x1
        self.local_rep = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            ConvBNAct(in_channels, transformer_dim, kernel_size=1),
        )

        # Global representation: Transformer blocks with FIXED dimensions
        self.global_rep = nn.Sequential(
            *[
                TransformerBlock(
                    transformer_dim,
                    num_heads=num_heads,
                    batch_size=batch_size,
                    seq_len=self.num_patches,
                )
                for _ in range(num_transformer_blocks)
            ]
        )

        # Fusion: Conv1x1 to project back
        self.fusion = ConvBNAct(transformer_dim, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FIXED reshaping dimensions.

        Args:
            x: Input tensor [B, in_channels, patch_h, patch_w]

        Returns:
            Output tensor [B, in_channels, patch_h, patch_w]
        """
        shortcut = x  # Save for residual connection

        # Local representation
        y = self.local_rep(x)  # [B, transformer_dim, patch_h, patch_w]

        # Unfold patches for transformer (FIXED dimensions only!)
        # Use reshape with FIXED batch_size instead of -1 for ONNX
        y = y.reshape(
            self.batch_size, self.transformer_dim, self.num_patches
        )  # [B, transformer_dim, H*W]
        y = y.transpose(1, 2)  # [B, num_patches, transformer_dim]

        # Global representation (Transformer)
        y = self.global_rep(y)  # [B, num_patches, transformer_dim]

        # Fold back to spatial (FIXED dimensions!)
        y = y.transpose(1, 2)  # [B, transformer_dim, num_patches]
        y = y.reshape(
            self.batch_size, self.transformer_dim, self.patch_h, self.patch_w
        )  # [B, transformer_dim, H, W]

        # Fusion
        y = self.fusion(y)  # [B, in_channels, patch_h, patch_w]

        # Residual connection
        return shortcut + y


class MobileViT(nn.Module):
    """
    MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer (Deploy Version).

    Hybrid CNN-Transformer architecture optimized for ONNX export.

    Architecture Overview:
        Input: [B, 3, 256, 256]

        Stem:
            Conv3x3 (s=2) → [B, 16, 128, 128]

        Stage 1 (MV2 blocks):
            MV2 → [B, 32, 128, 128]
            MV2 (s=2) → [B, 48, 64, 64]
            MV2 → [B, 48, 64, 64]
            MV2 (s=2) → [B, 64, 32, 32]

        Stage 2 (MobileViT block 1):
            MobileViT (64 → 64, trans_dim=96) → [B, 64, 32, 32]
            MV2 → [B, 64, 32, 32]

        Stage 3 (MobileViT block 2):
            MV2 (s=2) → [B, 80, 16, 16]
            MobileViT (80 → 80, trans_dim=120) → [B, 80, 16, 16]
            MV2 → [B, 80, 16, 16]

        Stage 4 (MobileViT block 3):
            MV2 (s=2) → [B, 96, 8, 8]
            MobileViT (96 → 96, trans_dim=144) → [B, 96, 8, 8]
            Conv1x1 → [B, 384, 8, 8]

        Classifier:
            GlobalAvgPool → [B, 384, 1, 1]
            Flatten → [B, 384]
            Linear → [B, num_classes]

    Optimizations for ONNX:
        ✅ No dropout
        ✅ All dimensions fixed at __init__
        ✅ No dynamic shape operations (x.size(), x.shape)
        ✅ No dynamic padding or reshaping
        ✅ inplace=False for all activations
        ✅ Uses nn.Flatten instead of torch.flatten
        ✅ Pre-computed patch dimensions for MobileViT blocks
    """

    def __init__(
        self,
        batch_size: int = 1,
        image_size: Tuple[int, int] = (256, 256),
        num_classes: int = 1000,
        dims: list = [96, 120, 144],
        channels: list = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
    ):
        """
        Initialize MobileViT with FIXED dimensions.

        Args:
            batch_size: Fixed batch size for deployment
            image_size: Input image size (H, W) - must be fixed
            num_classes: Number of output classes
            dims: Transformer dimensions for each MobileViT block
            channels: Channel configuration for each stage
        """
        super().__init__()

        # Fixed dimensions - defined at init, never queried at runtime
        self.batch_size = batch_size
        self.image_h, self.image_w = image_size
        self.num_classes = num_classes

        # Compute spatial dimensions at each stage (for MobileViT blocks)
        # After stem (stride=2): H/2, W/2 = 128x128
        # After mv2_2 (stride=2): H/4, W/4 = 64x64
        # After mv2_4 (stride=2): H/8, W/8 = 32x32  <- MobileViT block 1
        # After mv2_6 (stride=2): H/16, W/16 = 16x16 <- MobileViT block 2
        # After mv2_8 (stride=2): H/32, W/32 = 8x8   <- MobileViT block 3
        self.mvit_patch_dims = [
            (self.image_h // 8, self.image_w // 8),  # MobileViT block 1: 32x32
            (self.image_h // 16, self.image_w // 16),  # MobileViT block 2: 16x16
            (self.image_h // 32, self.image_w // 32),  # MobileViT block 3: 8x8
        ]

        # Initial convolution (stem)
        self.conv1 = ConvBNAct(3, channels[0], kernel_size=3, stride=2)

        # Stage 1: MV2 blocks
        self.mv2_1 = InvertedResidual(channels[0], channels[1], stride=1)
        self.mv2_2 = InvertedResidual(channels[1], channels[2], stride=2)
        self.mv2_3 = InvertedResidual(channels[2], channels[3], stride=1)
        self.mv2_4 = InvertedResidual(channels[3], channels[4], stride=2)

        # Stage 2: MobileViT block 1
        patch_h_1, patch_w_1 = self.mvit_patch_dims[0]
        self.mvit1 = MobileViTBlock(
            channels[4],
            dims[0],
            num_heads=4,
            patch_h=patch_h_1,
            patch_w=patch_w_1,
            batch_size=batch_size,
        )
        self.mv2_5 = InvertedResidual(channels[4], channels[5], stride=1)

        # Stage 3: MobileViT block 2
        self.mv2_6 = InvertedResidual(channels[5], channels[6], stride=2)
        patch_h_2, patch_w_2 = self.mvit_patch_dims[1]
        self.mvit2 = MobileViTBlock(
            channels[6],
            dims[1],
            num_heads=4,
            patch_h=patch_h_2,
            patch_w=patch_w_2,
            batch_size=batch_size,
        )
        self.mv2_7 = InvertedResidual(channels[6], channels[7], stride=1)

        # Stage 4: MobileViT block 3
        self.mv2_8 = InvertedResidual(channels[7], channels[8], stride=2)
        patch_h_3, patch_w_3 = self.mvit_patch_dims[2]
        self.mvit3 = MobileViTBlock(
            channels[8],
            dims[2],
            num_heads=4,
            patch_h=patch_h_3,
            patch_w=patch_w_3,
            batch_size=batch_size,
        )
        self.conv2 = ConvBNAct(channels[8], channels[9], kernel_size=1)

        # Classifier head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)  # Use nn.Flatten for ONNX
        self.fc = nn.Linear(channels[9], num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming/Xavier initialization.

        Conv layers: Kaiming normal (fan-out mode for ReLU-family activations)
        Linear layers: Xavier normal
        BatchNorm: weight=1, bias=0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",  # SiLU behaves similarly to ReLU
                )
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if m.weight is not None:
                    init.constant_(m.weight, 1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with detailed dimension tracking.

        Args:
            x: Input tensor [B, 3, 256, 256]

        Returns:
            Output logits [B, num_classes]
        """
        # Stem
        x = self.conv1(x)  # [B, 16, 128, 128]

        # Stage 1: MV2 blocks with downsampling
        x = self.mv2_1(x)  # [B, 32, 128, 128]
        x = self.mv2_2(x)  # [B, 48, 64, 64] (stride=2)
        x = self.mv2_3(x)  # [B, 48, 64, 64]
        x = self.mv2_4(x)  # [B, 64, 32, 32] (stride=2)

        # Stage 2: MobileViT block 1
        x = self.mvit1(x)  # [B, 64, 32, 32] (CNN + Transformer)
        x = self.mv2_5(x)  # [B, 64, 32, 32]

        # Stage 3: MobileViT block 2
        x = self.mv2_6(x)  # [B, 80, 16, 16] (stride=2)
        x = self.mvit2(x)  # [B, 80, 16, 16] (CNN + Transformer)
        x = self.mv2_7(x)  # [B, 80, 16, 16]

        # Stage 4: MobileViT block 3
        x = self.mv2_8(x)  # [B, 96, 8, 8] (stride=2)
        x = self.mvit3(x)  # [B, 96, 8, 8] (CNN + Transformer)
        x = self.conv2(x)  # [B, 384, 8, 8]

        # Classifier head
        x = self.pool(x)  # [B, 384, 1, 1]
        x = self.flatten(x)  # [B, 384]
        x = self.fc(x)  # [B, num_classes]

        return x


def mobile_vit_xxs(
    batch_size: int = 1,
    image_size: Tuple[int, int] = (256, 256),
    num_classes: int = 1000,
) -> MobileViT:
    """
    MobileViT-XXS: Extra-extra-small variant (~1.3M parameters).

    Args:
        batch_size: Fixed batch size for deployment
        image_size: Input image size (H, W)
        num_classes: Number of output classes

    Returns:
        MobileViT-XXS model (Deploy Version)
    """
    return MobileViT(
        batch_size=batch_size,
        image_size=image_size,
        num_classes=num_classes,
        dims=[64, 80, 96],
        channels=[16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320],
    )


def mobile_vit_xs(
    batch_size: int = 1,
    image_size: Tuple[int, int] = (256, 256),
    num_classes: int = 1000,
) -> MobileViT:
    """
    MobileViT-XS: Extra-small variant (~2.3M parameters).

    Args:
        batch_size: Fixed batch size for deployment
        image_size: Input image size (H, W)
        num_classes: Number of output classes

    Returns:
        MobileViT-XS model (Deploy Version)
    """
    return MobileViT(
        batch_size=batch_size,
        image_size=image_size,
        num_classes=num_classes,
        dims=[96, 120, 144],
        channels=[16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
    )


def mobile_vit_s(
    batch_size: int = 1,
    image_size: Tuple[int, int] = (256, 256),
    num_classes: int = 1000,
) -> MobileViT:
    """
    MobileViT-S: Small variant (~5.6M parameters).

    Args:
        batch_size: Fixed batch size for deployment
        image_size: Input image size (H, W)
        num_classes: Number of output classes

    Returns:
        MobileViT-S model (Deploy Version)
    """
    return MobileViT(
        batch_size=batch_size,
        image_size=image_size,
        num_classes=num_classes,
        dims=[144, 192, 240],
        channels=[16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
    )
