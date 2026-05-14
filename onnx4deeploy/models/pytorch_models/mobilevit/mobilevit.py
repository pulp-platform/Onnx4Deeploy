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


class SiLU(nn.Module):
    """SiLU activation that exports cleanly to ONNX.

    Uses torch.nn.functional.silu which has built-in ONNX support
    with proper shape inference.
    """

    def forward(self, x):
        return torch.nn.functional.silu(x)


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
        self.act = SiLU()  # Separate ONNX node

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
            SiLU(),  # Separate ONNX node
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
    """MobileViT block (Static deploy version, ONNX-friendly).

    Key properties:
    - FIXED B/H/W at init (no x.shape, no dynamic reshape)
    - TRUE patching with fixed patch_size (ph, pw)
    - Transformer is applied on tokens within each patch:
        effective_batch = B * (H/ph) * (W/pw)
        seq_len        = ph * pw
        dim            = transformer_dim
    - Fusion follows MobileViT definition:
        proj (1x1) -> concat(input, proj) -> 3x3 conv fusion
    """

    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        feat_h: int,
        feat_w: int,
        patch_size: Tuple[int, int] = (2, 2),  # fixed patch size (ph, pw)
        num_heads: int = 4,
        num_transformer_blocks: int = 2,
        batch_size: int = 1,  # fixed batch size for ONNX
    ):
        super().__init__()
        self.in_channels = in_channels
        self.transformer_dim = transformer_dim

        # Fixed dimensions (must be divisible)
        self.B = batch_size
        self.H = feat_h
        self.W = feat_w
        self.ph, self.pw = patch_size
        assert (
            self.H % self.ph == 0 and self.W % self.pw == 0
        ), "feat_h/feat_w must be divisible by patch_size"

        self.nh = self.H // self.ph
        self.nw = self.W // self.pw
        self.num_patches = self.nh * self.nw  # fixed
        self.patch_area = self.ph * self.pw  # fixed

        # Transformer will run on:
        # x: [B*num_patches, patch_area, transformer_dim]
        self.tr_B = self.B * self.num_patches  # fixed
        self.tr_T = self.patch_area  # fixed

        # Local representation
        self.local_rep = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            ConvBNAct(in_channels, transformer_dim, kernel_size=1),
        )

        # Global representation (transformer)
        self.global_rep = nn.Sequential(
            *[
                TransformerBlock(
                    dim=transformer_dim,
                    num_heads=num_heads,
                    mlp_ratio=2.0,
                    batch_size=self.tr_B,
                    seq_len=self.tr_T,
                )
                for _ in range(num_transformer_blocks)
            ]
        )

        # Project back + fusion (MobileViT definition)
        self.proj = ConvBNAct(transformer_dim, in_channels, kernel_size=1)
        self.fusion = ConvBNAct(in_channels * 2, in_channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W] where B/H/W are fixed constants
        shortcut = x

        # Local rep: [B, D, H, W]
        y = self.local_rep(x)

        # --- Unfold into patches (STATIC) ---
        # y: [B, D, H, W]
        # -> [B, D, nh, ph, nw, pw]
        y = y.reshape(self.B, self.transformer_dim, self.nh, self.ph, self.nw, self.pw)
        # -> [B, nh, nw, ph, pw, D]
        y = y.permute(0, 2, 4, 3, 5, 1)
        # -> [B*nh*nw, ph*pw, D]
        y = y.reshape(self.tr_B, self.tr_T, self.transformer_dim)

        # Transformer: [B*nh*nw, ph*pw, D]
        y = self.global_rep(y)

        # --- Fold back (STATIC) ---
        # -> [B, nh, nw, ph, pw, D]
        y = y.reshape(self.B, self.nh, self.nw, self.ph, self.pw, self.transformer_dim)
        # -> [B, D, nh, ph, nw, pw]
        y = y.permute(0, 5, 1, 3, 2, 4)
        # -> [B, D, H, W]
        y = y.reshape(self.B, self.transformer_dim, self.H, self.W)

        # Project back: [B, C_in, H, W]
        y = self.proj(y)

        # Fusion: concat + 3x3 conv (MobileViT definition)
        y = torch.cat([shortcut, y], dim=1)  # [B, 2*C_in, H, W]
        y = self.fusion(y)  # [B, C_in, H, W]

        return y


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
        transformer_depths: list = [2, 4, 3],
        mv2_expand_ratio: int = 4,
    ):
        """
        Initialize MobileViT with FIXED dimensions.

        Args:
            batch_size: Fixed batch size for deployment
            image_size: Input image size (H, W) - must be fixed
            num_classes: Number of output classes
            dims: Transformer dimensions for each MobileViT block
            channels: Channel configuration for each stage
            transformer_depths: Number of transformer layers (L) per MobileViT
                block. Paper Table-3 uses [2, 4, 3] for all three variants.
            mv2_expand_ratio: Expansion factor for the MV2 inverted-residual
                blocks. Paper §3.3 uses 2 for XXS and 4 for XS/S.
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
        self.mv2_1 = InvertedResidual(
            channels[0], channels[1], stride=1, expand_ratio=mv2_expand_ratio
        )
        self.mv2_2 = InvertedResidual(
            channels[1], channels[2], stride=2, expand_ratio=mv2_expand_ratio
        )
        self.mv2_3 = InvertedResidual(
            channels[2], channels[3], stride=1, expand_ratio=mv2_expand_ratio
        )
        self.mv2_4 = InvertedResidual(
            channels[3], channels[4], stride=2, expand_ratio=mv2_expand_ratio
        )

        # Stage 2: MobileViT block 1
        feat_h_1, feat_w_1 = self.mvit_patch_dims[0]
        self.mvit1 = MobileViTBlock(
            channels[4],
            dims[0],
            feat_h=feat_h_1,
            feat_w=feat_w_1,
            patch_size=(2, 2),
            num_heads=4,
            num_transformer_blocks=transformer_depths[0],
            batch_size=batch_size,
        )
        self.mv2_5 = InvertedResidual(
            channels[4], channels[5], stride=1, expand_ratio=mv2_expand_ratio
        )

        # Stage 3: MobileViT block 2
        self.mv2_6 = InvertedResidual(
            channels[5], channels[6], stride=2, expand_ratio=mv2_expand_ratio
        )
        feat_h_2, feat_w_2 = self.mvit_patch_dims[1]
        self.mvit2 = MobileViTBlock(
            channels[6],
            dims[1],
            feat_h=feat_h_2,
            feat_w=feat_w_2,
            patch_size=(2, 2),
            num_heads=4,
            num_transformer_blocks=transformer_depths[1],
            batch_size=batch_size,
        )
        self.mv2_7 = InvertedResidual(
            channels[6], channels[7], stride=1, expand_ratio=mv2_expand_ratio
        )

        # Stage 4: MobileViT block 3
        self.mv2_8 = InvertedResidual(
            channels[7], channels[8], stride=2, expand_ratio=mv2_expand_ratio
        )
        feat_h_3, feat_w_3 = self.mvit_patch_dims[2]
        self.mvit3 = MobileViTBlock(
            channels[8],
            dims[2],
            feat_h=feat_h_3,
            feat_w=feat_w_3,
            patch_size=(2, 2),
            num_heads=4,
            num_transformer_blocks=transformer_depths[2],
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
        transformer_depths=[2, 4, 3],
        mv2_expand_ratio=2,
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
        transformer_depths=[2, 4, 3],
        mv2_expand_ratio=4,
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
        transformer_depths=[2, 4, 3],
        mv2_expand_ratio=4,
    )
