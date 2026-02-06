# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""TinyViT Model for ONNX Export.

Based on "TinyViT: Fast Pretraining Distillation for Small Vision Transformers" (ECCV 2022)
https://github.com/microsoft/Cream/tree/main/TinyViT

This is a simplified implementation optimized for ONNX export.
"""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Image to Patch Embedding with Conv2d."""

    def __init__(
        self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 192
    ):
        """
        Initialize patch embedding.

        Args:
            img_size: Input image size
            patch_size: Patch size
            in_chans: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Use Conv2d for patch embedding (more efficient than unfold)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        """
        Initialize attention module.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class MLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None):
        """
        Initialize MLP.

        Args:
            in_features: Input feature dimension
            hidden_features: Hidden layer dimension
            out_features: Output feature dimension
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        """
        Initialize transformer block.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    """
    TinyViT: Fast Pretraining Distillation for Small Vision Transformers.

    This is a simplified version for ONNX export.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
    ):
        """
        Initialize TinyViT.

        Args:
            img_size: Input image size
            patch_size: Patch size
            in_chans: Number of input channels
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )

        # Classifier head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)

        # Add position embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classifier
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)

        return x


def tiny_vit_5m(num_classes: int = 1000, img_size: int = 224) -> TinyViT:
    """
    TinyViT-5M: 5M parameters variant.

    Args:
        num_classes: Number of output classes
        img_size: Input image size

    Returns:
        TinyViT-5M model
    """
    return TinyViT(
        img_size=img_size,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_classes=num_classes,
    )


def tiny_vit_11m(num_classes: int = 1000, img_size: int = 224) -> TinyViT:
    """
    TinyViT-11M: 11M parameters variant.

    Args:
        num_classes: Number of output classes
        img_size: Input image size

    Returns:
        TinyViT-11M model
    """
    return TinyViT(
        img_size=img_size,
        patch_size=16,
        embed_dim=256,
        depth=12,
        num_heads=4,
        num_classes=num_classes,
    )


def tiny_vit_21m(num_classes: int = 1000, img_size: int = 224) -> TinyViT:
    """
    TinyViT-21M: 21M parameters variant.

    Args:
        num_classes: Number of output classes
        img_size: Input image size

    Returns:
        TinyViT-21M model
    """
    return TinyViT(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
    )
