# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""TinyViT Model for ONNX Export.

Based on "TinyViT: Fast Pretraining Distillation for Small Vision Transformers" (ECCV 2022)
https://github.com/microsoft/Cream/tree/main/TinyViT

This is a deployment version optimized for ONNX export with:
- ✅ No dynamic shape operations (x.size(), x.shape[i])
- ✅ Fixed dimensions defined in __init__ (batch_size, num_patches)
- ✅ No dropout layers (removed for inference)
- ✅ Clean, linear computation graph
- ✅ All dimensions known at export time

Key Optimizations:
- Attention module uses fixed B, N, C dimensions for reshape operations
- All transformer blocks receive batch_size and num_patches parameters
- Position embeddings are fixed size (no dynamic slicing)
- No dynamic padding or runtime shape queries
"""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Image to Patch Embedding with Conv2d (ONNX-friendly)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        batch_size: int = 1,
    ):
        """
        Initialize patch embedding.

        Args:
            img_size: Input image size
            patch_size: Patch size
            in_chans: Number of input channels
            embed_dim: Embedding dimension
            batch_size: Fixed batch size for ONNX export
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.batch_size = batch_size
        self.embed_dim = embed_dim

        # Use Conv2d for patch embedding (more efficient than unfold)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fixed dimensions.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, N, embed_dim)
        """
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        # Use fixed reshape instead of flatten to avoid dynamic Shape nodes
        x = x.reshape(self.batch_size, self.embed_dim, self.num_patches).permute(0, 2, 1)
        # Now x is (B, N, embed_dim)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention with separate Q/K/V projections (CCT-style, ONNX-friendly)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        batch_size: int = 1,
        num_patches: int = 196,
    ):
        """
        Initialize attention module.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            batch_size: Fixed batch size for ONNX export
            num_patches: Fixed number of patches (sequence length)
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # Fixed dimensions for ONNX export
        self.B = batch_size
        self.N = num_patches
        self.C = dim
        self.H = num_heads
        self.Hd = dim // num_heads
        self.scale = self.Hd**-0.5

        # Separate Q, K, V projections (CCT style) - no Gather nodes
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fixed dimensions.

        Args:
            x: Input tensor of shape (B, N, C)

        Returns:
            Output tensor of shape (B, N, C)
        """
        # Separate Q, K, V projections - use fixed dimensions
        q = self.q_proj(x).reshape(self.B, self.N, self.H, self.Hd).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(self.B, self.N, self.H, self.Hd).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(self.B, self.N, self.H, self.Hd).permute(0, 2, 1, 3)

        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention to values - use fixed dimensions
        x = (attn @ v).transpose(1, 2).reshape(self.B, self.N, self.C)
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
    """Transformer block with fixed dimensions for ONNX export."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        batch_size: int = 1,
        num_patches: int = 196,
    ):
        """
        Initialize transformer block.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            batch_size: Fixed batch size for ONNX export
            num_patches: Fixed number of patches (sequence length)
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, batch_size=batch_size, num_patches=num_patches
        )
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
    TinyViT: Fast Pretraining Distillation for Small Vision Transformers (Deploy Version).

    This is a deployment version optimized for ONNX export:
    - No dropout layers
    - Fixed input shape (no dynamic operations)
    - Clean, linear computation graph
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
        batch_size: int = 1,
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
            batch_size: Fixed batch size for ONNX export
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.batch_size = batch_size

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            batch_size=batch_size,
        )
        num_patches = self.patch_embed.num_patches

        # CLS token (learnable parameter) - initialized with batch dimension for ONNX
        self.cls_token = nn.Parameter(torch.zeros(batch_size, 1, embed_dim))

        # Position embedding (for patches + CLS token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # CLS token selector (fixed one-hot vector for ONNX-friendly extraction)
        self.cls_selector = nn.Parameter(torch.zeros(1, num_patches + 1), requires_grad=False)
        self.cls_selector.data[0, 0] = 1.0  # Select only the first token (CLS)

        # Transformer blocks with fixed dimensions (now sequence length is num_patches + 1)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    batch_size=batch_size,
                    num_patches=num_patches + 1,  # Include CLS token in sequence
                )
                for _ in range(depth)
            ]
        )

        # Classifier head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
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
        """
        Forward pass with CLS token (ONNX-friendly).

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, num_classes)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)

        # Prepend CLS token: (B, N, embed_dim) -> (B, N+1, embed_dim)
        # cls_token is already (batch_size, 1, embed_dim), so no expand needed
        x = torch.cat((self.cls_token, x), dim=1)

        # Add position embedding (now includes CLS token position)
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Normalize
        x = self.norm(x)

        # Extract CLS token using matmul (ONNX-friendly, avoids ReduceMean issues)
        # cls_selector: [1, N+1], x: [B, N+1, embed_dim] -> result: [B, 1, embed_dim]
        x = torch.matmul(self.cls_selector, x)
        x = x.squeeze(1)  # [B, 1, embed_dim] -> [B, embed_dim]

        # Classification head
        x = self.head(x)

        return x


def tiny_vit_5m(num_classes: int = 1000, img_size: int = 224, batch_size: int = 1) -> TinyViT:
    """
    TinyViT-5M: 5M parameters variant (simplified for testing with depth=1).

    Args:
        num_classes: Number of output classes
        img_size: Input image size
        batch_size: Fixed batch size for ONNX export

    Returns:
        TinyViT-5M model
    """
    return TinyViT(
        img_size=img_size,
        patch_size=16,
        embed_dim=192,
        depth=1,  # Reduced from 12 for faster testing
        num_heads=3,
        num_classes=num_classes,
        batch_size=batch_size,
    )


def tiny_vit_11m(num_classes: int = 1000, img_size: int = 224, batch_size: int = 1) -> TinyViT:
    """
    TinyViT-11M: 11M parameters variant (simplified for testing with depth=1).

    Args:
        num_classes: Number of output classes
        img_size: Input image size
        batch_size: Fixed batch size for ONNX export

    Returns:
        TinyViT-11M model
    """
    return TinyViT(
        img_size=img_size,
        patch_size=16,
        embed_dim=256,
        depth=1,  # Reduced from 12 for faster testing
        num_heads=4,
        num_classes=num_classes,
        batch_size=batch_size,
    )


def tiny_vit_21m(num_classes: int = 1000, img_size: int = 224, batch_size: int = 1) -> TinyViT:
    """
    TinyViT-21M: 21M parameters variant (simplified for testing with depth=1).

    Args:
        num_classes: Number of output classes
        img_size: Input image size
        batch_size: Fixed batch size for ONNX export

    Returns:
        TinyViT-21M model
    """
    return TinyViT(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=1,  # Reduced from 12 for faster testing
        num_heads=6,
        num_classes=num_classes,
        batch_size=batch_size,
    )
