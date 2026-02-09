# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""SleepConViT - Vision Transformer for Sleep Stage Classification (Deploy Version)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class MultiheadSelfAttention(nn.Module):
    """
    Multi-head Self-Attention module with fixed dimensions for ONNX export.

    Args:
        d_model: Model dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        seq_len: Fixed sequence length
        batch_size: Fixed batch size
        dropout: Dropout rate (ignored in deploy version)
    """

    def __init__(self, d_model=128, num_heads=8, seq_len=95, batch_size=1, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.D = d_model
        self.H = num_heads
        self.T = seq_len
        self.B = batch_size
        self.Hd = d_model // num_heads
        self.scale = self.Hd**-0.5

        # Projection layers
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # Note: Dropout removed for inference/deployment

    def forward(self, x):
        """
        Forward pass for multi-head self-attention.

        Args:
            x: Input tensor of shape (B, T, D)

        Returns:
            Output tensor of shape (B, T, D)
        """
        # Project to Q, K, V and reshape to [B, H, T, Hd]
        q = self.q_proj(x).reshape(self.B, self.T, self.H, self.Hd).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(self.B, self.T, self.H, self.Hd).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(self.B, self.T, self.H, self.Hd).permute(0, 2, 1, 3)

        # Compute attention scores: [B, H, T, T]
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        # Apply softmax
        attn = F.softmax(attn_scores, dim=-1)

        # Apply attention to values and reshape: [B, T, D]
        out = (attn @ v).transpose(1, 2).reshape(self.B, self.T, self.D)
        out = self.out_proj(out)

        return out


class LinClassifier(nn.Module):
    """
    Linear classifier head.

    Args:
        input_dim: Input feature dimension
        output_dim: Number of output classes
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Classification logits
        """
        return self.lin1(x)


class MLPHead(nn.Module):
    """
    MLP feedforward head with GELU activation.

    Uses F.gelu (functional interface) for better ONNX export compatibility.
    This ensures proper shape inference and exports as a single fused operator.

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden layer dimension
        dropout_rate: Dropout rate (ignored in deploy version)
    """

    def __init__(self, dim, hidden_dim, dropout_rate=0.0):
        super().__init__()
        self.ff1 = nn.Linear(dim, hidden_dim)
        # Use F.gelu functional interface (same as CCT)
        self.activation = F.gelu
        self.ff2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = self.ff1(x)
        x = self.activation(x)
        x = self.ff2(x)
        return x


class ConvStem(nn.Module):
    """
    Dual-Branch Convolutional Stem for feature extraction and downsampling (2D version).

    This module applies two parallel convolutional branches with different kernel sizes
    to capture multi-scale temporal features, then concatenates the results.
    Simplified to 2 branches for better Deeploy compatibility.
    Uses Conv2d for better compatibility with ONNX Runtime transformer optimizer.

    Args:
        in_channels: Number of input channels
        out_channels: Total number of output channels across both branches
        kernel_sizes: Tuple of kernel sizes for the two branches (width dimension)
        stride: Stride for downsampling in convolutional layers
        pool_kernel: Kernel size for pooling layers
    """

    def __init__(
        self, in_channels=1, out_channels=48, kernel_sizes=(25, 100), stride=4, pool_kernel=4
    ):
        super().__init__()
        # Divide the total output channels equally across the 2 branches
        branch_out_channels = out_channels // 2

        # Branch 1: Kernel size 25 (fine-grained features)
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=branch_out_channels,
                kernel_size=(1, kernel_sizes[0]),  # (height, width)
                stride=(1, stride),
                padding=(0, kernel_sizes[0] // 2),
                bias=False,
            ),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel)),
            nn.Conv2d(
                in_channels=branch_out_channels,
                out_channels=branch_out_channels,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
                bias=False,
            ),
            nn.ReLU(inplace=False),
        )
        
        # Branch 2: Kernel size 200 (middle-grained features)
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=branch_out_channels,
                kernel_size=(1, kernel_sizes[1]),
                stride=(1, stride),
                padding=(0, kernel_sizes[1] // 2),
                bias=False
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel)),
            nn.Conv2d(
                in_channels=branch_out_channels,
                out_channels=branch_out_channels,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
                bias=False
            ),
            nn.ReLU(inplace=True)
        )

        # Branch 3: Kernel size 100 (coarse-grained features)
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=branch_out_channels,
                kernel_size=(1, kernel_sizes[1]),
                stride=(1, stride),
                padding=(0, kernel_sizes[1] // 2),
                bias=False,
            ),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel)),
            nn.Conv2d(
                in_channels=branch_out_channels,
                out_channels=branch_out_channels,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
                bias=False,
            ),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        """
        Forward pass through dual-branch convolutional stem.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               Expected: (B, 1, 1, 3000)

        Returns:
            Concatenated features from both branches (B, model_dim, 1, num_patches)
        """
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        # Single concatenation (Deeploy compatible)
        x12 = torch.cat((x1, x2), dim=1)
        x123 = torch.cat((x12, x3), dim=1)
        return x123


class Encoder(nn.Module):
    """
    Transformer Encoder block with multi-head attention and feedforward network.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        seq_len: Fixed sequence length
        batch_size: Fixed batch size
        att_dropout: Attention dropout rate (ignored in deploy version)
        mlp_head_hidden_dim: Hidden dimension for MLP head
        mlp_head_dropout: MLP dropout rate (ignored in deploy version)
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        seq_len,
        batch_size,
        att_dropout=0.0,
        mlp_head_hidden_dim=512,
        mlp_head_dropout=0.0,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.mha = MultiheadSelfAttention(
            embed_dim, num_heads, seq_len, batch_size, dropout=att_dropout
        )
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.ff = MLPHead(embed_dim, hidden_dim=mlp_head_hidden_dim, dropout_rate=mlp_head_dropout)

    def forward(self, x):
        """
        Forward pass through encoder block.

        Args:
            x: Input tensor of shape (B, T, D)

        Returns:
            Output tensor of shape (B, T, D)
        """
        # First residual block: LayerNorm -> Multi-head Attention -> Residual
        _x = x
        x = self.ln_1(x)
        x = self.mha(x)
        x = _x + x

        # Second residual block: LayerNorm -> MLP -> Residual
        _x = x
        x = self.ln_2(x)
        x = self.ff(x)
        x = _x + x

        return x


class SleepConViT(nn.Module):
    """
    SleepConViT - Vision Transformer for Sleep Stage Classification (Deploy Version).

    This is a clean deployment version optimized for ONNX export:
    - No dropout layers (removed for inference)
    - Fixed input shape (no dynamic operations)
    - inplace=False for all ReLU activations
    - Clean, linear computation graph

    Architecture:
    - Input: Time-series signal (batch_size, channels, height, width)
      Shape: (B, 1, 1, 3000)
    - ConvStem: Multi-scale Conv2d feature extraction -> (B, model_dim, 1, num_patches)
    - Reshape: (B, model_dim, 1, num_patches) -> (B, num_patches, model_dim)
    - Add CLS token -> (B, num_patches+1, model_dim)
    - Add positional embeddings
    - Transformer Encoder
    - LayerNorm
    - Select CLS token -> (B, model_dim)
    - Linear Classifier -> (B, num_classes)
    - Output: Class logits (batch_size, num_classes)

    Args:
        config: Dictionary containing model configuration:
            - batch_size: Fixed batch size (default: 1)
            - num_heads: Number of attention heads (default: 8)
            - model_dim: Model dimension (default: 48)
            - num_patches: Number of patches/sequence length after ConvStem (default: 94)
            - seq_len: Sequence length including CLS token (default: 95)
            - attention_dropout: Attention dropout (ignored, default: 0.0)
            - mlp_head_hidden_dim: MLP hidden dimension (default: 192)
            - encoder_ff_dropout: Encoder feedforward dropout (ignored, default: 0.0)
            - num_classes: Number of output classes (default: 5)
    """

    def __init__(self, config: dict):
        super().__init__()

        # Extract config parameters
        self.num_heads = config.get("num_heads", 8)
        self.model_dim = config.get("model_dim", 48)
        self.num_patches = config.get("num_patches", 94)
        self.num_classes = config.get("num_classes", 5)
        self.batch_size = config.get("batch_size", 1)
        seq_len = config.get("seq_len", 95)  # num_patches + 1 (for CLS token)

        # Convolutional Stem (2 branches for Deeploy compatibility)
        self.conv_stem = ConvStem(
            in_channels=1,
            out_channels=self.model_dim,
            kernel_sizes=(25, 100),  # 2 branches: fine-grained (25) and coarse-grained (100)
            stride=4,
        )

        # Transformer Encoder
        self.encoder = Encoder(
            embed_dim=self.model_dim,
            num_heads=self.num_heads,
            seq_len=seq_len,
            batch_size=self.batch_size,
            att_dropout=config.get("attention_dropout", 0.0),
            mlp_head_hidden_dim=config.get("mlp_head_hidden_dim", 192),
            mlp_head_dropout=config.get("encoder_ff_dropout", 0.0),
        )

        # CLS token and positional embeddings (learnable parameters)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.model_dim))

        # CLS token selector (fixed one-hot vector for ONNX-friendly extraction)
        self.cls_selector = nn.Parameter(torch.zeros(1, self.num_patches + 1), requires_grad=False)
        self.cls_selector.data[0, 0] = 1.0  # Select only the first token (CLS)

        # Final normalization and classifier
        self.norm = nn.LayerNorm(self.model_dim)
        self.classifier = LinClassifier(self.model_dim, self.num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights for the model.
        Applies Kaiming initialization for Conv layers, normal initialization for Linear layers,
        and proper initialization for LayerNorm.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                init.constant_(m.weight, 1.0)

        # Initialize CLS token and positional embeddings
        init.normal_(self.cls_token, std=0.02)
        init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Forward pass (clean, no branching, no dynamic operations).

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               Expected shape: (B, 1, 1, 3000)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Convolutional feature extraction (Conv2d)
        # (B, 1, 1, 3000) -> (B, model_dim, 1, num_patches)
        x = self.conv_stem(x)

        # Reshape: (B, model_dim, 1, num_patches) -> (B, num_patches, model_dim)
        # Avoid squeeze + transpose by using reshape directly
        x = x.reshape(self.batch_size, self.model_dim, self.num_patches).permute(0, 2, 1)

        # Prepend CLS token: (B, num_patches, model_dim) -> (B, num_patches+1, model_dim)
        # Expand CLS token to match batch size
        cls_tokens = self.cls_token.expand(self.batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings
        x = x + self.pos_embed

        # Transformer encoder
        x = self.encoder(x)

        # Final normalization
        x = self.norm(x)

        # Select CLS token: (B, num_patches+1, model_dim) -> (B, model_dim)
        # Use matrix multiplication with one-hot selector to avoid Gather in ONNX
        # cls_selector: [1, 95], x: [B, 95, 48] -> result: [B, 1, 48]
        x = torch.matmul(self.cls_selector, x)
        x = x.squeeze(1)  # [B, 1, 48] -> [B, 48]

        # Classification
        x = self.classifier(x)

        return x
