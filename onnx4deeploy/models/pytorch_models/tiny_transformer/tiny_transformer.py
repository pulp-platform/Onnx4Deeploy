# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Tiny patch-based Transformer for MNIST — minimal training target.

Architecture (single encoder block, no bias, ReLU FFN):
  Input : (B, num_patches, patch_dim)   e.g. (1, 16, 49) for 7×7 patches on 28×28 MNIST
  Embed : Linear(patch_dim → embed_dim)
  Norm1 : LayerNorm(embed_dim)
  Attn  : single-head self-attention (explicit Q/K/V via torch.bmm → MatMul in ONNX)
  Norm2 : LayerNorm(embed_dim)
  FFN   : Linear(embed_dim → ffn_hidden) + ReLU + Linear(ffn_hidden → embed_dim)
  Pool  : attention pooling — Linear(embed_dim→1) + Softmax → weighted sum → (B, embed_dim)
  Head  : Linear(embed_dim → num_classes)

Note on pooling: Global average pool (x.mean(dim=1)) produces a ReduceMean node whose
gradient nodes lack ONNX shape inference in ORT's training graph. We use attention-based
pooling instead (same approach as CCT), which generates only MatMul/Softmax/Transpose
nodes whose gradients are fully shape-inferable.

Total parameters (default: patch_dim=49, embed_dim=32, ffn_hidden=64, num_classes=10):
  patch_embed  : 49×32             = 1 568
  norm1 γ+β   : 32+32             =    64
  q/k/v proj   : 3×32×32          = 3 072
  out_proj     : 32×32            = 1 024
  norm2 γ+β   : 32+32             =    64
  ff1          : 32×64            = 2 048
  ff2          : 64×32            = 2 048
  attn_pool    : 32×1             =    32
  classifier   : 32×10            =   320
  ──────────────────────────────────────
  Total                           ≈ 10 240
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyTransformerMnist(nn.Module):
    """
    Tiny single-head Transformer for MNIST patch inputs.

    Uses explicit Q/K/V projection + torch.bmm so the ONNX graph contains only
    Gemm/MatMul, Transpose, Softmax, LayerNorm, Relu, Add — all supported by
    Deeploy in training mode.

    Pooling uses a learned attention score (Linear(embed_dim, 1) + Softmax) to
    compute a weighted sum over patches, avoiding ReduceMean whose gradient nodes
    lack shape inference in ORT training graphs.

    Args:
        num_patches: Number of patches per image (default: 16 for 7×7 on 28×28).
        patch_dim:   Flattened patch size in pixels (default: 49 for 7×7).
        embed_dim:   Token embedding dimension (default: 32).
        ffn_hidden:  Hidden dimension of the feed-forward block (default: 64).
        num_classes: Number of output classes (default: 10 for MNIST).
    """

    def __init__(
        self,
        num_patches: int = 16,
        patch_dim: int = 49,
        embed_dim: int = 32,
        ffn_hidden: int = 64,
        num_classes: int = 10,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        # Reciprocal scale — use Mul (not Div) to avoid unsupported Div in Deeploy
        self.inv_scale = 1.0 / math.sqrt(embed_dim)

        # Patch embedding: projects each patch to embed_dim
        self.patch_embed = nn.Linear(patch_dim, embed_dim, bias=False)

        # Encoder block — pre-norm style
        self.norm1 = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff1 = nn.Linear(embed_dim, ffn_hidden, bias=False)
        self.ff2 = nn.Linear(ffn_hidden, embed_dim, bias=False)

        # Attention pooling: replaces global average pool to avoid ReduceMean
        # whose gradient nodes have no ONNX shape inference in ORT training graphs.
        self.attn_pool = nn.Linear(embed_dim, 1, bias=False)

        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_patches, patch_dim) — pre-extracted, flattened patches.

        Returns:
            logits: (B, num_classes)
        """
        # ── Patch embedding ────────────────────────────────────────────────
        x = self.patch_embed(x)  # (B, S, D)  S=num_patches, D=embed_dim

        # ── Transformer encoder block (pre-norm, residual) ──────────────────
        residual = x
        x = self.norm1(x)  # LayerNorm over D

        # Single-head self-attention (explicit bmm → MatMul in ONNX)
        q = self.q_proj(x)  # (B, S, D)
        k = self.k_proj(x)  # (B, S, D)
        v = self.v_proj(x)  # (B, S, D)

        # Attention scores: (B, S, D) × (B, D, S) → (B, S, S)
        # Use Mul by reciprocal (not Div) — Deeploy has no Div binding
        attn = torch.bmm(q, k.transpose(1, 2)) * self.inv_scale
        attn = F.softmax(attn, dim=-1)  # (B, S, S)

        # Context: (B, S, S) × (B, S, D) → (B, S, D)
        x = torch.bmm(attn, v)
        x = self.out_proj(x)  # (B, S, D)
        x = x + residual

        # ── Feed-forward block (pre-norm, residual) ─────────────────────────
        residual = x
        x = self.norm2(x)
        x = self.ff2(F.relu(self.ff1(x)))
        x = x + residual

        # ── Attention pooling over patches → (B, D) ─────────────────────────
        # Transpose first so softmax runs over the last dim (axis=-1), avoiding
        # the softmax_axis optimizer's Reshape rewrite that breaks ORT gradient builder.
        # attn_pool(x): (B, S, D) → (B, S, 1) → transpose → (B, 1, S)
        pool_scores = self.attn_pool(x).transpose(1, 2)  # (B, 1, S)
        pool_w = F.softmax(pool_scores, dim=-1)  # (B, 1, S) — softmax over S
        x = torch.matmul(pool_w, x).squeeze(1)  # (B, 1, D) → (B, D)

        return self.classifier(x)  # (B, num_classes)
