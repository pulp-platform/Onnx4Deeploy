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

ONNX export strategy — all-3D:
  nn.Linear applied to a 3-D input (B, S, D) is rewritten as:
    Reshape(x, [B*S, D])  →  Gemm(transB=1)  →  Reshape(out, [B, S, out_dim])
  This generates only 2-D Gemm nodes (no Transpose(perm=[1,0]) nodes) so that
  Deeploy's MatMulLayer.computeShapes does not corrupt intermediate tensor shapes.
  The only Transpose that remains is k.transpose(1, 2) for attention scores, which
  has perm=[0, 2, 1] on a 3-D tensor and is handled correctly by Deeploy.

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

    All Linear layers that receive a 3-D activation (B, S, D) are applied via
    an explicit Reshape → Gemm → Reshape sequence so that the exported ONNX
    graph contains only 2-D Gemm nodes (no perm=[1,0] Transpose nodes that
    cause shape corruption in Deeploy's MatMulLayer.computeShapes).

    The only Transpose that appears is k.transpose(1, 2) in the attention score
    computation; it has perm=[0, 2, 1] on a 3-D tensor and is handled correctly.

    Args:
        num_patches: Number of patches per image (default: 16 for 7×7 on 28×28).
        patch_dim:   Flattened patch size in pixels (default: 49 for 7×7).
        embed_dim:   Token embedding dimension (default: 32).
        ffn_hidden:  Hidden dimension of the feed-forward block (default: 64).
        num_classes: Number of output classes (default: 10 for MNIST).
        use_lora:    If True, attach LoRA A/B adapters to the attention projections
                     (q/k/v/out_proj). The base projection weights are kept; the
                     forward path adds (x · A · B) * (alpha / r). With B initialised
                     to zero the initial output equals the non-LoRA model, so the
                     graph is functionally identical at step 0 and only the A/B
                     parameters need gradients during fine-tuning.
        lora_r:      LoRA rank (default: 4).
        lora_alpha:  LoRA scaling numerator; effective scale = alpha / r (default: 16).
    """

    def __init__(
        self,
        num_patches: int = 16,
        patch_dim: int = 49,
        embed_dim: int = 32,
        ffn_hidden: int = 64,
        num_classes: int = 10,
        use_lora: bool = False,
        lora_r: int = 4,
        lora_alpha: int = 16,
    ):
        super().__init__()

        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.ffn_hidden = ffn_hidden
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

        # ── Optional LoRA adapters on the attention block ──────────────────
        # Shapes are stored already-transposed so the forward path is
        #   x @ A @ B   (not x @ A.t() @ B.t())
        # which avoids any perm=[1,0] Transpose nodes in the exported graph.
        #   A: (D, r)   B: (r, D)
        self.use_lora = use_lora
        if use_lora:
            self.lora_r = lora_r
            self.lora_alpha = lora_alpha
            self.lora_scaling = lora_alpha / lora_r

            self.lora_q_A = nn.Parameter(torch.zeros(embed_dim, lora_r))
            self.lora_q_B = nn.Parameter(torch.zeros(lora_r, embed_dim))
            self.lora_k_A = nn.Parameter(torch.zeros(embed_dim, lora_r))
            self.lora_k_B = nn.Parameter(torch.zeros(lora_r, embed_dim))
            self.lora_v_A = nn.Parameter(torch.zeros(embed_dim, lora_r))
            self.lora_v_B = nn.Parameter(torch.zeros(lora_r, embed_dim))
            self.lora_out_A = nn.Parameter(torch.zeros(embed_dim, lora_r))
            self.lora_out_B = nn.Parameter(torch.zeros(lora_r, embed_dim))

            # Standard LoRA init: A ~ kaiming_uniform, B = 0 → initial delta is 0
            for A in (self.lora_q_A, self.lora_k_A, self.lora_v_A, self.lora_out_A):
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            # B already zeros

    # ------------------------------------------------------------------
    # Helper: apply an nn.Linear to a 3-D tensor via 2-D reshape.
    # ONNX export: Reshape → Gemm(transB=1) → Reshape  (no Transpose node)
    # ------------------------------------------------------------------

    @staticmethod
    def _lin3d(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        """Apply *linear* to a (B, S, D) tensor without generating a Transpose node.

        The forward path is:
            x_2d  = x.reshape(B*S, D)          # Reshape
            out2d = linear(x_2d)                # Gemm(transB=1)  [2-D input → 2-D Gemm]
            out   = out2d.reshape(B, S, -1)     # Reshape
        """
        B, S, D = x.shape
        return linear(x.reshape(B * S, D)).reshape(B, S, -1)

    # ------------------------------------------------------------------
    # Helper: LoRA delta path applied to a (B, S, D) tensor via 2-D reshape.
    # A is stored as (D_in, r) and B as (r, D_out) so the forward is
    #   (x_2d @ A) @ B * scaling
    # which produces only 2-D MatMul nodes — no perm=[1,0] Transpose, matching
    # the same export-friendly pattern used by _lin3d above.
    # ------------------------------------------------------------------

    @staticmethod
    def _lora3d(
        x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, scaling: float
    ) -> torch.Tensor:
        B_, S, D = x.shape
        x2d = x.reshape(B_ * S, D)
        out2d = (x2d @ A) @ B
        return out2d.reshape(B_, S, -1) * scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_patches, patch_dim) — pre-extracted, flattened patches.

        Returns:
            logits: (B, num_classes)
        """
        B, S, _ = x.shape
        self.embed_dim

        # ── Patch embedding ────────────────────────────────────────────────
        # Reshape([B,S,patch_dim]→[B*S,patch_dim]) → Gemm → Reshape([B*S,D]→[B,S,D])
        x = self._lin3d(x, self.patch_embed)  # (B, S, D)

        # ── Transformer encoder block (pre-norm, residual) ──────────────────
        residual = x
        x = self.norm1(x)  # LayerNorm over D — operates on last dim, no Transpose

        # Q/K/V projections via 2-D Gemm (reshape in/out)
        q = self._lin3d(x, self.q_proj)  # (B, S, D)
        k = self._lin3d(x, self.k_proj)  # (B, S, D)
        v = self._lin3d(x, self.v_proj)  # (B, S, D)

        # LoRA delta on Q/K/V — same export-friendly Reshape→2-D MatMul→Reshape pattern
        if self.use_lora:
            q = q + self._lora3d(x, self.lora_q_A, self.lora_q_B, self.lora_scaling)
            k = k + self._lora3d(x, self.lora_k_A, self.lora_k_B, self.lora_scaling)
            v = v + self._lora3d(x, self.lora_v_A, self.lora_v_B, self.lora_scaling)

        # Attention scores: (B, S, D) × (B, D, S) → (B, S, S)
        # k.transpose(1, 2) → perm=[0,2,1] on 3-D tensor — correct in Deeploy
        attn = torch.bmm(q, k.transpose(1, 2)) * self.inv_scale
        attn = F.softmax(attn, dim=-1)  # (B, S, S)

        # Context: (B, S, S) × (B, S, D) → (B, S, D)
        ctx = torch.bmm(attn, v)
        x_out = self._lin3d(ctx, self.out_proj)  # (B, S, D)
        if self.use_lora:
            x_out = x_out + self._lora3d(
                ctx, self.lora_out_A, self.lora_out_B, self.lora_scaling
            )
        x = x_out + residual

        # ── Feed-forward block (pre-norm, residual) ─────────────────────────
        residual = x
        x = self.norm2(x)
        # ff1: [B,S,D] → [B,S,ffn_hidden] via 2-D Gemm
        x = self._lin3d(x, self.ff1)  # (B, S, ffn_hidden)
        x = F.relu(x)
        # ff2: [B,S,ffn_hidden] → [B,S,D] via 2-D Gemm
        x = self._lin3d(x, self.ff2)  # (B, S, D)
        x = x + residual

        # ── Attention pooling over patches → (B, D) ─────────────────────────
        # attn_pool: [B,S,D] → [B,S,1] via 2-D Gemm, then transpose → [B,1,S]
        pool_scores = self._lin3d(x, self.attn_pool).transpose(1, 2)  # (B, 1, S)
        pool_w = F.softmax(pool_scores, dim=-1)  # (B, 1, S)
        x = torch.matmul(pool_w, x).squeeze(1)  # (B, D)

        # ── Classifier (2-D input — standard nn.Linear, no reshape needed) ──
        return self.classifier(x)  # (B, num_classes)
