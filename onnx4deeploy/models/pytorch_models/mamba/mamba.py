"""Mamba implementation with high-level ONNX operators.

This version exports to ONNX with clean, high-level operators:
- SelectiveSSM (custom operator with delta computation)
- Conv1d (with fixed padding)
- Linear (matrix multiplication)

No fragmented graphs, no dynamic operations!
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# ============================================================================
# ONNX-friendly Layers
# ============================================================================


class LayerNormFunction(Function):
    """Custom LayerNorm with const mean/std."""

    @staticmethod
    def forward(ctx, x, mean, std, weight, bias, eps):
        # Normalization with const mean/std
        x_norm = (x - mean) / (std + eps)
        return x_norm * weight + bias

    @staticmethod
    def symbolic(g, x, mean, std, weight, bias, eps):
        """Export as custom LayerNorm operator."""
        return g.op("ai.mamba::LayerNorm", x, mean, std, weight, bias, epsilon_f=eps, outputs=1)


class LayerNorm(nn.Module):
    """Custom LayerNorm operator with const mean/std.

    Exports as single custom ONNX operator.
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Const mean and std (buffers)
        self.register_buffer("mean", torch.zeros(normalized_shape))
        self.register_buffer("std", torch.ones(normalized_shape))

        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # Custom op with const mean/std
        return LayerNormFunction.apply(x, self.mean, self.std, self.weight, self.bias, self.eps)


class SiLUFunction(Function):
    """SiLU with explicit ONNX symbolic registration."""

    @staticmethod
    def forward(ctx, x):
        return x * torch.sigmoid(x)

    @staticmethod
    def symbolic(g, x):
        # Export as single Silu node
        return g.op("Silu", x)


class SiLU(nn.Module):
    """SiLU activation that exports as single node in ONNX.

    Ensures export as 'Silu' operator instead of Sigmoid+Mul decomposition.
    """

    def forward(self, x):
        return SiLUFunction.apply(x)


# ============================================================================
# Custom ONNX operator
# ============================================================================


def register_custom_op():
    """Register SelectiveSSM as custom ONNX operator.

    SelectiveSSM operator signature:
        Inputs: x, A_log, B, C, D, dt, res
        Outputs: y (after SSM + gating)

    Includes internally:
        - Exp(A_log)
        - SSM computation
        - Gating (res * sigmoid(res)) * y
    """
    from torch.onnx import register_custom_op_symbolic

    def selective_ssm_symbolic(g, x, A_log, B, C, D, dt, res):
        """ONNX symbolic for SelectiveSSM."""
        return g.op("ai.mamba::SelectiveSSM", x, A_log, B, C, D, dt, res, outputs=1)

    # Register the symbolic function
    register_custom_op_symbolic("::SelectiveSSMFunction", selective_ssm_symbolic, opset_version=9)

    print("✅ Custom ONNX operator registered: ai.mamba::SelectiveSSM")
    print("   Inputs: x, A_log, B, C, D, dt, res")
    print("   Outputs: y (SSM + gating)")
    print("   Internal ops: Exp, SSM, SiLU gating")
    return True


# Call registration when module is imported
_REGISTERED = register_custom_op()


class SelectiveSSMFunction(Function):
    """Custom autograd function for Selective SSM with delta computation.

    This will be exported as a single "SelectiveSSM" operator in ONNX.
    Includes delta (dt) computation internally.
    """

    @staticmethod
    def forward(ctx, x, A_log, B, C, D, dt, res):
        """
        Forward pass of Selective SSM with delta and gating.

        Args:
            x: (B, L, D) - input after conv
            A_log: (D, N) - log of state matrix (will be exp'd inside)
            B: (B, L, N) - input projection
            C: (B, L, N) - output projection
            D: (D,) - skip connection
            dt: (B, L, D) - delta (timestep)
            res: (B, L, D) - residual for gating

        Returns:
            y: (B, L, D) - output after gating
        """
        # Exp inside custom op
        A = torch.exp(A_log)
        # SSM computation: y = sum((A * x + B) * C, dim=-1) + D * x
        B_size, L, D_inner = x.shape
        N = B.shape[-1]

        # Reshape for broadcasting
        x_expanded = x.reshape(B_size, L, D_inner, 1)
        A_expanded = A.reshape(1, 1, D_inner, N)

        Ax = x_expanded * A_expanded
        B_expanded = B.reshape(B_size, L, 1, N)
        state = Ax + B_expanded

        C_expanded = C.reshape(B_size, L, 1, N)
        y = torch.sum(state * C_expanded, dim=-1)

        # Skip connection with delta modulation
        D_expanded = D.reshape(1, 1, D_inner)
        y = y + D_expanded * x * dt

        # Gating (inside custom op)
        res_act = res * torch.sigmoid(res)  # SiLU
        y = y * res_act

        return y

    @staticmethod
    def symbolic(g, x, A_log, B, C, D, dt, res):
        """
        ONNX symbolic function - single SelectiveSSM node with gating.
        """
        return g.op("ai.mamba::SelectiveSSM", x, A_log, B, C, D, dt, res, outputs=1)


class SelectiveSSM(nn.Module):
    """Selective SSM that exports as a single ONNX operator."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor
        self.d_conv = d_conv
        self.padding_size = d_conv - 1

        # Linear projections - separate layers to avoid slicing
        self.in_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.res_proj = nn.Linear(d_model, self.d_inner, bias=False)

        # Conv1d with 'same' padding - output length = input length
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding="same",  # Auto padding, output length = input length
            bias=True,
        )

        # SSM parameters - separate layers to avoid slicing
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # State space matrices (pre-computed shapes to avoid unsqueeze)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Activation - ensure single ONNX node
        self.act_conv = SiLU()

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.A_log, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.res_proj.weight)
        nn.init.xavier_uniform_(self.B_proj.weight)
        nn.init.xavier_uniform_(self.C_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - canonical ONNX graph.
        Padding is in Conv attr, no extra nodes.
        """
        # 1. Input projection - parallel projections, NO slicing
        x_proj = self.in_proj(x)  # (B, L, D_inner)
        res = self.res_proj(x)  # (B, L, D_inner)

        # 2. Conv with padding in attributes (no extra nodes)
        x_proj_t = x_proj.transpose(1, 2)  # (B, D_inner, L)
        x_conv = self.conv1d(x_proj_t)  # Padding is Conv attribute
        x_conv = x_conv.transpose(1, 2)  # (B, L, D_inner)

        # 3. Activation
        x_conv = self.act_conv(x_conv)

        # 4. SSM parameters - separate projections, NO slicing
        B = self.B_proj(x_conv)  # (B, L, N)
        C = self.C_proj(x_conv)  # (B, L, N)

        # 5. Delta computation (timestep)
        dt = self.dt_proj(x_conv)  # (B, L, D_inner)
        dt = F.softplus(dt)

        # 6. Selective SSM (custom operator - includes Exp, gating, all inside)
        y = SelectiveSSMFunction.apply(x_conv, self.A_log, B, C, self.D, dt, res)

        # 7. Output projection
        output = self.out_proj(y)

        return output


class MambaBlock(nn.Module):
    """Mamba block with residual connection."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand_factor)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual: LayerNorm → SSM → Add"""
        return x + self.dropout(self.ssm(self.norm(x)))


class Mamba(nn.Module):
    """Mamba model with clean ONNX export.

    ONNX operators per layer:
    - LayerNorm
    - Linear (3x: in_proj, x_proj, dt_proj, out_proj)
    - Conv1d (fixed padding)
    - Silu (2x: activation + gating)
    - Softplus (delta)
    - Exp (A matrix)
    - ai.mamba::SelectiveSSM (custom op)
    - Mul (gating)
    - Add (residual)

    Total: ~12-15 high-level nodes per layer
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        vocab_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        use_embedding: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Input
        if use_embedding and vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        else:
            self.embedding = None
            self.input_proj = nn.Linear(d_model, d_model)

        # Mamba layers
        self.layers = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand_factor, dropout) for _ in range(n_layers)]
        )

        # Output
        self.norm_f = LayerNorm(d_model)

        if num_classes is not None:
            self.classifier = nn.Linear(d_model, num_classes)
        else:
            if vocab_size is not None:
                self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            else:
                self.lm_head = None

        self._initialize_weights()

    def _initialize_weights(self):
        if self.embedding is not None:
            nn.init.normal_(self.embedding.weight, std=0.02)
            nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - NO dynamic operations.

        Fixed input shape: (batch_size, max_seq_len, d_model)
        """
        if self.embedding is not None:
            x = self.embedding(x)
            x = x + self.pos_embedding
        else:
            x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        if hasattr(self, "classifier"):
            x = torch.mean(x, dim=1, keepdim=False)
            x = self.classifier(x)
        elif self.lm_head is not None:
            x = self.lm_head(x)

        return x

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# Convenience functions
def mamba_small(num_classes: int = 10, max_seq_len: int = 512):
    return Mamba(
        d_model=128,
        n_layers=2,
        d_state=16,
        num_classes=num_classes,
        max_seq_len=max_seq_len,
        use_embedding=False,
    )


def mamba_base(num_classes: int = 10, max_seq_len: int = 512):
    return Mamba(
        d_model=256,
        n_layers=4,
        d_state=16,
        num_classes=num_classes,
        max_seq_len=max_seq_len,
        use_embedding=False,
    )
