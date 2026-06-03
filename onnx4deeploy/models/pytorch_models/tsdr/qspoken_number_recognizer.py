# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""QSpokenNumberRecognizer – Brevitas INT8 quantized T-SDR.

Follows the same quantization pattern as QSleepConViT:
- QuantConv1d for the input projection.
- QuantMultiheadAttention for self-attention in each encoder layer.
- QuantLinear for FFN layers and the output head.
- QuantIdentity for residual rescaling and requantization.
"""

import math
import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant.scaled_int import (
    Int8ActPerTensorFloat,
    Int32Bias,
    Int8WeightPerChannelFloat,
    Uint8ActPerTensorFloat,
)
from brevitas.quant_tensor import QuantTensor


# ---------------------------------------------------------------------------
# Quantization parameter dicts (match QSleepConViT style)
# ---------------------------------------------------------------------------

_CONV_LIN_PARAMS = {
    "weight_bit_width": 8,
    "bias_quant": Int32Bias,
    "input_quant": Int8ActPerTensorFloat,
    "weight_quant": Int8WeightPerChannelFloat,
    "output_quant": Int8ActPerTensorFloat,
    "return_quant_tensor": True,
}

_ACT_PARAMS = {
    "act_quant": Int8ActPerTensorFloat,
    "return_quant_tensor": True,
    "bit_width": 8,
}

_DEQUANT_PARAMS = {
    "act_quant": Int8ActPerTensorFloat,
    "return_quant_tensor": False,
    "bit_width": 8,
}

_MHA_PARAMS = {
    "in_proj_input_quant": Int8ActPerTensorFloat,
    "in_proj_weight_quant": Int8WeightPerChannelFloat,
    "in_proj_bias_quant": Int32Bias,
    "attn_output_weights_quant": Uint8ActPerTensorFloat,
    "q_scaled_quant": Int8ActPerTensorFloat,
    "k_transposed_quant": Int8ActPerTensorFloat,
    "v_quant": Int8ActPerTensorFloat,
    "out_proj_input_quant": Int8ActPerTensorFloat,
    "out_proj_weight_quant": Int8WeightPerChannelFloat,
    "out_proj_bias_quant": Int32Bias,
    "out_proj_output_quant": Int8ActPerTensorFloat,
    "return_quant_tensor": True,
}


# ---------------------------------------------------------------------------
# Positional encoding (uses register_buffer so it works with ONNX export)
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, fixed_len: int = None):
        super().__init__()
        enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        enc[:, 0::2] = torch.sin(position * div_term)
        enc[:, 1::2] = torch.cos(position * div_term)
        # Store only `fixed_len` positions when known (avoids dynamic slice during FX tracing)
        stored = enc[:fixed_len] if fixed_len is not None else enc
        self._fixed = fixed_len is not None
        self.register_buffer("encoding", stored.unsqueeze(0))  # (1, L, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fixed:
            return x + self.encoding
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :]


# ---------------------------------------------------------------------------
# Quantized encoder layer
# ---------------------------------------------------------------------------

class _QEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 256):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = qnn.QuantMultiheadAttention(
            d_model, nhead,
            dropout=0.0,
            batch_first=True,
            packed_in_proj=False,
            **_MHA_PARAMS,
        )
        self.rescale1 = qnn.QuantIdentity(**_ACT_PARAMS)

        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = qnn.QuantLinear(d_model, dim_feedforward, bias=True, **_CONV_LIN_PARAMS)
        self.act = nn.GELU()
        self.linear2 = qnn.QuantLinear(dim_feedforward, d_model, bias=True, **_CONV_LIN_PARAMS)
        self.rescale2 = qnn.QuantIdentity(**_ACT_PARAMS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with quantized residual
        _x = x
        x_ln = self.norm1(x)
        attn_out, _ = self.mha(x_ln, x_ln, x_ln, need_weights=False)
        attn_out = self.rescale1(attn_out)
        _x = self.rescale1(_x)
        x = attn_out + _x

        # Pre-norm FFN with quantized residual
        _x = x
        x_ln = self.norm2(x)
        ff = self.linear2(self.act(self.linear1(x_ln)))
        ff = self.rescale2(ff)
        _x = self.rescale2(_x)
        x = ff + _x
        return x


class _QTransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int,
                 num_layers: int, dim_feedforward: int = 256):
        super().__init__()
        self.layers = nn.ModuleList([
            _QEncoderLayer(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Top-level quantized model
# ---------------------------------------------------------------------------

class QSpokenNumberRecognizer(nn.Module):
    """INT8 quantized SpokenNumberRecognizer for ONNX/Deeploy deployment.

    Args:
        num_classes: Number of digit classes (default 10).
        d_model: Transformer hidden dimension (default 128).
        nhead: Number of attention heads (default 8).
        num_layers: Number of encoder layers (default 4).
        max_len: Maximum sequence length for positional encoding.
    """

    def __init__(self, num_classes: int = 10, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 4, max_len: int = 5000,
                 time_steps: int = 101):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(**_ACT_PARAMS)
        self.conv1d = qnn.QuantConv1d(
            80, d_model, kernel_size=3, stride=1, padding=1,
            bias=True, **_CONV_LIN_PARAMS,
        )
        self.act_conv = nn.GELU()
        self.positional_encoding = _PositionalEncoding(d_model, max_len=max_len, fixed_len=time_steps)
        self.rescale_pos = qnn.QuantIdentity(**_ACT_PARAMS)
        self.encoder = _QTransformerEncoderBlock(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
        )
        self.rescale_out = qnn.QuantIdentity(**_ACT_PARAMS)
        self.fc = qnn.QuantLinear(d_model, num_classes, bias=True, **_CONV_LIN_PARAMS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 80, T)
        x = self.input_quant(x)
        x = self.conv1d(x)              # (B, d_model, T) — QuantTensor
        x = self.act_conv(x)            # GELU unwraps QuantTensor → float
        x = x.permute(0, 2, 1)         # (B, T, d_model)
        x = self.positional_encoding(x)
        x = self.rescale_pos(x)         # requantize after positional encoding
        x = self.encoder(x)             # QuantTensor from last rescale
        x = self.rescale_out(x)         # rescale before pooling; QuantTensor
        x = torch.mean(x, dim=1)        # __torch_function__ extracts value → plain Tensor
        x = self.fc(x)                  # QuantLinear requantizes via input_quant
        return x
