# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""SpokenDigitTransformer (T-SDR) – fp32 version.

Transformer-based model for spoken digit recognition.
Input: (batch_size, 80, time_steps) — 80-band mel spectrogram.
Architecture mirrors on-device-learning/models/SpokenDigitTransformer.py.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, fixed_len: int = None):
        super().__init__()
        enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        enc[:, 0::2] = torch.sin(position * div_term)
        enc[:, 1::2] = torch.cos(position * div_term)
        stored = enc[:fixed_len] if fixed_len is not None else enc
        self._fixed = fixed_len is not None
        self.register_buffer("encoding", stored.unsqueeze(0))  # (1, L, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fixed:
            return x + self.encoding
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :]


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        B, T, D = x.shape
        H, Hd = self.nhead, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, Hd).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, H, Hd).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, H, Hd).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, T, D)
        out = self.out_proj(out)
        return self.proj_drop(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 8,
                 dim_feedforward: int = 192, dropout: float = 0.0):
        super().__init__()
        self.self_attn = MultiheadSelfAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        attn_out = self.self_attn(x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff = self.linear2(self.act(self.linear1(x)))
        x = self.norm2(x + self.dropout2(ff))
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 3, dim_feedforward: int = 192, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class SpokenDigitTransformer(nn.Module):
    """Transformer-based spoken digit classifier (T-SDR).

    Args:
        num_classes: Number of digit classes (default 10).
        d_model: Transformer hidden dimension (default 128).
        nhead: Number of attention heads (default 8).
        num_layers: Number of encoder layers (default 3).
        dim_feedforward: FFN hidden dimension (default d_model + d_model//2 = 192).
        max_len: Maximum sequence length for positional encoding (default 5000).
        fixed_len: If set, stores only this many positional encoding entries,
                   eliminating the dynamic Slice in the ONNX graph.
    """

    def __init__(self, num_classes: int = 10, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 3,
                 dim_feedforward: int = 192, max_len: int = 5000,
                 fixed_len: int = None):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv2d(80, d_model, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
        )
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len, fixed_len=fixed_len)
        self.encoder = TransformerEncoderBlock(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward,
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        # x: (B, 80, T, 1)
        x = self.conv1d(x)                   # (B, d_model, T, 1)
        x = x.squeeze(-1)                    # (B, d_model, T)
        x = x.permute(0, 2, 1)              # (B, T, d_model)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask=mask)       # (B, T, d_model)
        x = x.mean(dim=1)                    # (B, d_model) — global average over T
        return self.fc(x)                    # (B, num_classes)
