# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Brevitas-quantized FC Autoencoder for the MLperf Tiny AD benchmark.

Mirrors the FP32 FCAutoencoder in ``autoencoder.py`` but with Brevitas
QuantLinear / QuantReLU substitutions. No BatchNorm and no residual
adds — the simplest possible quantization recipe. Designed to be
``DeepQuant.exportBrevitas``-compatible and to lower to Deeploy's
RequantizedGemm via the ``qcdq_to_deeploy`` adapter pipeline.
"""

from typing import List

import brevitas.nn as qnn
import torch
import torch.nn as nn
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Int32Bias

_LINEAR_KW = dict(
    weight_quant=Int8WeightPerTensorFloat,
    bias_quant=Int32Bias,
    output_quant=Int8ActPerTensorFloat,
    return_quant_tensor=True,
)


class QuantFCAutoencoder(nn.Module):
    """Brevitas-quantized symmetric FC autoencoder (MLperf Tiny AD)."""

    def __init__(self, input_dim: int = 128, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 128]

        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=True
        )

        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(qnn.QuantLinear(in_dim, h, bias=True, **_LINEAR_KW))
            encoder_layers.append(qnn.QuantReLU(bit_width=8, return_quant_tensor=True))
            in_dim = h
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder, linear final output)
        decoder_layers = []
        dims = list(reversed(hidden_dims)) + [input_dim]
        for i, out_dim in enumerate(dims):
            is_last = i == len(dims) - 1
            if is_last:
                decoder_layers.append(
                    qnn.QuantLinear(
                        in_dim,
                        out_dim,
                        bias=True,
                        weight_quant=Int8WeightPerTensorFloat,
                        bias_quant=Int32Bias,
                        output_quant=Int8ActPerTensorFloat,
                        return_quant_tensor=False,
                    )
                )
            else:
                decoder_layers.append(qnn.QuantLinear(in_dim, out_dim, bias=True, **_LINEAR_KW))
                decoder_layers.append(qnn.QuantReLU(bit_width=8, return_quant_tensor=True))
            in_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        z = self.encoder(x)
        return self.decoder(z)


def quant_autoencoder_mlperf(input_dim: int = 128) -> QuantFCAutoencoder:
    """Brevitas-quantized MLperf Tiny AD reference autoencoder."""
    return QuantFCAutoencoder(input_dim=input_dim, hidden_dims=[128, 128, 128])


def quant_autoencoder_tiny(input_dim: int = 128) -> QuantFCAutoencoder:
    """Brevitas-quantized tiny FC autoencoder for PULP embedded deployment."""
    return QuantFCAutoencoder(input_dim=input_dim, hidden_dims=[64, 32, 64])
