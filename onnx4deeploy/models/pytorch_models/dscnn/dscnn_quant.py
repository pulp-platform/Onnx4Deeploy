# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Brevitas-quantized DS-CNN for the MLperf Tiny KWS benchmark.

Mirrors the FP32 DS-CNN in ``dscnn.py`` but with Brevitas QuantConv2d /
QuantLinear / QuantReLU substitutions. No residual adds — purely
feed-forward depthwise-separable blocks. Designed to be
``DeepQuant.exportBrevitas``-compatible and to lower to Deeploy's
RequantizedConv / RequantizedGemm via ``qcdq_to_deeploy``.
"""

import brevitas.nn as qnn
import torch
import torch.nn as nn
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Int32Bias

_QUANT_KW = dict(
    weight_quant=Int8WeightPerTensorFloat,
    bias_quant=Int32Bias,
    output_quant=Int8ActPerTensorFloat,
    return_quant_tensor=True,
)


class QuantDSConvBlock(nn.Module):
    """Brevitas-quantized depthwise-separable block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = qnn.QuantConv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch,
            bias=True,
            **_QUANT_KW,
        )
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.relu_dw = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        self.pw = qnn.QuantConv2d(in_ch, out_ch, kernel_size=1, bias=True, **_QUANT_KW)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.relu_pw = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu_dw(self.bn_dw(self.dw(x)))
        x = self.relu_pw(self.bn_pw(self.pw(x)))
        return x


class QuantDSCNN(nn.Module):
    """Brevitas-quantized DS-CNN (MLperf Tiny KWS).

    Functionally identical to ``dscnn.DSCNN`` modulo int8 quantization.
    """

    def __init__(
        self,
        num_classes: int = 12,
        n_time: int = 49,
        n_freq: int = 10,
        base_channels: int = 64,
        n_ds_blocks: int = 4,
    ):
        super().__init__()
        self.n_time = n_time
        self.n_freq = n_freq

        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=True
        )

        self.conv_stem = qnn.QuantConv2d(
            1,
            base_channels,
            kernel_size=(min(10, n_time), min(4, n_freq)),
            stride=2,
            padding=0,
            bias=True,
            **_QUANT_KW,
        )
        self.bn_stem = nn.BatchNorm2d(base_channels)
        self.relu_stem = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        self.ds_blocks = nn.Sequential(
            *[QuantDSConvBlock(base_channels, base_channels) for _ in range(n_ds_blocks)]
        )

        # Pool + classifier: torch.mean(dim=(2,3)) → ReduceMean (Deeploy-supported).
        self.pool_dq = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc_iq = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.fc = qnn.QuantLinear(
            base_channels,
            num_classes,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
            return_quant_tensor=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        x = self.relu_stem(self.bn_stem(self.conv_stem(x)))
        x = self.ds_blocks(x)
        x = self.pool_dq(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = self.flatten(x)
        x = self.fc_iq(x)
        x = self.fc(x)
        return x


def quant_dscnn_s(num_classes: int = 12, n_time: int = 49, n_freq: int = 10) -> QuantDSCNN:
    """Brevitas-quantized DS-CNN-S (MLperf Tiny KWS reference, base_channels=64)."""
    return QuantDSCNN(
        num_classes=num_classes,
        n_time=n_time,
        n_freq=n_freq,
        base_channels=64,
        n_ds_blocks=4,
    )


def quant_dscnn_xs(num_classes: int = 12, n_time: int = 49, n_freq: int = 10) -> QuantDSCNN:
    """Brevitas-quantized DS-CNN-XS (PULP-deployable, base_channels=16)."""
    return QuantDSCNN(
        num_classes=num_classes,
        n_time=n_time,
        n_freq=n_freq,
        base_channels=16,
        n_ds_blocks=4,
    )
