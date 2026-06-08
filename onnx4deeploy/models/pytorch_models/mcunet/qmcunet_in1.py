# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Brevitas INT8 quantized MCUNet-In1.

Mirrors the fp32 ProxylessNASNets MCUNet-In1 architecture (96×96 RGB, 10 classes)
using Brevitas QuantConv2d / QuantLinear so that Export4Deeploy.exportBrevitas
can produce the Deeploy-compatible ONNX (RequantShift nodes).

No BatchNorm — weights absorb BN parameters after QAT training or are initialised
to random values for a test export.
"""

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant.scaled_int import (
    Int8ActPerTensorFloat,
    Int32Bias,
    Int8WeightPerChannelFloat,
)

# ---------------------------------------------------------------------------
# Quantisation parameter dicts
# ---------------------------------------------------------------------------

# Convolutions that are followed by an activation (expand / depthwise).
# output_quant + return_quant_tensor=True so the output is a QuantTensor;
# the activation (Hardtanh) unwraps it to a float via __torch_function__.
_CONV_ACT = dict(
    weight_bit_width=8,
    bias_quant=Int32Bias,
    input_quant=Int8ActPerTensorFloat,
    weight_quant=Int8WeightPerChannelFloat,
    output_quant=Int8ActPerTensorFloat,
    return_quant_tensor=True,
)

# Projection convolutions (no activation after, end of MB block).
# output_quant produces a RequantShift (int8→int8); return_quant_tensor=True
# so that torch.mean on the QuantTensor output triggers the scalar Dequant
# (int8→float) before the FP32 classifier head.
_CONV_PROJ = dict(
    weight_bit_width=8,
    bias_quant=Int32Bias,
    input_quant=Int8ActPerTensorFloat,
    weight_quant=Int8WeightPerChannelFloat,
    output_quant=Int8ActPerTensorFloat,
    return_quant_tensor=True,
)



# ---------------------------------------------------------------------------
# Building-blocks
# ---------------------------------------------------------------------------

class _QConvAct(nn.Module):
    """QuantConv2d + optional ReLU6 (no BatchNorm)."""

    def __init__(self, in_ch, out_ch, k, stride=1, groups=1, has_act=True):
        super().__init__()
        self.conv = qnn.QuantConv2d(
            in_ch, out_ch, k,
            stride=stride, padding=k // 2, groups=groups,
            bias=True, **(_CONV_ACT if has_act else _CONV_PROJ),
        )
        self.act = nn.Hardtanh(0.0, 6.0) if has_act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


class _QMBBlock(nn.Module):
    """Quantized MobileInverted block (no residual shortcut).

    When expand_ratio == 1 there is no expand conv; the depthwise conv
    operates directly on in_ch channels.
    """

    def __init__(self, in_ch, out_ch, k, stride, expand_ratio):
        super().__init__()
        mid = in_ch * expand_ratio

        self.has_expand = expand_ratio != 1
        if self.has_expand:
            self.expand_conv = qnn.QuantConv2d(
                in_ch, mid, 1, bias=True, **_CONV_ACT,
            )
            self.expand_act = nn.Hardtanh(0.0, 6.0)

        self.depth_conv = qnn.QuantConv2d(
            mid, mid, k,
            stride=stride, padding=k // 2, groups=mid,
            bias=True, **_CONV_ACT,
        )
        self.depth_act = nn.Hardtanh(0.0, 6.0)

        self.proj_conv = qnn.QuantConv2d(mid, out_ch, 1, bias=True, **_CONV_PROJ)

    def forward(self, x):
        if self.has_expand:
            x = self.expand_act(self.expand_conv(x))
        x = self.depth_act(self.depth_conv(x))
        return self.proj_conv(x)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

# (in_ch, out_ch, kernel, stride, expand_ratio) for each of the 14 blocks.
_BLOCK_CONFIGS = [
    (16,  8, 3, 1, 1),   # 0
    ( 8, 16, 3, 2, 4),   # 1
    (16, 16, 3, 1, 3),   # 2
    (16, 24, 7, 2, 3),   # 3
    (24, 24, 3, 1, 5),   # 4
    (24, 40, 3, 2, 5),   # 5
    (40, 40, 7, 1, 4),   # 6
    (40, 48, 5, 2, 4),   # 7
    (48, 48, 3, 1, 3),   # 8
    (48, 48, 3, 1, 4),   # 9
    (48, 96, 7, 1, 5),   # 10
    (96, 96, 5, 1, 4),   # 11
    (96, 96, 5, 1, 4),   # 12
    (96,160, 3, 1, 6),   # 13
]


class QMCUNetIn1(nn.Module):
    """INT8 Brevitas-quantized MCUNet-In1 (CIFAR-10, 96×96 RGB).

    Args:
        num_classes: Number of output classes (default 10).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Input quantisation gate.
        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
            bit_width=8,
        )
        # First conv: 3 → 16, k=3, stride=2.
        self.first_conv = _QConvAct(3, 16, 3, stride=2, has_act=True)
        # 14 inverted-residual blocks.
        self.blocks = nn.ModuleList([
            _QMBBlock(*cfg) for cfg in _BLOCK_CONFIGS
        ])
        # Global average pool then classifier.
        self.fc = nn.Linear(160, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        # Global average pooling over H and W.
        x = torch.mean(x, dim=[2, 3])   # (B, 160)
        return self.fc(x)
