# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Brevitas-quantized ResNet8 for the MLperf Tiny IC benchmark.

Mirrors the FP32 ResNet8 in ``resnet.py`` but with Brevitas QuantConv2d /
QuantLinear / QuantReLU substitutions and explicit QuantIdentity wraps around
residual adds. Designed to be ``DeepQuant.exportBrevitas``-compatible.
"""

import torch
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.quant.scaled_int import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int32Bias,
)


# Common kwargs for QuantConv2d / QuantLinear: per-tensor INT8 weight + INT8
# activation, INT32 bias. ``return_quant_tensor=True`` so downstream layers see
# a QuantTensor (carries scale/zp metadata that BN folding + the next quant op
# can absorb).
_QUANT_KW = dict(
    weight_quant=Int8WeightPerTensorFloat,
    bias_quant=Int32Bias,
    output_quant=Int8ActPerTensorFloat,
    return_quant_tensor=True,
)


class QuantBasicBlock(nn.Module):
    """Brevitas-quantized counterpart of ``resnet.BasicBlock``."""

    expansion = 1

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None
    ) -> None:
        super().__init__()
        self.conv1 = qnn.QuantConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            **_QUANT_KW,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        self.conv2 = qnn.QuantConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            **_QUANT_KW,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

        # Wraps the residual add output so it carries a quant tensor into the
        # next stage (lets Brevitas/DeepQuant absorb the add into RequantShift
        # downstream).
        self.add_q = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.add_q(out + identity)
        out = self.relu(out)
        return out


class _QuantDownsample(nn.Module):
    """1×1 stride-S downsample (used inside ``ResNet8`` stages 2/3)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv = qnn.QuantConv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            bias=False,
            **_QUANT_KW,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class QuantResNet8(nn.Module):
    """Brevitas-quantized ResNet8 (MLperf Tiny IC).

    Functionally identical to ``resnet.ResNet8`` modulo the int8 quantization
    of weights/activations. Input is fp32; ``QuantIdentity`` at the front
    quantizes it once, after which the network stays integer until the final
    classifier.
    """

    def __init__(
        self, num_classes: int = 10, input_channels: int = 3, base_channels: int = 16
    ) -> None:
        super().__init__()
        c = base_channels  # 16 by default

        # Quantize the input once (fp32 → int8). All downstream ops consume
        # QuantTensors.
        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=True
        )

        self.conv1 = qnn.QuantConv2d(
            input_channels,
            c,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            **_QUANT_KW,
        )
        self.bn1 = nn.BatchNorm2d(c)
        self.relu = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        self.layer1 = QuantBasicBlock(c, c, stride=1, downsample=None)
        self.layer2 = QuantBasicBlock(
            c, c * 2, stride=2, downsample=_QuantDownsample(c, c * 2, stride=2)
        )
        self.layer3 = QuantBasicBlock(
            c * 2, c * 4, stride=2, downsample=_QuantDownsample(c * 2, c * 4, stride=2)
        )

        # Pool + classifier. Adaptive pool stays vanilla — only the data
        # quantization on entry/exit matters.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = qnn.QuantLinear(
            c * 4,
            num_classes,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
            return_quant_tensor=False,  # final output: dequantize back to fp32
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def quant_resnet8(
    num_classes: int = 10, input_channels: int = 3, base_channels: int = 16
) -> QuantResNet8:
    """Factory for the Brevitas-quantized ResNet8 (MLperf Tiny IC)."""
    return QuantResNet8(
        num_classes=num_classes,
        input_channels=input_channels,
        base_channels=base_channels,
    )
