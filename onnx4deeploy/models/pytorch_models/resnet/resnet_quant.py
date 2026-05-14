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
    # return a regular Tensor (with an implicit Dequant at the boundary) so
    # downstream residual adds and BN-strip points don't hit Brevitas's
    # "Scaling factors are different" check. Each layer transition becomes
    # Quant→op→Dequant, matching the QCDQ contract Deeploy expects.
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
            # bias=True so Brevitas wires up an Int32Bias quant proxy. The bias
            # starts zero and absorbs BN's beta/running-stats during the Conv+BN
            # fold step in `BaseONNXExporter.export_quantized` (the proxy is
            # already attached, so the fused value gets correctly quantized).
            bias=True,
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
            # bias=True so Brevitas wires up an Int32Bias quant proxy. The bias
            # starts zero and absorbs BN's beta/running-stats during the Conv+BN
            # fold step in `BaseONNXExporter.export_quantized` (the proxy is
            # already attached, so the fused value gets correctly quantized).
            bias=True,
            **_QUANT_KW,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

        # Strip QuantTensors right before the residual add so the `+` runs on
        # fp32 operands (avoiding Brevitas's per-tensor scale-match check),
        # then re-quantize the sum. Each ``QuantIdentity`` here has a real
        # ``act_quant`` so it actually emits a Quant→Dequant pair (one int8
        # round-trip) — that strips the QuantTensor wrapper. Deeploy's
        # PULPAddRequantMergePass folds Dequant→Add→Quant into RequantizedAdd.
        self.dq_main = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        self.dq_identity = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=False
        )
        self.add_q = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.dq_identity(x if self.downsample is None else self.downsample(x))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dq_main(out)

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
            # bias=True so Brevitas wires up an Int32Bias quant proxy. The bias
            # starts zero and absorbs BN's beta/running-stats during the Conv+BN
            # fold step in `BaseONNXExporter.export_quantized` (the proxy is
            # already attached, so the fused value gets correctly quantized).
            bias=True,
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
            # bias=True so Brevitas wires up an Int32Bias quant proxy. The bias
            # starts zero and absorbs BN's beta/running-stats during the Conv+BN
            # fold step in `BaseONNXExporter.export_quantized` (the proxy is
            # already attached, so the fused value gets correctly quantized).
            bias=True,
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

        # Pool + classifier. ``nn.AdaptiveAvgPool2d(1)`` exports to ONNX
        # ``GlobalAveragePool`` which vanilla Deeploy:devel does not map on
        # Siracusa; ``x.mean(dim=(2,3), keepdim=True)`` exports to ``ReduceMean
        # axes=[2,3]`` (mathematically identical) which IS supported.
        # ``pool_dq`` strips the QuantTensor wrapper so ``.mean()`` (which the
        # QuantTensor type doesn't override) operates on a plain fp32 tensor.
        self.pool_dq = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        self.flatten = nn.Flatten(start_dim=1)
        # Re-quantize before fc (its Int32Bias proxy needs an input scale).
        self.fc_iq = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
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
        x = self.pool_dq(x)
        # Use functional form so even if Brevitas's tracer passes a
        # QuantTensor through here, ``torch.mean`` dispatches via __torch_function__
        # and gets the fp32 view (QuantTensor doesn't define `.mean` method).
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = self.flatten(x)
        x = self.fc_iq(x)  # re-quant for fc's bias proxy
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
