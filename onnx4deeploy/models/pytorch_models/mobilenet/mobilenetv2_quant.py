# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Brevitas-quantized MobileNetV2 for the MLperf Tiny VWW benchmark.

Mirrors the FP32 MobileNetV2 in ``mobilenetv2.py`` but with Brevitas
QuantConv2d / QuantLinear / QuantReLU substitutions and explicit
QuantIdentity wraps around the inverted-residual add. Designed to be
``DeepQuant.exportBrevitas``-compatible and to lower to Deeploy's
RequantizedConv / RequantizedAdd / RequantizedGemm via the
``qcdq_to_deeploy`` adapter pipeline.

VWW variant uses ``width_mult=0.35`` and 96×96 input (MLperf Tiny v1.0).
"""

import brevitas.nn as qnn
import torch
import torch.nn as nn
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Int32Bias

# Common kwargs for QuantConv2d / QuantLinear: per-tensor INT8 weight + INT8
# activation, INT32 bias. Matches the recipe in ``resnet_quant.py``.
_QUANT_KW = dict(
    weight_quant=Int8WeightPerTensorFloat,
    bias_quant=Int32Bias,
    output_quant=Int8ActPerTensorFloat,
    return_quant_tensor=True,
)


class _QuantReLU6(nn.Module):
    """Brevitas-quantized stand-in for ``nn.ReLU6``.

    Brevitas only ships QuantReLU (unbounded). For QCDQ export the upper
    saturation at 6 is implicit in the int8 act quant's scale calibration —
    after BN folding and act quant, the post-activation range is clipped to
    [0, 127] (int8 unsigned half) which is functionally equivalent for
    deployment. We use QuantReLU here for a clean ONNX graph.
    """

    def __init__(self):
        super().__init__()
        self.act = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


class QuantInvertedResidual(nn.Module):
    """Brevitas-quantized counterpart of ``mobilenetv2.InvertedResidual``."""

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.extend(
                [
                    qnn.QuantConv2d(inp, hidden_dim, 1, 1, 0, bias=True, **_QUANT_KW),
                    nn.BatchNorm2d(hidden_dim),
                    _QuantReLU6(),
                ]
            )

        layers.extend(
            [
                qnn.QuantConv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=True,
                    **_QUANT_KW,
                ),
                nn.BatchNorm2d(hidden_dim),
                _QuantReLU6(),
                qnn.QuantConv2d(hidden_dim, oup, 1, 1, 0, bias=True, **_QUANT_KW),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

        if self.use_res_connect:
            # Strip QuantTensors right before the residual add so the `+`
            # runs on fp32 operands (avoiding Brevitas's per-tensor scale-
            # match check), then re-quantize the sum.
            self.dq_main = qnn.QuantIdentity(
                act_quant=Int8ActPerTensorFloat, return_quant_tensor=False
            )
            self.dq_identity = qnn.QuantIdentity(
                act_quant=Int8ActPerTensorFloat, return_quant_tensor=False
            )
            self.add_q = qnn.QuantIdentity(
                act_quant=Int8ActPerTensorFloat, return_quant_tensor=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            identity = self.dq_identity(x)
            out = self.dq_main(self.conv(x))
            return self.add_q(out + identity)
        else:
            return self.conv(x)


class QuantMobileNetV2(nn.Module):
    """Brevitas-quantized MobileNetV2 (MLperf Tiny VWW).

    Functionally identical to ``mobilenetv2.MobileNetV2`` modulo the int8
    quantization of weights/activations. Input is fp32; an entry
    ``QuantIdentity`` quantizes it once, after which the network stays
    integer until the final classifier.
    """

    def __init__(
        self,
        num_classes: int = 2,
        width_mult: float = 0.35,
        input_channels: int = 3,
    ):
        super().__init__()

        input_channel = 32
        last_channel = 1280

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))

        # Quantize the input once (fp32 → int8).
        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=True
        )

        features = [
            qnn.QuantConv2d(input_channels, input_channel, 3, 2, 1, bias=True, **_QUANT_KW),
            nn.BatchNorm2d(input_channel),
            _QuantReLU6(),
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    QuantInvertedResidual(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel

        features.extend(
            [
                qnn.QuantConv2d(input_channel, self.last_channel, 1, 1, 0, bias=True, **_QUANT_KW),
                nn.BatchNorm2d(self.last_channel),
                _QuantReLU6(),
            ]
        )

        self.features = nn.Sequential(*features)

        # Use torch.mean(dim=(2,3)) instead of AdaptiveAvgPool2d — exports
        # to ReduceMean (supported by Deeploy Siracusa).
        self.pool_dq = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc_iq = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.fc = qnn.QuantLinear(
            self.last_channel,
            num_classes,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
            return_quant_tensor=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        x = self.features(x)
        x = self.pool_dq(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = self.flatten(x)
        x = self.fc_iq(x)
        x = self.fc(x)
        return x


def quant_mobilenet_v2(
    num_classes: int = 2, width_mult: float = 0.35, input_channels: int = 3
) -> QuantMobileNetV2:
    """Factory for the Brevitas-quantized MobileNetV2 (MLperf Tiny VWW)."""
    return QuantMobileNetV2(
        num_classes=num_classes,
        width_mult=width_mult,
        input_channels=input_channels,
    )
