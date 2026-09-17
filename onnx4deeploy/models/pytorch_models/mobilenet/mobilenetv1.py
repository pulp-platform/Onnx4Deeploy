# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MobileNetV1 — MLperf Tiny Visual Wake Words (VWW) reference model.

The MLperf Tiny VWW benchmark is MobileNetV1 with width multiplier 0.25 at
96x96x3, two classes (person / no-person), ~213 K parameters and 7.49 M MACs.
(Note: it is *not* MobileNetV2-0.35, which the registry previously listed.)

Topology: a strided 3x3 stem followed by 13 depthwise-separable blocks
(DW 3x3 + BN + ReLU, then PW 1x1 + BN + ReLU), global average pool, FC.

Module/attribute names (``stem``, ``blocks.N.dw`` / ``bn_dw`` / ``pw`` /
``bn_pw``, ``fc``) are chosen to match the parameter names in the training
fixtures already deployed in TrainDeeploy, so regenerated graphs stay
drop-in compatible with the existing test configs.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

# (out_channels_at_width_1.0, stride) for each depthwise-separable block.
_BLOCKS: List[Tuple[int, int]] = [
    (64, 1),
    (128, 2),
    (128, 1),
    (256, 2),
    (256, 1),
    (512, 2),
    (512, 1),
    (512, 1),
    (512, 1),
    (512, 1),
    (512, 1),
    (1024, 2),
    (1024, 1),
]


def _scale(channels: int, width_mult: float, divisor: int = 8) -> int:
    """MobileNet channel rounding: nearest multiple of ``divisor``, never < divisor."""
    scaled = int(channels * width_mult)
    rounded = max(divisor, (scaled + divisor // 2) // divisor * divisor)
    if rounded < 0.9 * scaled:  # never drop more than 10 %
        rounded += divisor
    return rounded


class DWSeparableBlock(nn.Module):
    """Depthwise 3x3 + BN + ReLU, then pointwise 1x1 + BN + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.relu_dw = nn.ReLU(inplace=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.relu_pw = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu_dw(self.bn_dw(self.dw(x)))
        return self.relu_pw(self.bn_pw(self.pw(x)))


class MobileNetV1(nn.Module):
    """MobileNetV1 for small-image classification.

    Input:  (N, input_channels, img_size, img_size)
    Output: (N, num_classes)

    At ``width_mult=0.25``, ``img_size=96``, ``num_classes=2`` this is the
    MLperf Tiny VWW reference model (213,586 trainable params, 7.49 M MACs).
    """

    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 3,
        width_mult: float = 0.25,
    ):
        super().__init__()

        stem_ch = _scale(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, stem_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=False),
        )

        blocks: List[nn.Module] = []
        in_ch = stem_ch
        for out_ch, stride in _BLOCKS:
            out_ch = _scale(out_ch, width_mult)
            blocks.append(DWSeparableBlock(in_ch, out_ch, stride=stride))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def mobilenetv1_vww(num_classes: int = 2, input_channels: int = 3) -> MobileNetV1:
    """MLperf Tiny VWW reference: MobileNetV1-0.25 @ 96x96, 2 classes."""
    return MobileNetV1(num_classes=num_classes, input_channels=input_channels, width_mult=0.25)
