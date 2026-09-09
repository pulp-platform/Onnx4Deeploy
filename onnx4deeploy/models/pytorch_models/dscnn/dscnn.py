# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""DS-CNN Model for MLperf Tiny Keyword Spotting (KWS) benchmark.

Reference: "Hello Edge: Keyword Spotting on Microcontrollers" — Zhang et al. (2018)
MLperf Tiny v1.0 reference model: DS-CNN-S (Small).

Input:  (N, 1, n_time, n_freq)   — MFCC spectrogram features
Output: (N, num_classes)

Architecture
------------
1. Conv2d stem: kernel=(10,4), 64 channels, stride=2
2. 4 × Depthwise-Separable Conv blocks (DW 3×3 + PW 1×1), BatchNorm + ReLU
3. Global AvgPool → FC classifier

All BatchNorm and ReLU layers use inplace=False for clean ONNX / autograd export.
"""

import torch
import torch.nn as nn


class DSConvBlock(nn.Module):
    """Depthwise-Separable Conv block: DW 3×3 → BN → ReLU → PW 1×1 → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn_dw(self.dw(x)))
        x = self.relu(self.bn_pw(self.pw(x)))
        return x


def _same_padding(size: int, kernel: int, stride: int) -> int:
    """Symmetric padding that reproduces Keras ``padding='same'`` output size.

    Keras emits ``ceil(size / stride)``; this returns the smallest symmetric pad
    ``p`` with ``floor((size + 2p - kernel) / stride) + 1 == ceil(size / stride)``.
    Keras splits its padding asymmetrically (more on the bottom/right); the
    symmetric equivalent shifts the receptive field by at most half a pixel and
    keeps the graph free of a separate Pad node.
    """
    target = -(-size // stride)  # ceil
    p = 0
    while (size + 2 * p - kernel) // stride + 1 < target:
        p += 1
    return p


class DSCNN(nn.Module):
    """
    DS-CNN for keyword spotting.

    Parameters
    ----------
    num_classes : int
        Number of output classes (12 for MLperf Tiny: 10 commands + silence + unknown).
    n_time : int
        Time dimension of input MFCC (default 49 for 1-second @ 25 ms shift).
    n_freq : int
        Frequency dimension of input MFCC (default 10 for 10 mel-filterbanks).
    base_channels : int
        Channel width for DS blocks (64 = DS-CNN-S, 172 = DS-CNN-M).
    n_ds_blocks : int
        Number of depthwise-separable blocks (4 for DS-CNN-S).
    """

    def __init__(
        self,
        num_classes: int = 12,
        n_time: int = 49,
        n_freq: int = 10,
        base_channels: int = 64,
        n_ds_blocks: int = 4,
        stem_padding=0,
    ):
        super().__init__()
        self.n_time = n_time
        self.n_freq = n_freq

        # Stem conv: 1 → base_channels, kernel covers most of time/freq axis.
        # stem_padding="same" reproduces the MLperf Tiny reference, whose stem is
        # Conv2D(64, (10,4), strides=2, padding='same') and therefore emits a
        # ceil(n_time/2) x ceil(n_freq/2) map (25x5 for the 49x10 reference input).
        # The default 0 ("valid") is kept for the legacy PULP-small XS fixtures.
        kernel = (min(10, n_time), min(4, n_freq))
        if stem_padding == "same":
            stem_padding = (
                _same_padding(n_time, kernel[0], 2),
                _same_padding(n_freq, kernel[1], 2),
            )
        self.conv_stem = nn.Conv2d(
            1,
            base_channels,
            kernel_size=kernel,
            stride=2,
            padding=stem_padding,
            bias=False,
        )
        self.bn_stem = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=False)

        # DS blocks (all same channel width for DS-CNN-S)
        self.ds_blocks = nn.Sequential(
            *[DSConvBlock(base_channels, base_channels) for _ in range(n_ds_blocks)]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn_stem(self.conv_stem(x)))
        x = self.ds_blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def dscnn_s(num_classes: int = 12, n_time: int = 49, n_freq: int = 10) -> DSCNN:
    """DS-CNN-S (Small) — MLperf Tiny KWS reference model (~270 K params at base_channels=64)."""
    return DSCNN(
        num_classes=num_classes, n_time=n_time, n_freq=n_freq, base_channels=64, n_ds_blocks=4
    )


def dscnn_xs(num_classes: int = 12, n_time: int = 49, n_freq: int = 10) -> DSCNN:
    """DS-CNN-XS (Extra-Small) — 16 channels, 4 blocks (~10 K params); PULP-deployable."""
    return DSCNN(
        num_classes=num_classes, n_time=n_time, n_freq=n_freq, base_channels=16, n_ds_blocks=4
    )
