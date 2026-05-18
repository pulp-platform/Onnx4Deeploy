# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
SpeechNet — Deployment-ready variant for Deeploy on PULP MCUs.

Based on the SpeechNet architecture from:
  Spacone et al., "SilentWear: an Ultra-Low Power Wearable System for
  EMG-based Silent Speech Recognition", arXiv: 2603.02847.

Differences from the upstream SilentWear SpeechNet:
  - Input is 4-D (B, 1, C, T) directly; no unsqueeze in forward().
  - MaxPool2d replaced by AvgPool2d for Deeploy tiling/gradient compatibility.
  - No Dropout (deployment / inference mode).

Paper default configuration (5 blocks):
  Input:   (1, 1, 14, 700)   14 EMG channels, 700 samples (1.4 s @ 500 Hz)
  Block 0: Conv2d(1, 8,  k=(1,4),  pad=(0,2))  -> BN -> ReLU -> AvgPool(1,8)
  Block 1: Conv2d(8, 16, k=(1,16), pad=(0,8))  -> BN -> ReLU -> AvgPool(1,4)
  Block 2: Conv2d(16,16, k=(1,8),  pad=(0,4))  -> BN -> ReLU -> AvgPool(1,4)
  Block 3: Conv2d(16,32, k=(7,1),  pad=(0,0))  -> BN -> ReLU -> AvgPool(1,1)
  Block 4: Conv2d(32,32, k=(7,1),  pad=(0,0))  -> BN -> ReLU -> AvgPool(1,1)
  Global:  AdaptiveAvgPool2d(1,1) -> Flatten -> Linear(32, num_classes)
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class SpeechNetDeploy(nn.Module):
    """
    Deployment-ready SpeechNet for Deeploy.

    Parameters
    ----------
    num_channels : int
        Number of EMG input channels (H dimension). Default: 14.
    time_steps : int
        Number of time samples (W dimension). Default: 700.
    num_classes : int
        Number of output classes. Default: 9 (8 commands + rest).
    blocks_config : list of dict, optional
        Per-block configuration. Each dict has keys:
          out_channels (int), kernel (tuple), pool (tuple).
        If None, uses the paper's 5-block default.
    """

    def __init__(
        self,
        num_channels: int = 14,
        time_steps: int = 700,
        num_classes: int = 9,
        blocks_config: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.time_steps = time_steps
        self.num_classes = num_classes

        if blocks_config is None:
            blocks_config = [
                dict(out_channels=8, kernel=(1, 4), pool=(1, 8)),
                dict(out_channels=16, kernel=(1, 16), pool=(1, 4)),
                dict(out_channels=16, kernel=(1, 8), pool=(1, 4)),
                dict(out_channels=32, kernel=(7, 1), pool=(1, 1)),
                dict(out_channels=32, kernel=(7, 1), pool=(1, 1)),
            ]

        self.blocks = nn.ModuleList()
        in_ch = 1

        for cfg in blocks_config:
            out_ch = cfg["out_channels"]
            k_c, k_t = cfg["kernel"]
            k_c, k_t = int(k_c), int(k_t)
            pool_c, pool_t = cfg.get("pool", (1, 1))
            pool_c, pool_t = int(pool_c), int(pool_t)

            layers: List[nn.Module] = [
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=(k_c, k_t),
                    stride=(1, 1),
                    padding=(0, k_t // 2),
                    bias=True,
                ),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=False),
                nn.AvgPool2d(kernel_size=(pool_c, pool_t), stride=(pool_c, pool_t)),
            ]

            self.blocks.append(nn.Sequential(*layers))
            in_ch = out_ch

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._fc_in = in_ch  # stored as Python int for static ONNX reshape
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, 1, C, T)."""
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        # Static reshape avoiding dynamic Shape ops in ONNX.
        # After GlobalAvgPool2d(1,1): (1, C, 1, 1) → (1, C)
        # Both dims are Python ints known at trace time.
        x = x.reshape(1, self._fc_in)
        x = self.fc(x)
        return x
