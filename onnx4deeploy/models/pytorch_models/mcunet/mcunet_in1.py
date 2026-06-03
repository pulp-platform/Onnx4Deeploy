# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MCUNet-In1 architecture reconstructed from weight analysis.

Architecture spec (derived from MCUNet-In1_CIFAR10.pth):
- Input: (B, 3, 96, 96)
- first_conv: ConvLayer(3→16, k=3, stride=2) → 48×48
- 14 MobileInvertedResidualBlocks (no shortcuts)
- Global average pool → Linear(160, 10)
"""

import copy
import torch.nn as nn
from mcunet.tinynas.nn.networks.proxyless_nets import ProxylessNASNets

# Architecture config derived from weight key shape analysis of MCUNet-In1_CIFAR10.pth.
# Strides chosen to reduce 96×96 → 3×3 before global avg pool (5 stride-2 ops total).
_MCUNET_IN1_CONFIG = {
    "name": "ProxylessNASNets",
    "bn": {"momentum": 0.1, "eps": 1e-3},
    "first_conv": {
        "name": "ConvLayer",
        "in_channels": 3,
        "out_channels": 16,
        "kernel_size": 3,
        "stride": 2,
        "dilation": 1,
        "groups": 1,
        "bias": False,
        "use_bn": True,
        "act_func": "relu6",
        "ops_order": "weight_bn_act",
    },
    "blocks": [
        # Block 0: expand=1 (no inverted_bottleneck), 48×48 → 48×48
        {
            "name": "MobileInvertedResidualBlock",
            "mobile_inverted_conv": {
                "name": "MBInvertedConvLayer",
                "in_channels": 16, "out_channels": 8,
                "kernel_size": 3, "stride": 1, "expand_ratio": 1,
                "mid_channels": None, "act_func": "relu6", "use_se": False,
            },
            "shortcut": None,
        },
        # Block 1: expand=4, stride=2 → 24×24
        {
            "name": "MobileInvertedResidualBlock",
            "mobile_inverted_conv": {
                "name": "MBInvertedConvLayer",
                "in_channels": 8, "out_channels": 16,
                "kernel_size": 3, "stride": 2, "expand_ratio": 4,
                "mid_channels": None, "act_func": "relu6", "use_se": False,
            },
            "shortcut": None,
        },
        # Block 2: expand=3, 24×24 → 24×24
        {
            "name": "MobileInvertedResidualBlock",
            "mobile_inverted_conv": {
                "name": "MBInvertedConvLayer",
                "in_channels": 16, "out_channels": 16,
                "kernel_size": 3, "stride": 1, "expand_ratio": 3,
                "mid_channels": None, "act_func": "relu6", "use_se": False,
            },
            "shortcut": None,
        },
        # Block 3: expand=3, k=7, stride=2 → 12×12
        {
            "name": "MobileInvertedResidualBlock",
            "mobile_inverted_conv": {
                "name": "MBInvertedConvLayer",
                "in_channels": 16, "out_channels": 24,
                "kernel_size": 7, "stride": 2, "expand_ratio": 3,
                "mid_channels": None, "act_func": "relu6", "use_se": False,
            },
            "shortcut": None,
        },
        # Block 4: expand=5, 12×12 → 12×12
        {
            "name": "MobileInvertedResidualBlock",
            "mobile_inverted_conv": {
                "name": "MBInvertedConvLayer",
                "in_channels": 24, "out_channels": 24,
                "kernel_size": 3, "stride": 1, "expand_ratio": 5,
                "mid_channels": None, "act_func": "relu6", "use_se": False,
            },
            "shortcut": None,
        },
        # Block 5: expand=5, stride=2 → 6×6
        {
            "name": "MobileInvertedResidualBlock",
            "mobile_inverted_conv": {
                "name": "MBInvertedConvLayer",
                "in_channels": 24, "out_channels": 40,
                "kernel_size": 3, "stride": 2, "expand_ratio": 5,
                "mid_channels": None, "act_func": "relu6", "use_se": False,
            },
            "shortcut": None,
        },
        # Block 6: expand=4, k=7, 6×6 → 6×6
        {
            "name": "MobileInvertedResidualBlock",
            "mobile_inverted_conv": {
                "name": "MBInvertedConvLayer",
                "in_channels": 40, "out_channels": 40,
                "kernel_size": 7, "stride": 1, "expand_ratio": 4,
                "mid_channels": None, "act_func": "relu6", "use_se": False,
            },
            "shortcut": None,
        },
        # Block 7: expand=4, k=5, stride=2 → 3×3
        {
            "name": "MobileInvertedResidualBlock",
            "mobile_inverted_conv": {
                "name": "MBInvertedConvLayer",
                "in_channels": 40, "out_channels": 48,
                "kernel_size": 5, "stride": 2, "expand_ratio": 4,
                "mid_channels": None, "act_func": "relu6", "use_se": False,
            },
            "shortcut": None,
        },
        # Blocks 8-13: all stride=1, 3×3 → 3×3
        {
            "name": "MobileInvertedResidualBlock",
            "mobile_inverted_conv": {
                "name": "MBInvertedConvLayer",
                "in_channels": 48, "out_channels": 48,
                "kernel_size": 3, "stride": 1, "expand_ratio": 3,
                "mid_channels": None, "act_func": "relu6", "use_se": False,
            },
            "shortcut": None,
        },
        {
            "name": "MobileInvertedResidualBlock",
            "mobile_inverted_conv": {
                "name": "MBInvertedConvLayer",
                "in_channels": 48, "out_channels": 48,
                "kernel_size": 3, "stride": 1, "expand_ratio": 4,
                "mid_channels": None, "act_func": "relu6", "use_se": False,
            },
            "shortcut": None,
        },
        {
            "name": "MobileInvertedResidualBlock",
            "mobile_inverted_conv": {
                "name": "MBInvertedConvLayer",
                "in_channels": 48, "out_channels": 96,
                "kernel_size": 7, "stride": 1, "expand_ratio": 5,
                "mid_channels": None, "act_func": "relu6", "use_se": False,
            },
            "shortcut": None,
        },
        {
            "name": "MobileInvertedResidualBlock",
            "mobile_inverted_conv": {
                "name": "MBInvertedConvLayer",
                "in_channels": 96, "out_channels": 96,
                "kernel_size": 5, "stride": 1, "expand_ratio": 4,
                "mid_channels": None, "act_func": "relu6", "use_se": False,
            },
            "shortcut": None,
        },
        {
            "name": "MobileInvertedResidualBlock",
            "mobile_inverted_conv": {
                "name": "MBInvertedConvLayer",
                "in_channels": 96, "out_channels": 96,
                "kernel_size": 5, "stride": 1, "expand_ratio": 4,
                "mid_channels": None, "act_func": "relu6", "use_se": False,
            },
            "shortcut": None,
        },
        {
            "name": "MobileInvertedResidualBlock",
            "mobile_inverted_conv": {
                "name": "MBInvertedConvLayer",
                "in_channels": 96, "out_channels": 160,
                "kernel_size": 3, "stride": 1, "expand_ratio": 6,
                "mid_channels": None, "act_func": "relu6", "use_se": False,
            },
            "shortcut": None,
        },
    ],
    "feature_mix_layer": None,
    "classifier": {
        "name": "LinearLayer",
        "in_features": 160,
        "out_features": 10,
        "bias": True,
        "use_bn": False,
        "act_func": None,
        "dropout_rate": 0,
        "ops_order": "weight_bn_act",
    },
}


def build_mcunet_in1(num_classes: int = 10) -> nn.Module:
    """Build MCUNet-In1 model for the given number of classes.

    Returns a ProxylessNASNets model configured for 96×96 RGB input.
    The classifier head outputs `num_classes` logits.
    """
    cfg = copy.deepcopy(_MCUNET_IN1_CONFIG)
    cfg["classifier"]["out_features"] = num_classes
    return ProxylessNASNets.build_from_config(cfg)
