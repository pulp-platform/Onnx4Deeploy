# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""ResNet models for ONNX export."""

from .resnet import BasicBlock, Bottleneck, ResNet, ResNet8, resnet8, resnet18, resnet34, resnet50

# Brevitas-quantized ResNet8 (MLperf Tiny IC). Imported lazily so that
# environments without brevitas don't fail at package import time.
try:
    from .resnet_quant import QuantResNet8, quant_resnet8

    __all__ = [
        "ResNet",
        "BasicBlock",
        "Bottleneck",
        "ResNet8",
        "resnet8",
        "resnet18",
        "resnet34",
        "resnet50",
        "QuantResNet8",
        "quant_resnet8",
    ]
except ImportError:
    __all__ = [
        "ResNet",
        "BasicBlock",
        "Bottleneck",
        "ResNet8",
        "resnet8",
        "resnet18",
        "resnet34",
        "resnet50",
    ]
