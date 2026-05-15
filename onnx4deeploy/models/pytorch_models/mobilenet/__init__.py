# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MobileNet models for ONNX export."""

from .mobilenetv2 import MobileNetV2, mobilenet_v2

# Brevitas-quantized MobileNetV2 (MLperf Tiny VWW). Imported lazily so that
# environments without brevitas don't fail at package import time.
try:
    from .mobilenetv2_quant import QuantMobileNetV2, quant_mobilenet_v2

    __all__ = ["MobileNetV2", "mobilenet_v2", "QuantMobileNetV2", "quant_mobilenet_v2"]
except ImportError:
    __all__ = ["MobileNetV2", "mobilenet_v2"]
