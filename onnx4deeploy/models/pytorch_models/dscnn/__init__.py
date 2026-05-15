# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""DS-CNN models for ONNX export."""

from .dscnn import DSCNN, DSConvBlock, dscnn_s, dscnn_xs

# Brevitas-quantized DS-CNN (MLperf Tiny KWS). Imported lazily so that
# environments without brevitas don't fail at package import time.
try:
    from .dscnn_quant import QuantDSCNN, QuantDSConvBlock, quant_dscnn_s, quant_dscnn_xs

    __all__ = [
        "DSCNN",
        "DSConvBlock",
        "dscnn_s",
        "dscnn_xs",
        "QuantDSCNN",
        "QuantDSConvBlock",
        "quant_dscnn_s",
        "quant_dscnn_xs",
    ]
except ImportError:
    __all__ = ["DSCNN", "DSConvBlock", "dscnn_s", "dscnn_xs"]
