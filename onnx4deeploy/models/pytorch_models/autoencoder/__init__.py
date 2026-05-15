# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""FC Autoencoder models for ONNX export."""

from .autoencoder import FCAutoencoder, autoencoder_mlperf, autoencoder_tiny

# Brevitas-quantized FC Autoencoder (MLperf Tiny AD). Imported lazily so that
# environments without brevitas don't fail at package import time.
try:
    from .autoencoder_quant import (
        QuantFCAutoencoder,
        quant_autoencoder_mlperf,
        quant_autoencoder_tiny,
    )

    __all__ = [
        "FCAutoencoder",
        "autoencoder_mlperf",
        "autoencoder_tiny",
        "QuantFCAutoencoder",
        "quant_autoencoder_mlperf",
        "quant_autoencoder_tiny",
    ]
except ImportError:
    __all__ = ["FCAutoencoder", "autoencoder_mlperf", "autoencoder_tiny"]
