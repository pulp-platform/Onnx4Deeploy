# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Lightweight CNN PyTorch model."""

from .lightweight_cnn import LightweightCNN
from .qlite_cnn import QLiteCNN

__all__ = ["LightweightCNN", "QLiteCNN"]
