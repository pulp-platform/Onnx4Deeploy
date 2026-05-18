# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""SpeechNet model for ONNX export (SilentWear EMG silent speech recognition)."""

from .speechnet import SpeechNetDeploy

__all__ = ["SpeechNetDeploy"]
