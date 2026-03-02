# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Onnx4Deeploy Core Module.

This module provides the core abstractions for ONNX model export and optimization.
"""

from .base_exporter import BaseONNXExporter, ExportMode
from .onnx_utils import (
    create_test_input_output,
    ensure_output_dir,
    load_train_config,
    print_model_info,
    randomize_onnx_initializers,
    setup_output_paths,
)
from .optimizer_onnx import create_optimizer_onnx, derive_optimizer_dir

__all__ = [
    # Base classes
    "BaseONNXExporter",
    "ExportMode",
    # Optimizer ONNX generation
    "create_optimizer_onnx",
    "derive_optimizer_dir",
    # Utility functions
    "load_train_config",
    "create_test_input_output",
    "randomize_onnx_initializers",
    "print_model_info",
    "setup_output_paths",
    "ensure_output_dir",
]
