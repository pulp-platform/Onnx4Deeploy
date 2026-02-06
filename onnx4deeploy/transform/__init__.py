# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Model transformation utilities."""

from .model_transform import (
    ensure_all_tensor_shapes,
    fix_shared_initializers_by_node_name,
    randomize_layernorm_params,
    randomize_onnx_initializers,
    split_convgrad_nodes,
    type_inference,
)

__all__ = [
    "split_convgrad_nodes",
    "randomize_layernorm_params",
    "fix_shared_initializers_by_node_name",
    "randomize_onnx_initializers",
    "type_inference",
    "ensure_all_tensor_shapes",
]
