"""Utility functions for ONNX model manipulation."""

from .node_naming import make_c_name, rename_and_save_onnx, rename_nodes, rename_onnx_nodes

__all__ = [
    "make_c_name",
    "rename_onnx_nodes",
    "rename_and_save_onnx",
    "rename_nodes",
]
