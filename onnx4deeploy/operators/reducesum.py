# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""ReduceSum operator test implementation."""

from typing import Any, Dict, Tuple

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class ReduceSumOperatorTest(BaseOperatorTest):
    """Test generator for ONNX ReduceSum operator."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.axes = None
        self.keepdims = 1

    def get_operator_name(self) -> str:
        return "ReduceSum"

    def load_config(self) -> Dict[str, Any]:
        """Load ReduceSum-specific configuration."""
        config = super().load_config()

        reducesum_config = config.get("reducesum", {})
        self.input_shape = tuple(reducesum_config["input_shape"])
        self.axes = reducesum_config.get("axes", None)
        self.keepdims = reducesum_config.get("keepdims", 1)

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random input data."""
        return {
            "input": np.random.randn(*self.input_shape).astype(np.float32) * 0.1,
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for ReduceSum operator."""
        # Input tensor
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, self.input_shape)

        # Output shape
        output_shape = self._compute_output_shape()
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        # ReduceSum node (opset 11 uses axes as attribute)
        node_attributes = {"keepdims": self.keepdims}
        if self.axes is not None:
            node_attributes["axes"] = self.axes

        reducesum_node = helper.make_node(
            "ReduceSum",
            inputs=["input"],
            outputs=["output"],
            name="reducesum_node",
            **node_attributes,
        )

        # Graph
        graph = helper.make_graph(
            [reducesum_node], "reducesum_graph", [input_tensor], [output_tensor]
        )

        return graph

    def create_model(self, graph, opset_version: int = 11):
        """Create ONNX model with opset 11 for ReduceSum."""
        return super().create_model(graph, opset_version=11)

    def _compute_output_shape(self) -> Tuple[int, ...]:
        """Compute output shape based on axes and keepdims."""
        output_shape = list(self.input_shape)

        if self.axes is not None:
            # Normalize negative axes
            normalized_axes = [ax if ax >= 0 else ax + len(self.input_shape) for ax in self.axes]
            if self.keepdims:
                for ax in normalized_axes:
                    output_shape[ax] = 1
            else:
                # Remove dimensions
                for ax in sorted(normalized_axes, reverse=True):
                    del output_shape[ax]
        else:
            # Reduce all dimensions
            if self.keepdims:
                output_shape = [1] * len(self.input_shape)
            else:
                output_shape = []

        return tuple(output_shape)

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute expected output using NumPy."""
        input_data = inputs["input"]

        # Convert axes to tuple if it's a list
        axes_tuple = tuple(self.axes) if self.axes is not None else None

        result = np.sum(input_data, axis=axes_tuple, keepdims=bool(self.keepdims))

        return {"output": result}
