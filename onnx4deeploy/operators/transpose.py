# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Transpose operator test implementation."""

from typing import Any, Dict, Tuple

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class TransposeOperatorTest(BaseOperatorTest):
    """Test generator for ONNX Transpose operator."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.perm = None

    def get_operator_name(self) -> str:
        return "Transpose"

    def load_config(self) -> Dict[str, Any]:
        """Load Transpose-specific configuration."""
        config = super().load_config()

        transpose_config = config.get("transpose", {})
        self.input_shape = tuple(transpose_config["input_shape"])
        self.perm = tuple(transpose_config["perm"])

        # Validate perm
        if len(self.perm) != len(self.input_shape):
            raise ValueError(
                f"perm length {len(self.perm)} does not match "
                f"input_shape dimensions {len(self.input_shape)}"
            )

        if sorted(self.perm) != list(range(len(self.perm))):
            raise ValueError(f"perm must be a permutation of range({len(self.perm)})")

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random input data."""
        return {
            "input": np.random.randn(*self.input_shape).astype(np.float32) * 0.1,
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for Transpose operator."""
        # Input tensor
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, self.input_shape)

        # Output shape
        output_shape = self._compute_output_shape()
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        # Transpose node
        transpose_node = helper.make_node(
            "Transpose", inputs=["input"], outputs=["output"], name="transpose_node", perm=self.perm
        )

        # Graph
        graph = helper.make_graph(
            [transpose_node], "transpose_graph", [input_tensor], [output_tensor]
        )

        return graph

    def _compute_output_shape(self) -> Tuple[int, ...]:
        """Compute output shape based on permutation."""
        output_shape = tuple(self.input_shape[i] for i in self.perm)
        return output_shape

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute expected output using NumPy."""
        input_data = inputs["input"]
        result = np.transpose(input_data, self.perm)
        return {"output": result}
