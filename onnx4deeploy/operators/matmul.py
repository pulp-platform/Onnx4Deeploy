# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MatMul operator test implementation."""

from typing import Any, Dict, Tuple

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class MatMulOperatorTest(BaseOperatorTest):
    """Test generator for ONNX MatMul operator."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_a_shape = None
        self.input_b_shape = None

    def get_operator_name(self) -> str:
        return "MatMul"

    def load_config(self) -> Dict[str, Any]:
        """Load MatMul-specific configuration."""
        config = super().load_config()

        matmul_config = config.get("matmul", {})
        self.input_a_shape = tuple(matmul_config["input_a_shape"])
        self.input_b_shape = tuple(matmul_config["input_b_shape"])

        # Validate matrix dimensions
        if len(self.input_a_shape) < 2 or len(self.input_b_shape) < 2:
            raise ValueError("Input shapes must have at least 2 dimensions")

        # For 2D: (M, K) @ (K, N) -> (M, N)
        if self.input_a_shape[-1] != self.input_b_shape[-2]:
            raise ValueError(
                f"Incompatible matrix dimensions: A={self.input_a_shape}, B={self.input_b_shape}"
            )

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random input data."""
        return {
            "input_a": np.random.randn(*self.input_a_shape).astype(np.float32) * 0.1,
            "input_b": np.random.randn(*self.input_b_shape).astype(np.float32) * 0.1,
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for MatMul operator."""
        # Input tensors
        input_tensor_a = helper.make_tensor_value_info(
            "input_a", TensorProto.FLOAT, self.input_a_shape
        )
        input_tensor_b = helper.make_tensor_value_info(
            "input_b", TensorProto.FLOAT, self.input_b_shape
        )

        # Output shape
        output_shape = self._compute_output_shape()
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        # MatMul node
        matmul_node = helper.make_node(
            "MatMul", inputs=["input_a", "input_b"], outputs=["output"], name="matmul_node"
        )

        # Graph
        graph = helper.make_graph(
            [matmul_node], "matmul_graph", [input_tensor_a, input_tensor_b], [output_tensor]
        )

        return graph

    def _compute_output_shape(self) -> Tuple[int, ...]:
        """Compute output shape for matrix multiplication."""
        # Standard matrix multiplication rules
        a_shape = self.input_a_shape
        b_shape = self.input_b_shape

        # Last two dimensions: (..., M, K) @ (..., K, N) -> (..., M, N)
        output_shape = a_shape[:-1] + (b_shape[-1],)
        return output_shape

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute expected output using NumPy."""
        result = np.matmul(inputs["input_a"], inputs["input_b"])
        return {"output": result}
