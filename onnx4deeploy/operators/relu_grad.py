# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""ReluGrad operator test implementation."""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class ReluGradOperatorTest(BaseOperatorTest):
    """Test generator for ONNX ReluGrad operator (custom/training op)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None

    def get_operator_name(self) -> str:
        return "ReluGrad"

    def load_config(self) -> Dict[str, Any]:
        """Load ReluGrad-specific configuration."""
        config = super().load_config()

        relugrad_config = config.get("relugrad", {})
        self.input_shape = tuple(relugrad_config["input_shape"])

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random gradient and original input data."""
        return {
            "grad_output": np.random.randn(*self.input_shape).astype(np.float32),
            "original_input": np.random.uniform(-1, 1, self.input_shape).astype(np.float32),
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for ReluGrad operator."""
        # Input tensors
        grad_output_tensor = helper.make_tensor_value_info(
            "grad_output", TensorProto.FLOAT, self.input_shape
        )
        original_input_tensor = helper.make_tensor_value_info(
            "original_input", TensorProto.FLOAT, self.input_shape
        )

        # Output tensor
        grad_input_tensor = helper.make_tensor_value_info(
            "grad_input", TensorProto.FLOAT, self.input_shape
        )

        # ReluGrad node (custom Microsoft domain)
        relugrad_node = helper.make_node(
            "ReluGrad",
            inputs=["grad_output", "original_input"],
            outputs=["grad_input"],
            name="relu_grad_node",
            domain="com.microsoft",
        )

        # Graph
        graph = helper.make_graph(
            [relugrad_node],
            "relugrad_graph",
            [grad_output_tensor, original_input_tensor],
            [grad_input_tensor],
        )

        return graph

    def create_model(self, graph, opset_version: int = 13):
        """Create ONNX model for ReluGrad with custom domain."""
        model = helper.make_model(
            graph,
            producer_name=f"{self.get_operator_name().lower()}_test",
            opset_imports=[
                helper.make_opsetid("", opset_version),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

        return model

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute expected output using NumPy.

        ReluGrad logic: grad_input = grad_output if original_input > 0 else 0
        """
        grad_output = inputs["grad_output"]
        original_input = inputs["original_input"]

        grad_input = np.where(original_input > 0, grad_output, 0).astype(np.float32)

        return {"grad_input": grad_input}
