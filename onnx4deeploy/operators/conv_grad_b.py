# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""ConvGradB operator test implementation."""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class ConvGradBOperatorTest(BaseOperatorTest):
    """Test generator for ConvGradB operator (custom op - gradient w.r.t. bias)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.output_grad_shape = None
        self.bias_shape = None

    def get_operator_name(self) -> str:
        return "ConvGradB"

    def load_config(self) -> Dict[str, Any]:
        """Load ConvGradB-specific configuration."""
        config = super().load_config()

        cgb_config = config.get("convgradb", {})
        self.output_grad_shape = tuple(cgb_config.get("output_grad_shape", [1, 2, 5, 5]))
        self.bias_shape = tuple(cgb_config.get("bias_shape", [2]))

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random gradient data."""
        return {
            "output_grad": np.random.randn(*self.output_grad_shape).astype(np.float32),
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for ConvGradB operator."""
        # Input tensor
        output_grad_tensor = helper.make_tensor_value_info(
            "output_grad", TensorProto.FLOAT, self.output_grad_shape
        )

        # Output tensor
        bias_grad_tensor = helper.make_tensor_value_info(
            "bias_grad", TensorProto.FLOAT, self.bias_shape
        )

        # ConvGradB node
        convgradb_node = helper.make_node(
            "ConvGradB", inputs=["output_grad"], outputs=["bias_grad"], name="convgradb_node"
        )

        # Graph
        graph = helper.make_graph(
            [convgradb_node], "convgradb_graph", [output_grad_tensor], [bias_grad_tensor]
        )

        return graph

    def create_model(self, graph, opset_version: int = 13):
        """Create ONNX model for ConvGradB."""
        model = helper.make_model(
            graph,
            producer_name=f"{self.get_operator_name().lower()}_test",
            opset_imports=[helper.make_opsetid("", opset_version)],
        )

        # Manually set output shape
        output_tensor = model.graph.output[0]
        del output_tensor.type.tensor_type.shape.dim[:]
        for dim in self.bias_shape:
            output_tensor.type.tensor_type.shape.dim.add().dim_value = dim

        return model

    def run_inference(self, onnx_file: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Skip ONNX Runtime inference for custom operators.

        Compute output directly using NumPy.
        """
        return self.compute_expected_output(inputs)

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute expected output using NumPy."""
        output_grad = inputs["output_grad"]

        # For bias gradient: sum over batch, height, and width dimensions
        # bias_grad[c] = sum over (batch, h, w) of output_grad[batch, c, h, w]
        bias_grad = output_grad.sum(axis=(0, 2, 3)).astype(np.float32)

        return {"bias_grad": bias_grad}
