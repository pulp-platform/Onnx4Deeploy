# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""SGD operator test implementation."""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class SGDOperatorTest(BaseOperatorTest):
    """Test generator for ONNX SGD operator (custom/training op)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.weight_shape = None
        self.learning_rate = 0.01

    def get_operator_name(self) -> str:
        return "SGD"

    def load_config(self) -> Dict[str, Any]:
        """Load SGD-specific configuration."""
        config = super().load_config()

        sgd_config = config.get("sgd", {})
        self.weight_shape = tuple(sgd_config.get("weight_shape", (10, 8)))
        self.learning_rate = float(sgd_config.get("learning_rate", 0.01))

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random weights and gradient data."""
        return {
            "weights": np.random.randn(*self.weight_shape).astype(np.float32),
            "gradient": np.random.randn(*self.weight_shape).astype(np.float32) * 0.1,
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for SGD operator."""
        # Input tensors
        weights_tensor = helper.make_tensor_value_info(
            "weights", TensorProto.FLOAT, self.weight_shape
        )
        gradient_tensor = helper.make_tensor_value_info(
            "gradient", TensorProto.FLOAT, self.weight_shape
        )

        # Output tensor
        updated_weights_tensor = helper.make_tensor_value_info(
            "updated_weights", TensorProto.FLOAT, self.weight_shape
        )

        # SGD node (custom domain)
        sgd_node = helper.make_node(
            "SGD",
            inputs=["weights", "gradient"],
            outputs=["updated_weights"],
            name="sgd_node",
            domain="com.mydomain",
            lr=float(self.learning_rate),
        )

        # Graph
        graph = helper.make_graph(
            [sgd_node], "sgd_graph", [weights_tensor, gradient_tensor], [updated_weights_tensor]
        )

        return graph

    def create_model(self, graph, opset_version: int = 13):
        """Create ONNX model for SGD with custom domain."""
        model = helper.make_model(
            graph,
            producer_name=f"{self.get_operator_name().lower()}_test",
            opset_imports=[
                helper.make_opsetid("", opset_version),
                helper.make_opsetid("com.mydomain", 1),
            ],
        )

        # Manually set output shape for custom operators
        output_tensor = model.graph.output[0]
        del output_tensor.type.tensor_type.shape.dim[:]
        for dim in self.weight_shape:
            output_tensor.type.tensor_type.shape.dim.add().dim_value = dim

        return model

    def run_inference(self, onnx_file: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Skip ONNX Runtime inference for custom operators.

        Compute output directly using NumPy.
        """
        return self.compute_expected_output(inputs)

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute expected output using NumPy.

        SGD logic: updated_weights = weights - learning_rate * gradient
        """
        weights = inputs["weights"]
        gradient = inputs["gradient"]

        updated_weights = weights - self.learning_rate * gradient

        return {"updated_weights": updated_weights}
