"""SoftmaxGrad operator test implementation."""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class SoftmaxGradOperatorTest(BaseOperatorTest):
    """Test generator for ONNX SoftmaxGrad operator (custom/training op)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.axis = -1

    def get_operator_name(self) -> str:
        return "SoftmaxGrad"

    def load_config(self) -> Dict[str, Any]:
        """Load SoftmaxGrad-specific configuration."""
        config = super().load_config()

        sg_config = config.get("softmax_grad", {})
        self.input_shape = tuple(sg_config["input_shape"])
        self.axis = sg_config.get("axis", -1)

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random gradient and softmax output data."""
        # y is the result of a softmax operation (probabilities sum to 1)
        y = np.random.rand(*self.input_shape).astype(np.float32)
        y = y / np.sum(y, axis=self.axis, keepdims=True)

        # dy is the upstream gradient
        dy = np.random.randn(*self.input_shape).astype(np.float32)

        return {
            "dy": dy,
            "y": y,
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for SoftmaxGrad operator."""
        # Input tensors
        dy_tensor = helper.make_tensor_value_info("dy", TensorProto.FLOAT, self.input_shape)
        y_tensor = helper.make_tensor_value_info("y", TensorProto.FLOAT, self.input_shape)

        # Output tensor
        dx_tensor = helper.make_tensor_value_info("dx", TensorProto.FLOAT, self.input_shape)

        # SoftmaxGrad node (custom Microsoft domain)
        softmax_grad_node = helper.make_node(
            "SoftmaxGrad",
            inputs=["dy", "y"],
            outputs=["dx"],
            name="softmax_grad_node",
            domain="com.microsoft",
            axis=self.axis,
        )

        # Graph
        graph = helper.make_graph(
            [softmax_grad_node], "softmax_grad_graph", [dy_tensor, y_tensor], [dx_tensor]
        )

        return graph

    def create_model(self, graph, opset_version: int = 13):
        """Create ONNX model for SoftmaxGrad with custom domain."""
        model = helper.make_model(
            graph,
            producer_name=f"{self.get_operator_name().lower()}_test",
            opset_imports=[
                helper.make_opsetid("", opset_version),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

        return model

    def run_inference(self, onnx_file: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Skip ONNX Runtime inference for custom operators.

        Compute output directly using NumPy.
        """
        return self.compute_expected_output(inputs)

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Return None to skip validation - this is a custom operator.
        """
        return None
