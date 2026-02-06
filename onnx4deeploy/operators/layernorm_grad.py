"""LayerNormGrad operator test implementation."""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class LayerNormGradOperatorTest(BaseOperatorTest):
    """Test generator for ONNX LayerNormalizationGrad operator (custom/training op)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.axis = -1
        self.epsilon = 0.00000999999974737852
        self.stash_type = 1

    def get_operator_name(self) -> str:
        return "LayerNormalizationGrad"

    def load_config(self) -> Dict[str, Any]:
        """Load LayerNormGrad-specific configuration."""
        config = super().load_config()

        lng_config = config.get("layernormgrad", {})
        self.input_shape = tuple(lng_config["input_shape"])
        self.axis = lng_config.get("axis", -1)
        self.epsilon = lng_config.get("epsilon", 0.00000999999974737852)
        self.stash_type = lng_config.get("stash_type", 1)

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random gradient and original input data."""
        # Calculate params shape based on axis
        last_dim = self.input_shape[self.axis]

        return {
            "grad_in": np.random.randn(*self.input_shape).astype(np.float32),
            "original_input": np.random.randn(*self.input_shape).astype(np.float32),
            "weight": np.random.randn(last_dim).astype(np.float32),
            "bias": np.random.randn(last_dim).astype(np.float32),
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for LayerNormalizationGrad operator."""
        last_dim = self.input_shape[self.axis]

        # Input tensors
        grad_in_tensor = helper.make_tensor_value_info(
            "grad_in", TensorProto.FLOAT, self.input_shape
        )
        original_input_tensor = helper.make_tensor_value_info(
            "original_input", TensorProto.FLOAT, self.input_shape
        )
        weight_tensor = helper.make_tensor_value_info("weight", TensorProto.FLOAT, [last_dim])
        bias_tensor = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [last_dim])

        # Output tensor
        output_grad_tensor = helper.make_tensor_value_info(
            "output_grad", TensorProto.FLOAT, self.input_shape
        )

        # LayerNormalizationGrad node (custom Microsoft domain)
        layernorm_grad_node = helper.make_node(
            "LayerNormalizationGrad",
            inputs=["grad_in", "original_input", "weight", "bias"],
            outputs=["output_grad"],
            name="layernorm_grad_node",
            domain="com.microsoft",
            axis=self.axis,
            epsilon=self.epsilon,
            stash_type=self.stash_type,
        )

        # Graph
        graph = helper.make_graph(
            [layernorm_grad_node],
            "layernormgrad_graph",
            [grad_in_tensor, original_input_tensor, weight_tensor, bias_tensor],
            [output_grad_tensor],
        )

        return graph

    def create_model(self, graph, opset_version: int = 13):
        """Create ONNX model for LayerNormalizationGrad with custom domain."""
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
        Compute expected output using NumPy.

        Simplified implementation for testing purposes.
        """
        grad_in = inputs["grad_in"]
        original_input = inputs["original_input"]
        weight = inputs["weight"]

        # Calculate mean and variance used in the forward pass
        mean = np.mean(original_input, axis=self.axis, keepdims=True)
        var = np.var(original_input, axis=self.axis, keepdims=True)
        std = np.sqrt(var + self.epsilon)

        # Normalized input from forward pass
        (original_input - mean) / std

        # Gradient for normalized input
        dnormalized = grad_in * weight

        # Gradient for original input
        N = original_input.shape[self.axis]
        dx_norm = dnormalized

        # First component: direct gradient through normalization
        dx = dx_norm / std

        # Second component: gradient through the mean
        dx_mean = -np.sum(dx_norm, axis=self.axis, keepdims=True) / std
        dx = dx + dx_mean / N

        # Third component: gradient through the variance
        dx_var = (
            -0.5
            * np.sum(dx_norm * (original_input - mean), axis=self.axis, keepdims=True)
            * (var + self.epsilon) ** (-1.5)
        )
        dx = dx + dx_var * 2 * (original_input - mean) / N

        return {"output_grad": dx.astype(np.float32)}
