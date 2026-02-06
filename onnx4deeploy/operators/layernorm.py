# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""LayerNorm operator test implementation."""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class LayerNormOperatorTest(BaseOperatorTest):
    """Test generator for ONNX LayerNormalization operator."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.axis = -1
        self.epsilon = 1e-5

    def get_operator_name(self) -> str:
        return "LayerNormalization"

    def load_config(self) -> Dict[str, Any]:
        """Load LayerNorm-specific configuration."""
        config = super().load_config()

        ln_config = config.get("layernorm", {})
        self.input_shape = tuple(ln_config["input_shape"])
        self.axis = ln_config.get("axis", -1)
        self.epsilon = ln_config.get("epsilon", 1e-5)

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random input, scale, and bias data."""
        # Convert negative axis to positive
        norm_axis = self.axis if self.axis >= 0 else self.axis + len(self.input_shape)

        # Calculate shapes for scale and bias
        params_shape = [self.input_shape[norm_axis]]

        return {
            "input": np.random.randn(*self.input_shape).astype(np.float32),
            "scale": np.random.randn(*params_shape).astype(np.float32),
            "bias": np.random.randn(*params_shape).astype(np.float32),
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for LayerNormalization operator."""
        # Get params shape
        params_shape = list(inputs["scale"].shape)

        # Input tensors
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, self.input_shape)
        scale_tensor = helper.make_tensor_value_info("scale", TensorProto.FLOAT, params_shape)
        bias_tensor = helper.make_tensor_value_info("bias", TensorProto.FLOAT, params_shape)

        # Output tensor
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, self.input_shape)

        # LayerNormalization node
        layernorm_node = helper.make_node(
            "LayerNormalization",
            inputs=["input", "scale", "bias"],
            outputs=["output"],
            name="layernorm_node",
            axis=self.axis,
            epsilon=self.epsilon,
        )

        # Graph
        graph = helper.make_graph(
            [layernorm_node],
            "layernorm_graph",
            [input_tensor, scale_tensor, bias_tensor],
            [output_tensor],
        )

        return graph

    def create_model(self, graph, opset_version: int = 17):
        """Create ONNX model for LayerNormalization (requires opset 17)."""
        model = helper.make_model(
            graph,
            producer_name=f"{self.get_operator_name().lower()}_test",
            opset_imports=[helper.make_opsetid("", opset_version)],
        )

        return model

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Return None to skip validation - we use ONNX Runtime as ground truth.
        """
        return None
