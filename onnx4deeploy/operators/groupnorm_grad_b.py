# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""GroupNormGradB operator test implementation."""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class GroupNormGradBOperatorTest(BaseOperatorTest):
    """Test generator for GroupNormGradB operator (gradient w.r.t. bias)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.num_groups = 1
        self.epsilon = 0.001

    def get_operator_name(self) -> str:
        return "GroupNormGradB"

    def load_config(self) -> Dict[str, Any]:
        """Load GroupNormGradB-specific configuration."""
        config = super().load_config()

        gngb_config = config.get("groupnormgradb", {})
        self.input_shape = tuple(gngb_config.get("input_shape", [1, 8, 1, 64]))
        self.num_groups = gngb_config.get("num_groups", 1)
        self.epsilon = gngb_config.get("epsilon", 0.001)

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random gradient data."""
        return {
            "dY": np.random.randn(*self.input_shape).astype(np.float32),
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for GroupNormGradB operator."""
        N, C, H, W = self.input_shape

        # Input tensor
        dY_tensor = helper.make_tensor_value_info("dY", TensorProto.FLOAT, self.input_shape)

        # Output tensor
        dBeta_tensor = helper.make_tensor_value_info("dBeta", TensorProto.FLOAT, [C])

        # GroupNormGradB node
        groupnormgradb_node = helper.make_node(
            "GroupNormGradB",
            inputs=["dY"],
            outputs=["dBeta"],
            name="groupnormgradb_node",
            epsilon=self.epsilon,
            num_groups=self.num_groups,
        )

        # Graph
        graph = helper.make_graph(
            [groupnormgradb_node],
            "groupnormgradb_graph",
            [dY_tensor],
            [dBeta_tensor],
        )

        return graph

    def create_model(self, graph, opset_version: int = 13):
        """Create ONNX model for GroupNormGradB."""
        model = helper.make_model(
            graph,
            producer_name=f"{self.get_operator_name().lower()}_test",
            opset_imports=[helper.make_opsetid("", opset_version)],
        )

        N, C, H, W = self.input_shape

        # Manually set output shape
        output_tensor = model.graph.output[0]
        del output_tensor.type.tensor_type.shape.dim[:]
        output_tensor.type.tensor_type.shape.dim.add().dim_value = C

        return model

    def run_inference(self, onnx_file: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Skip ONNX Runtime inference for custom operators.

        Compute output directly using NumPy.
        """
        return self.compute_expected_output(inputs)

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute expected output using NumPy."""
        dY = inputs["dY"]

        # Sum over batch, height, and width dimensions
        dBeta = dY.sum(axis=(0, 2, 3)).astype(np.float32)

        return {"dBeta": dBeta}
