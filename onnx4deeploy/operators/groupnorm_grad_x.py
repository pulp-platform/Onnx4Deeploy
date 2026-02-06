# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""GroupNormGradX operator test implementation."""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class GroupNormGradXOperatorTest(BaseOperatorTest):
    """Test generator for GroupNormGradX operator (gradient w.r.t. input)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.num_groups = 1
        self.epsilon = 0.001

    def get_operator_name(self) -> str:
        return "GroupNormGradX"

    def load_config(self) -> Dict[str, Any]:
        """Load GroupNormGradX-specific configuration."""
        config = super().load_config()

        gngx_config = config.get("groupnormgradx", {})
        self.input_shape = tuple(gngx_config.get("input_shape", [1, 8, 1, 64]))
        self.num_groups = gngx_config.get("num_groups", 1)
        self.epsilon = gngx_config.get("epsilon", 0.001)

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random gradient, input, gamma data, and compute stat."""
        N, C, H, W = self.input_shape

        # Generate random data
        dY = np.random.randn(*self.input_shape).astype(np.float32)
        X = np.random.randn(*self.input_shape).astype(np.float32)
        gamma = np.random.randn(C).astype(np.float32)

        # Compute stat array [N, G, 2]
        channels_per_group = C // self.num_groups
        X_reshaped = X.reshape(N, self.num_groups, channels_per_group, H, W)
        mean = X_reshaped.mean(axis=(2, 3, 4))
        var = X_reshaped.var(axis=(2, 3, 4))
        inv_std = 1.0 / np.sqrt(var + self.epsilon)

        stat = np.stack([mean, inv_std], axis=-1).astype(np.float32)

        return {
            "dY": dY,
            "X": X,
            "gamma": gamma,
            "stat": stat,
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for GroupNormGradX operator."""
        N, C, H, W = self.input_shape

        # Input tensors
        dY_tensor = helper.make_tensor_value_info("dY", TensorProto.FLOAT, self.input_shape)
        X_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, self.input_shape)
        gamma_tensor = helper.make_tensor_value_info("gamma", TensorProto.FLOAT, [C])
        stat_tensor = helper.make_tensor_value_info(
            "stat", TensorProto.FLOAT, [N, self.num_groups, 2]
        )

        # Output tensor
        dX_tensor = helper.make_tensor_value_info("dX", TensorProto.FLOAT, self.input_shape)

        # GroupNormGradX node
        groupnormgradx_node = helper.make_node(
            "GroupNormGradX",
            inputs=["dY", "X", "gamma", "stat"],
            outputs=["dX"],
            name="groupnormgradx_node",
            epsilon=self.epsilon,
            num_groups=self.num_groups,
        )

        # Graph
        graph = helper.make_graph(
            [groupnormgradx_node],
            "groupnormgradx_graph",
            [dY_tensor, X_tensor, gamma_tensor, stat_tensor],
            [dX_tensor],
            value_info=[stat_tensor],  # Add stat to value_info for shape inference
        )

        return graph

    def run_inference(self, onnx_file: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Skip ONNX Runtime inference for custom operators.

        Compute output directly using NumPy.
        """
        return self.compute_expected_output(inputs)

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute expected output using NumPy."""
        dY = inputs["dY"]
        X = inputs["X"]
        gamma = inputs["gamma"]
        stat = inputs["stat"]

        N, C, H, W = X.shape
        G = self.num_groups
        C_p_G = C // G

        # Parse stat
        mean = stat[:, :, 0]
        inv_std = stat[:, :, 1]

        # Reshape
        X_reshaped = X.reshape(N, G, C_p_G, H, W)
        dY_reshaped = dY.reshape(N, G, C_p_G, H, W)
        gamma_reshaped = gamma.reshape(1, G, C_p_G, 1, 1)

        # Broadcast
        mean_b = mean.reshape(N, G, 1, 1, 1)
        inv_std_b = inv_std.reshape(N, G, 1, 1, 1)

        # X_norm = (X - mean) * inv_std
        X_norm = (X_reshaped - mean_b) * inv_std_b

        # Weighted gradient
        gamma_dY = gamma_reshaped * dY_reshaped

        # Mean of (gamma * dY) and (gamma * dY * X_norm) over group
        mean_gamma_dY = gamma_dY.mean(axis=(2, 3, 4), keepdims=True)
        mean_gamma_dY_Xnorm = (gamma_dY * X_norm).mean(axis=(2, 3, 4), keepdims=True)

        # dX = inv_std * (gamma_dY - mean_gamma_dY - X_norm * mean_gamma_dY_Xnorm)
        dX_reshaped = inv_std_b * (gamma_dY - mean_gamma_dY - X_norm * mean_gamma_dY_Xnorm)

        return {"dX": dX_reshaped.reshape(N, C, H, W).astype(np.float32)}
