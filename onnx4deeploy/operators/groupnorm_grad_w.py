# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""GroupNormGradW operator test implementation."""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class GroupNormGradWOperatorTest(BaseOperatorTest):
    """Test generator for GroupNormGradW operator (gradient w.r.t. weight)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.num_groups = 1
        self.epsilon = 0.001

    def get_operator_name(self) -> str:
        return "GroupNormGradW"

    def load_config(self) -> Dict[str, Any]:
        """Load GroupNormGradW-specific configuration."""
        config = super().load_config()

        gngw_config = config.get("groupnormgradw", {})
        self.input_shape = tuple(gngw_config.get("input_shape", [1, 8, 1, 64]))
        self.num_groups = gngw_config.get("num_groups", 1)
        self.epsilon = gngw_config.get("epsilon", 0.001)

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random gradient and input data, and compute stat."""
        N, C, H, W = self.input_shape

        # Generate random data
        dY = np.random.randn(*self.input_shape).astype(np.float32)
        X = np.random.randn(*self.input_shape).astype(np.float32)

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
            "stat": stat,
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for GroupNormGradW operator."""
        N, C, H, W = self.input_shape

        # Input tensors
        dY_tensor = helper.make_tensor_value_info("dY", TensorProto.FLOAT, self.input_shape)
        X_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, self.input_shape)
        stat_tensor = helper.make_tensor_value_info(
            "stat", TensorProto.FLOAT, [N, self.num_groups, 2]
        )

        # Output tensor
        dGamma_tensor = helper.make_tensor_value_info("dGamma", TensorProto.FLOAT, [C])

        # GroupNormGradW node
        groupnormgradw_node = helper.make_node(
            "GroupNormGradW",
            inputs=["dY", "X", "stat"],
            outputs=["dGamma"],
            name="groupnormgradw_node",
            num_groups=self.num_groups,
        )

        # Graph
        graph = helper.make_graph(
            [groupnormgradw_node],
            "groupnormgradw_graph",
            [dY_tensor, X_tensor, stat_tensor],
            [dGamma_tensor],
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

        # Broadcast
        mean_b = mean.reshape(N, G, 1, 1, 1)
        inv_std_b = inv_std.reshape(N, G, 1, 1, 1)

        # X_norm = (X - mean) * inv_std
        X_norm = (X_reshaped - mean_b) * inv_std_b

        # dGamma = sum(dY * X_norm) over N, H, W
        dGamma_reshaped = (dY_reshaped * X_norm).sum(axis=(0, 3, 4))
        dGamma = dGamma_reshaped.reshape(C).astype(np.float32)

        return {"dGamma": dGamma}
