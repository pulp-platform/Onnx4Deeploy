"""GroupNormH2 operator test implementation (for H=2 cases like EPIDENET)."""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class GroupNormH2OperatorTest(BaseOperatorTest):
    """Test generator for GroupNormalization with H=2 (uses GroupNormStats/GroupNormForward nodes)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = (1, 4, 2, 997)  # Default H=2 like EPIDENET ln1
        self.num_groups = 1
        self.epsilon = 0.001

    def get_operator_name(self) -> str:
        return "GroupNormalizationH2"

    def load_config(self) -> Dict[str, Any]:
        """Load GroupNormH2-specific configuration."""
        config = super().load_config()

        # Try to find config under different keys
        gn_config = config.get("groupnormh2", config.get("groupnorm", {}))
        self.input_shape = tuple(gn_config.get("input_shape", [1, 4, 2, 997]))
        self.num_groups = gn_config.get("num_groups", 1)
        self.epsilon = gn_config.get("epsilon", 0.001)

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random input, gamma (scale), and beta (bias) data."""
        N, C, H, W = self.input_shape

        # Use fixed seed for reproducibility (like the legacy test)
        np.random.seed(42)

        return {
            "X": np.random.randn(*self.input_shape).astype(np.float32),
            "gamma": np.random.randn(C).astype(np.float32),
            "beta": np.random.randn(C).astype(np.float32),
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for GroupNormalization with H=2."""
        N, C, H, W = self.input_shape

        # Input tensors
        X_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, self.input_shape)
        gamma_tensor = helper.make_tensor_value_info("gamma", TensorProto.FLOAT, [C])
        beta_tensor = helper.make_tensor_value_info("beta", TensorProto.FLOAT, [C])

        # Output tensors
        Y_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, self.input_shape)
        stat_tensor = helper.make_tensor_value_info(
            "stat", TensorProto.FLOAT, [N, self.num_groups, 2]
        )

        # GroupNormStats node (different name than GroupNormalizationStat)
        groupnorm_stat_node = helper.make_node(
            "GroupNormStats",
            inputs=["X"],
            outputs=["stat"],
            name="groupnorm_stat_node",
            epsilon=self.epsilon,
            num_groups=self.num_groups,
        )

        # GroupNormForward node (different name than GroupNormalization)
        groupnorm_node = helper.make_node(
            "GroupNormForward",
            inputs=["X", "gamma", "beta", "stat"],
            outputs=["Y"],
            name="groupnorm_fwd_node",
            num_groups=self.num_groups,
        )

        # Graph with two nodes
        graph = helper.make_graph(
            [groupnorm_stat_node, groupnorm_node],
            "groupnorm_h2_graph",
            [X_tensor, gamma_tensor, beta_tensor],
            [Y_tensor, stat_tensor],
        )

        return graph

    def create_model(self, graph, opset_version: int = 13):
        """Create ONNX model for GroupNormalization H2."""
        model = helper.make_model(
            graph,
            producer_name=f"{self.get_operator_name().lower()}_test",
            opset_imports=[helper.make_opsetid("", opset_version)],
        )

        N, C, H, W = self.input_shape

        # Manually set output shapes
        Y_output = model.graph.output[0]
        del Y_output.type.tensor_type.shape.dim[:]
        for dim in self.input_shape:
            Y_output.type.tensor_type.shape.dim.add().dim_value = dim

        stat_output = model.graph.output[1]
        del stat_output.type.tensor_type.shape.dim[:]
        stat_output.type.tensor_type.shape.dim.add().dim_value = N
        stat_output.type.tensor_type.shape.dim.add().dim_value = self.num_groups
        stat_output.type.tensor_type.shape.dim.add().dim_value = 2

        return model

    def run_inference(self, onnx_file: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Skip ONNX Runtime inference for custom operators.

        Compute output directly using NumPy.
        """
        return self.compute_expected_output(inputs)

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute expected output using NumPy."""
        X = inputs["X"]
        gamma = inputs["gamma"]
        beta = inputs["beta"]

        N, C, H, W = X.shape
        channels_per_group = C // self.num_groups

        # Reshape for group computation
        X_reshaped = X.reshape(N, self.num_groups, channels_per_group, H, W)

        # Compute statistics per group
        mean = X_reshaped.mean(axis=(2, 3, 4), keepdims=True)
        var = X_reshaped.var(axis=(2, 3, 4), keepdims=True)
        std = np.sqrt(var + self.epsilon)
        inv_std = 1.0 / std

        # Normalize
        X_norm = (X_reshaped - mean) / std

        # Reshape back to [N, C, H, W]
        X_norm = X_norm.reshape(N, C, H, W)

        # Apply scale and shift
        gamma_reshaped = gamma.reshape(1, C, 1, 1)
        beta_reshaped = beta.reshape(1, C, 1, 1)

        Y = gamma_reshaped * X_norm + beta_reshaped

        # Create stat array [N, G, 2]
        mean_flat = mean.reshape(N, self.num_groups)
        inv_std_flat = inv_std.reshape(N, self.num_groups)
        stat = np.stack([mean_flat, inv_std_flat], axis=-1).astype(np.float32)

        return {"Y": Y.astype(np.float32), "stat": stat}
