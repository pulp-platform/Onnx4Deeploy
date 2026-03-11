# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""GroupNormGrad operator test implementation.

Merged backward node: fuses GradXStat + GradX + GradW + GradB into one.
  Inputs : dY[N,C,H,W], X[N,C,H,W], gamma[C], stat[N,G,2]
  Outputs: dX[N,C,H,W], weight_grad[C], bias_grad[C]
  Attrs  : num_groups
"""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class GroupNormGradOperatorTest(BaseOperatorTest):
    """Test generator for merged GroupNormGrad operator."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.num_groups = 4
        self.epsilon = 0.001

    def get_operator_name(self) -> str:
        return "GroupNormGrad"

    def load_config(self) -> Dict[str, Any]:
        config = super().load_config()
        cfg = config.get("groupnormgrad", {})
        self.input_shape = tuple(cfg.get("input_shape", [2, 8, 4, 4]))
        self.num_groups = cfg.get("num_groups", 4)
        self.epsilon = cfg.get("epsilon", 0.001)
        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        N, C, H, W = self.input_shape
        G = self.num_groups
        Cg = C // G

        dY = np.random.randn(*self.input_shape).astype(np.float32)
        X = np.random.randn(*self.input_shape).astype(np.float32)
        gamma = np.random.randn(C).astype(np.float32)

        # Compute stat[N, G, 2]: mean and inv_std per (n, g) pair
        X_r = X.reshape(N, G, Cg, H, W)
        mean = X_r.mean(axis=(2, 3, 4))  # [N, G]
        var = X_r.var(axis=(2, 3, 4))  # [N, G]
        inv_std = 1.0 / np.sqrt(var + self.epsilon)

        stat = np.stack([mean, inv_std], axis=-1).astype(np.float32)  # [N, G, 2]

        return {
            "dY": dY,
            "X": X,
            "gamma": gamma,
            "stat": stat,
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        N, C, H, W = self.input_shape
        G = self.num_groups

        dY_t = helper.make_tensor_value_info("dY", TensorProto.FLOAT, [N, C, H, W])
        X_t = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, C, H, W])
        gam_t = helper.make_tensor_value_info("gamma", TensorProto.FLOAT, [C])
        stat_t = helper.make_tensor_value_info("stat", TensorProto.FLOAT, [N, G, 2])

        dX_t = helper.make_tensor_value_info("dX", TensorProto.FLOAT, [N, C, H, W])
        wg_t = helper.make_tensor_value_info("weight_grad", TensorProto.FLOAT, [C])
        bg_t = helper.make_tensor_value_info("bias_grad", TensorProto.FLOAT, [C])

        node = helper.make_node(
            "GroupNormGrad",
            inputs=["dY", "X", "gamma", "stat"],
            outputs=["dX", "weight_grad", "bias_grad"],
            name="groupnormgrad_node",
            num_groups=self.num_groups,
        )

        graph = helper.make_graph(
            [node],
            "groupnormgrad_graph",
            [dY_t, X_t, gam_t, stat_t],
            [dX_t, wg_t, bg_t],
        )
        return graph

    def create_model(self, graph, opset_version: int = 13):
        model = helper.make_model(
            graph,
            producer_name="groupnormgrad_test",
            opset_imports=[helper.make_opsetid("", opset_version)],
        )
        # Set output shapes explicitly (custom op — shape inference won't run)
        N, C, H, W = self.input_shape
        shapes = [[N, C, H, W], [C], [C]]
        for out_t, shape in zip(model.graph.output, shapes):
            del out_t.type.tensor_type.shape.dim[:]
            for d in shape:
                out_t.type.tensor_type.shape.dim.add().dim_value = d
        return model

    def run_inference(self, onnx_file, inputs):
        return self.compute_expected_output(inputs)

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """GroupNorm backward pass reference implementation."""
        dY = inputs["dY"]
        X = inputs["X"]
        gamma = inputs["gamma"]
        stat = inputs["stat"]

        N, C, H, W = X.shape
        G = self.num_groups
        Cg = C // G

        mean = stat[:, :, 0]  # [N, G]
        inv_std = stat[:, :, 1]  # [N, G]

        # Reshape to [N, G, Cg, H, W] for group operations
        X_r = X.reshape(N, G, Cg, H, W)
        dY_r = dY.reshape(N, G, Cg, H, W)
        gamma_r = gamma.reshape(1, G, Cg, 1, 1)

        mean_b = mean.reshape(N, G, 1, 1, 1)
        inv_std_b = inv_std.reshape(N, G, 1, 1, 1)

        X_norm = (X_r - mean_b) * inv_std_b  # [N, G, Cg, H, W]
        gamma_dY = gamma_r * dY_r  # [N, G, Cg, H, W]

        # Group-level means (over Cg * H * W)
        mean_gdy = gamma_dY.mean(axis=(2, 3, 4), keepdims=True)
        mean_gdy_xhat = (gamma_dY * X_norm).mean(axis=(2, 3, 4), keepdims=True)

        # dX = inv_std * (gamma*dY - mean(gamma*dY) - X_norm * mean(gamma*dY*X_norm))
        dX_r = inv_std_b * (gamma_dY - mean_gdy - X_norm * mean_gdy_xhat)
        dX = dX_r.reshape(N, C, H, W).astype(np.float32)

        # weight_grad = sum over N, H, W of (dY * X_norm) per channel
        # bias_grad   = sum over N, H, W of dY per channel
        dX_norm_full = X_norm.reshape(N, C, H, W)
        dY_flat = dY

        reduce_axes = (0, 2, 3)  # sum over N, H, W
        weight_grad = (dY_flat * dX_norm_full).sum(axis=reduce_axes).astype(np.float32)
        bias_grad = dY_flat.sum(axis=reduce_axes).astype(np.float32)

        return {
            "dX": dX,
            "weight_grad": weight_grad,
            "bias_grad": bias_grad,
        }
