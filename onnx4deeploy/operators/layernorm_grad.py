# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""LayerNormGrad operator test implementation.

Matches ORT com.microsoft.LayerNormalizationGrad signature:
  Inputs : grad_in (dY), original_input (X), weight (scale), mean, inv_std
  Outputs: output_grad (dX), d_weight (dScale), d_bias (dBeta)
"""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class LayerNormGradOperatorTest(BaseOperatorTest):
    """Test generator for com.microsoft LayerNormalizationGrad (3-output variant).

    ORT I/O signature:
      Inputs : dY [*batch, D], X [*batch, D], scale [D], mean [*batch, 1], inv_std [*batch, 1]
      Outputs: dX [*batch, D], dScale [D], dBias [D]
    """

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.axis = -1
        self.epsilon = 1e-5
        self.stash_type = 1

    def get_operator_name(self) -> str:
        return "LayerNormalizationGrad"

    def load_config(self) -> Dict[str, Any]:
        config = super().load_config()
        cfg = config.get("layernormgrad", {})
        self.input_shape = tuple(cfg["input_shape"])
        self.axis = cfg.get("axis", -1)
        self.epsilon = cfg.get("epsilon", 1e-5)
        self.stash_type = cfg.get("stash_type", 1)
        return config

    def _stat_shape(self):
        """Shape of mean / inv_std: same as input but with axis dim = 1."""
        ax = self.axis if self.axis >= 0 else len(self.input_shape) + self.axis
        return tuple(d if i != ax else 1 for i, d in enumerate(self.input_shape))

    def _norm_dim(self):
        ax = self.axis if self.axis >= 0 else len(self.input_shape) + self.axis
        return self.input_shape[ax]

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        X = np.random.randn(*self.input_shape).astype(np.float32)
        mean = X.mean(axis=self.axis, keepdims=True).astype(np.float32)
        var = X.var(axis=self.axis, keepdims=True).astype(np.float32)
        inv_std = (1.0 / np.sqrt(var + self.epsilon)).astype(np.float32)
        D = self._norm_dim()
        return {
            "grad_in": np.random.randn(*self.input_shape).astype(np.float32),
            "original_input": X,
            "weight": np.random.randn(D).astype(np.float32),
            "mean": mean,
            "inv_std": inv_std,
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        D = self._norm_dim()
        stat_shape = list(self._stat_shape())

        grad_in_t = helper.make_tensor_value_info(
            "grad_in", TensorProto.FLOAT, list(self.input_shape)
        )
        x_t = helper.make_tensor_value_info(
            "original_input", TensorProto.FLOAT, list(self.input_shape)
        )
        weight_t = helper.make_tensor_value_info("weight", TensorProto.FLOAT, [D])
        mean_t = helper.make_tensor_value_info("mean", TensorProto.FLOAT, stat_shape)
        inv_std_t = helper.make_tensor_value_info("inv_std", TensorProto.FLOAT, stat_shape)

        dx_t = helper.make_tensor_value_info(
            "output_grad", TensorProto.FLOAT, list(self.input_shape)
        )
        dscale_t = helper.make_tensor_value_info("d_weight", TensorProto.FLOAT, [D])
        dbias_t = helper.make_tensor_value_info("d_bias", TensorProto.FLOAT, [D])

        node = helper.make_node(
            "LayerNormalizationGrad",
            inputs=["grad_in", "original_input", "weight", "mean", "inv_std"],
            outputs=["output_grad", "d_weight", "d_bias"],
            name="layernorm_grad_node",
            domain="com.microsoft",
            axis=self.axis,
            epsilon=self.epsilon,
            stash_type=self.stash_type,
        )

        graph = helper.make_graph(
            [node],
            "layernormgrad_graph",
            [grad_in_t, x_t, weight_t, mean_t, inv_std_t],
            [dx_t, dscale_t, dbias_t],
        )
        return graph

    def create_model(self, graph, opset_version: int = 13):
        model = helper.make_model(
            graph,
            producer_name="layernormgrad_test",
            opset_imports=[
                helper.make_opsetid("", opset_version),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )
        # Set output shapes explicitly (custom op — shape inference won't run)
        shapes = [list(self.input_shape), [self._norm_dim()], [self._norm_dim()]]
        for out_t, shape in zip(model.graph.output, shapes):
            del out_t.type.tensor_type.shape.dim[:]
            for d in shape:
                out_t.type.tensor_type.shape.dim.add().dim_value = d
        return model

    def run_inference(self, onnx_file, inputs):
        return self.compute_expected_output(inputs)

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        LayerNorm backward (axis=-1 generalised):
          X_norm = (X - mean) * inv_std
          dBias  = sum(dY,        reduce over batch dims)   [D]
          dScale = sum(dY*X_norm, reduce over batch dims)   [D]
          d_x_norm = dY * scale
          dX = inv_std * (d_x_norm
                          - mean(d_x_norm,        axis)
                          - X_norm * mean(d_x_norm * X_norm, axis))
        """
        dY = inputs["grad_in"]
        X = inputs["original_input"]
        scale = inputs["weight"]
        mean = inputs["mean"]
        inv_std = inputs["inv_std"]

        ax = self.axis

        X_norm = (X - mean) * inv_std

        # Broadcast scale to full shape for elementwise ops
        # scale shape [D] — need to reshape to [..., D]
        scale_b = scale  # numpy broadcasts trailing dim automatically

        # dBias: sum dY over all axes except the norm axis
        reduce_axes = tuple(i for i in range(dY.ndim) if i != (ax % dY.ndim))
        dBias = dY.sum(axis=reduce_axes).astype(np.float32)
        dScale = (dY * X_norm).sum(axis=reduce_axes).astype(np.float32)

        # dX
        dY.shape[ax]
        d_x_norm = dY * scale_b
        dX = inv_std * (
            d_x_norm
            - d_x_norm.mean(axis=ax, keepdims=True)
            - X_norm * (d_x_norm * X_norm).mean(axis=ax, keepdims=True)
        )

        return {
            "output_grad": dX.astype(np.float32),
            "d_weight": dScale,
            "d_bias": dBias,
        }
