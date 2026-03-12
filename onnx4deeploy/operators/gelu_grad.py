# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""GeluGrad operator test implementation."""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class GeluGradOperatorTest(BaseOperatorTest):
    """Test generator for GeluGrad operator (com.microsoft training op).

    GeluGrad(dY, X) -> dX
      dX = dY * gelu'(X)
      gelu'(x) = 0.5 * (1 + erf(x / sqrt(2))) + x * exp(-0.5 * x^2) / sqrt(2 * pi)
    """

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None

    def get_operator_name(self) -> str:
        return "GeluGrad"

    def load_config(self) -> Dict[str, Any]:
        config = super().load_config()
        cfg = config.get("gelugrad", {})
        self.input_shape = tuple(cfg.get("input_shape", [1, 64]))
        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        return {
            "dY": np.random.randn(*self.input_shape).astype(np.float32),
            "X": np.random.uniform(-3, 3, self.input_shape).astype(np.float32),
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        dY_tensor = helper.make_tensor_value_info("dY", TensorProto.FLOAT, self.input_shape)
        X_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, self.input_shape)
        dX_tensor = helper.make_tensor_value_info("dX", TensorProto.FLOAT, self.input_shape)

        node = helper.make_node(
            "GeluGrad",
            inputs=["dY", "X"],
            outputs=["dX"],
            name="gelugrad_node",
            domain="com.microsoft",
        )

        graph = helper.make_graph(
            [node],
            "gelugrad_graph",
            [dY_tensor, X_tensor],
            [dX_tensor],
        )
        return graph

    def create_model(self, graph, opset_version: int = 13):
        return helper.make_model(
            graph,
            producer_name="gelugrad_test",
            opset_imports=[
                helper.make_opsetid("", opset_version),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

    def run_inference(self, onnx_file, inputs):
        return self.compute_expected_output(inputs)

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        dY = inputs["dY"]
        X = inputs["X"]
        # gelu'(x) = 0.5*(1+erf(x/sqrt(2))) + x*exp(-0.5*x^2)/sqrt(2*pi)
        sqrt2 = np.sqrt(2.0)
        sqrt2pi = np.sqrt(2.0 * np.pi)
        gelu_prime = (
            0.5 * (1.0 + np.vectorize(lambda v: float(np.math.erf(v / sqrt2)))(X))
            + X * np.exp(-0.5 * X * X) / sqrt2pi
        )
        dX = (dY * gelu_prime).astype(np.float32)
        return {"dX": dX}
