# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MaxPoolGrad operator test implementation."""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class MaxPoolGradOperatorTest(BaseOperatorTest):
    """Test generator for MaxPoolGrad operator (custom training op).

    MaxPoolGrad(input_grad, original_input) -> output_grad

    Inputs:
      - input_grad     [N, C, Ho, Wo]  upstream gradient
      - original_input [N, C, Hi, Wi]  original forward input (used to recompute argmax)
    Output:
      - output_grad    [N, C, Hi, Wi]  gradient w.r.t. forward input
    """

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_grad_shape = None  # [N, C, Ho, Wo]
        self.original_input_shape = None  # [N, C, Hi, Wi]
        self.kernel_shape = None
        self.strides = None
        self.pads = None
        self.ceil_mode = 0

    def get_operator_name(self) -> str:
        return "MaxPoolGrad"

    def load_config(self) -> Dict[str, Any]:
        config = super().load_config()
        cfg = config.get("maxpoolgrad", {})
        self.input_grad_shape = tuple(cfg.get("input_grad_shape", [1, 4, 4, 4]))
        self.original_input_shape = tuple(cfg.get("original_input_shape", [1, 4, 8, 8]))
        self.kernel_shape = tuple(cfg.get("kernel_shape", [2, 2]))
        self.strides = tuple(cfg.get("strides", [2, 2]))
        self.pads = tuple(cfg.get("pads", [0, 0, 0, 0]))
        self.ceil_mode = int(cfg.get("ceil_mode", 0))
        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        return {
            "input_grad": np.random.randn(*self.input_grad_shape).astype(np.float32),
            "original_input": np.random.randn(*self.original_input_shape).astype(np.float32),
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        input_grad_tensor = helper.make_tensor_value_info(
            "input_grad", TensorProto.FLOAT, self.input_grad_shape
        )
        original_input_tensor = helper.make_tensor_value_info(
            "original_input", TensorProto.FLOAT, self.original_input_shape
        )
        output_grad_tensor = helper.make_tensor_value_info(
            "output_grad", TensorProto.FLOAT, self.original_input_shape
        )

        node = helper.make_node(
            "MaxPoolGrad",
            inputs=["input_grad", "original_input"],
            outputs=["output_grad"],
            name="maxpoolgrad_node",
            kernel_shape=list(self.kernel_shape),
            strides=list(self.strides),
            pads=list(self.pads),
            ceil_mode=self.ceil_mode,
        )

        graph = helper.make_graph(
            [node],
            "maxpoolgrad_graph",
            [input_grad_tensor, original_input_tensor],
            [output_grad_tensor],
        )
        return graph

    def create_model(self, graph, opset_version: int = 13):
        model = helper.make_model(
            graph,
            producer_name="maxpoolgrad_test",
            opset_imports=[helper.make_opsetid("", opset_version)],
        )
        # Manually set output shape (custom op)
        out = model.graph.output[0]
        del out.type.tensor_type.shape.dim[:]
        for d in self.original_input_shape:
            out.type.tensor_type.shape.dim.add().dim_value = d
        return model

    def run_inference(self, onnx_file, inputs):
        return self.compute_expected_output(inputs)

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Scatter upstream gradient to the argmax positions of each pooling window."""
        input_grad = inputs["input_grad"]  # [N, C, Ho, Wo]
        X = inputs["original_input"]  # [N, C, Hi, Wi]

        N, C, Hi, Wi = self.original_input_shape
        _, _, Ho, Wo = self.input_grad_shape
        kH, kW = self.kernel_shape
        sH, sW = self.strides
        pad_top, pad_bottom, pad_left, pad_right = self.pads

        output_grad = np.zeros((N, C, Hi, Wi), dtype=np.float32)

        for n in range(N):
            for c in range(C):
                for oh in range(Ho):
                    for ow in range(Wo):
                        h_start = oh * sH - pad_top
                        w_start = ow * sW - pad_left
                        h_start + kH
                        w_start + kW

                        # collect valid positions
                        best_val = -np.inf
                        best_h, best_w = max(0, h_start), max(0, w_start)
                        for kh in range(kH):
                            for kw in range(kW):
                                ih = h_start + kh
                                iw = w_start + kw
                                if 0 <= ih < Hi and 0 <= iw < Wi:
                                    if X[n, c, ih, iw] > best_val:
                                        best_val = X[n, c, ih, iw]
                                        best_h, best_w = ih, iw

                        output_grad[n, c, best_h, best_w] += input_grad[n, c, oh, ow]

        return {"output_grad": output_grad}
