# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""ConvGrad operator test: combined dX + dW (no-bias, 2-output variant).

ORT com.microsoft.ConvGrad no-bias signature:
  Inputs : [dY (output_grad), X (input_data), W (weight)]
  Outputs: [dX (input_grad), dW (weight_grad)]
"""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class ConvGradXWOperatorTest(BaseOperatorTest):
    """Test generator for ConvGrad (dX + dW, no bias)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.output_grad_shape = None  # dY: [N, C_out, H_out, W_out]
        self.input_shape = None  # X:  [N, C_in,  H_in,  W_in]
        self.weight_shape = None  # W:  [C_out, C_in/group, kH, kW]
        self.input_grad_shape = None  # dX: same as X
        self.weight_grad_shape = None  # dW: same as W
        self.kernel_shape = [3, 3]
        self.strides = [1, 1]
        self.pads = [0, 0, 0, 0]
        self.dilations = [1, 1]
        self.group = 1

    def get_operator_name(self) -> str:
        return "ConvGrad"

    def load_config(self) -> Dict[str, Any]:
        config = super().load_config()
        cfg = config.get("convgradxw", {})
        self.output_grad_shape = tuple(cfg.get("output_grad_shape", [1, 4, 2, 2]))
        self.input_shape = tuple(cfg.get("input_shape", [1, 2, 4, 4]))
        self.weight_shape = tuple(cfg.get("weight_shape", [4, 2, 3, 3]))
        self.input_grad_shape = tuple(cfg.get("input_grad_shape", [1, 2, 4, 4]))
        self.weight_grad_shape = tuple(cfg.get("weight_grad_shape", [4, 2, 3, 3]))
        self.kernel_shape = cfg.get("kernel_shape", [3, 3])
        self.strides = cfg.get("strides", [1, 1])
        self.pads = cfg.get("pads", [0, 0, 0, 0])
        self.dilations = cfg.get("dilations", [1, 1])
        self.group = cfg.get("group", 1)
        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        np.random.seed(42)
        return {
            "output_grad": np.random.randn(*self.output_grad_shape).astype(np.float32),
            "input_data": np.random.randn(*self.input_shape).astype(np.float32),
            "weight": np.random.randn(*self.weight_shape).astype(np.float32),
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        # Graph inputs (all variable, no initializers)
        output_grad_t = helper.make_tensor_value_info(
            "output_grad", TensorProto.FLOAT, self.output_grad_shape
        )
        input_data_t = helper.make_tensor_value_info(
            "input_data", TensorProto.FLOAT, self.input_shape
        )
        weight_t = helper.make_tensor_value_info("weight", TensorProto.FLOAT, self.weight_shape)

        # Graph outputs
        input_grad_t = helper.make_tensor_value_info(
            "input_grad", TensorProto.FLOAT, self.input_grad_shape
        )
        weight_grad_t = helper.make_tensor_value_info(
            "weight_grad", TensorProto.FLOAT, self.weight_grad_shape
        )

        node = helper.make_node(
            "ConvGrad",
            inputs=["output_grad", "input_data", "weight"],
            outputs=["input_grad", "weight_grad"],
            name="convgrad_node",
            kernel_shape=self.kernel_shape,
            strides=self.strides,
            pads=self.pads,
            dilations=self.dilations,
            group=self.group,
        )

        graph = helper.make_graph(
            [node],
            "convgrad_graph",
            [output_grad_t, input_data_t, weight_t],
            [input_grad_t, weight_grad_t],
        )
        return graph

    def create_model(self, graph, opset_version: int = 13):
        model = helper.make_model(
            graph,
            producer_name="convgrad_test",
            opset_imports=[helper.make_opsetid("", opset_version)],
        )
        shapes = [self.input_grad_shape, self.weight_grad_shape]
        for out_t, shape in zip(model.graph.output, shapes):
            del out_t.type.tensor_type.shape.dim[:]
            for d in shape:
                out_t.type.tensor_type.shape.dim.add().dim_value = d
        return model

    def run_inference(self, onnx_file, inputs):
        return self.compute_expected_output(inputs)

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute dX and dW numerically."""
        output_grad = inputs["output_grad"]  # dY
        input_data = inputs["input_data"]  # X
        weight = inputs["weight"]  # W

        batch, c_out, out_h, out_w = self.output_grad_shape
        _, c_in, in_h, in_w = self.input_shape
        stride_h, stride_w = self.strides
        dil_h, dil_w = self.dilations
        pad_top, pad_left, pad_bottom, pad_right = self.pads
        kernel_h, kernel_w = self.kernel_shape
        c_in_per_group = c_in // self.group
        c_out_per_group = c_out // self.group

        # ── dX ────────────────────────────────────────────────────────────────
        input_grad = np.zeros(self.input_grad_shape, dtype=np.float32)
        for b in range(batch):
            for g in range(self.group):
                ic_start = g * c_in_per_group
                oc_start = g * c_out_per_group
                for ic in range(ic_start, ic_start + c_in_per_group):
                    for ih in range(in_h):
                        for iw in range(in_w):
                            gs = 0.0
                            for oc in range(oc_start, oc_start + c_out_per_group):
                                for kh in range(kernel_h):
                                    for kw in range(kernel_w):
                                        oh_c = ih + pad_top - kh * dil_h
                                        ow_c = iw + pad_left - kw * dil_w
                                        if oh_c % stride_h == 0 and ow_c % stride_w == 0:
                                            oh = oh_c // stride_h
                                            ow = ow_c // stride_w
                                            if 0 <= oh < out_h and 0 <= ow < out_w:
                                                ic_g = ic - ic_start
                                                gs += (
                                                    output_grad[b, oc, oh, ow]
                                                    * weight[oc, ic_g, kh, kw]
                                                )
                            input_grad[b, ic, ih, iw] = gs

        # ── dW ────────────────────────────────────────────────────────────────
        weight_grad = np.zeros(self.weight_grad_shape, dtype=np.float32)
        for b in range(batch):
            for g in range(self.group):
                oc_start = g * c_out_per_group
                ic_start = g * c_in_per_group
                for oc_idx, oc in enumerate(range(oc_start, oc_start + c_out_per_group)):
                    for ic_idx, ic in enumerate(range(ic_start, ic_start + c_in_per_group)):
                        for kh in range(kernel_h):
                            for kw in range(kernel_w):
                                gs = 0.0
                                for oh in range(out_h):
                                    for ow in range(out_w):
                                        ih = oh * stride_h + kh * dil_h - pad_top
                                        iw = ow * stride_w + kw * dil_w - pad_left
                                        if 0 <= ih < in_h and 0 <= iw < in_w:
                                            gs += (
                                                output_grad[b, oc, oh, ow]
                                                * input_data[b, ic, ih, iw]
                                            )
                                weight_grad[oc, ic_idx, kh, kw] += gs

        return {
            "input_grad": input_grad,
            "weight_grad": weight_grad,
        }


class ConvGradXWPaddedOperatorTest(ConvGradXWOperatorTest):
    """ConvGradXW with asymmetric padding — uses convgradxw_padded config key."""

    def load_config(self) -> Dict[str, Any]:
        # Call grandparent load_config to get raw config dict
        from .base_operator import BaseOperatorTest

        config = BaseOperatorTest.load_config(self)
        cfg = config.get("convgradxw_padded", {})
        self.output_grad_shape = tuple(cfg.get("output_grad_shape", [1, 4, 5, 3]))
        self.input_shape = tuple(cfg.get("input_shape", [1, 2, 4, 4]))
        self.weight_shape = tuple(cfg.get("weight_shape", [4, 2, 3, 3]))
        self.input_grad_shape = tuple(cfg.get("input_grad_shape", [1, 2, 4, 4]))
        self.weight_grad_shape = tuple(cfg.get("weight_grad_shape", [4, 2, 3, 3]))
        self.kernel_shape = cfg.get("kernel_shape", [3, 3])
        self.strides = cfg.get("strides", [1, 1])
        self.pads = cfg.get("pads", [1, 0, 2, 1])
        self.dilations = cfg.get("dilations", [1, 1])
        self.group = cfg.get("group", 1)
        return config


# Legacy alias used by Onnx4Deeploy.py CLI ("ConvGradXW" → snake_case "conv_grad_xw")
ConvGradOperatorTest = ConvGradXWOperatorTest
