# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""ConvGradX operator test implementation."""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class ConvGradXOperatorTest(BaseOperatorTest):
    """Test generator for ConvGradX operator (custom op - gradient w.r.t. input)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.output_grad_shape = None
        self.weight_shape = None
        self.input_grad_shape = None
        self.kernel_shape = [3, 3]
        self.strides = [1, 1]
        self.pads = [1, 1, 1, 1]
        self.dilations = [1, 1]
        self.group = 1

    def get_operator_name(self) -> str:
        return "ConvGradX"

    def load_config(self) -> Dict[str, Any]:
        """Load ConvGradX-specific configuration."""
        config = super().load_config()

        cgx_config = config.get("convgradx", {})
        self.output_grad_shape = tuple(cgx_config.get("output_grad_shape", [1, 8, 8, 8]))
        self.weight_shape = tuple(cgx_config.get("weight_shape", [8, 16, 3, 3]))
        self.input_grad_shape = tuple(cgx_config.get("input_grad_shape", [1, 16, 8, 8]))
        self.kernel_shape = cgx_config.get("kernel_shape", [3, 3])
        self.strides = cgx_config.get("strides", [1, 1])
        self.pads = cgx_config.get("pads", [1, 1, 1, 1])
        self.dilations = cgx_config.get("dilations", [1, 1])
        self.group = cgx_config.get("group", 1)

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random gradient and weight data."""
        np.random.seed(42)
        return {
            "output_grad": np.random.randn(*self.output_grad_shape).astype(np.float32),
            "weight": np.random.randn(*self.weight_shape).astype(np.float32),
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for ConvGradX operator."""
        # Input tensors
        output_grad_tensor = helper.make_tensor_value_info(
            "output_grad", TensorProto.FLOAT, self.output_grad_shape
        )
        weight_tensor = helper.make_tensor_value_info(
            "weight", TensorProto.FLOAT, self.weight_shape
        )

        # Output tensor
        input_grad_tensor = helper.make_tensor_value_info(
            "input_grad", TensorProto.FLOAT, self.input_grad_shape
        )

        # ConvGradX node
        convgradx_node = helper.make_node(
            "ConvGradX",
            inputs=["output_grad", "weight"],
            outputs=["input_grad"],
            name="convgradx_node",
            kernel_shape=self.kernel_shape,
            strides=self.strides,
            pads=self.pads,
            dilations=self.dilations,
            group=self.group,
        )

        # Graph (weight is now a graph input, not an initializer)
        graph = helper.make_graph(
            [convgradx_node],
            "convgradx_graph",
            [output_grad_tensor, weight_tensor],
            [input_grad_tensor],
        )

        return graph

    def create_model(self, graph, opset_version: int = 13):
        """Create ONNX model for ConvGradX."""
        model = helper.make_model(
            graph,
            producer_name=f"{self.get_operator_name().lower()}_test",
            opset_imports=[helper.make_opsetid("", opset_version)],
        )

        # Manually set output shape
        output_tensor = model.graph.output[0]
        del output_tensor.type.tensor_type.shape.dim[:]
        for dim in self.input_grad_shape:
            output_tensor.type.tensor_type.shape.dim.add().dim_value = dim

        return model

    def run_inference(self, onnx_file: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Skip ONNX Runtime inference for custom operators.

        Compute output directly using NumPy.
        """
        return self.compute_expected_output(inputs)

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute expected output using NumPy."""
        output_grad = inputs["output_grad"]
        weight = inputs["weight"]

        batch, out_ch, out_h, out_w = self.output_grad_shape
        out_ch_w, in_ch_per_group_from_weight, kernel_h, kernel_w = self.weight_shape
        batch_in, in_ch, in_h, in_w = self.input_grad_shape
        stride_h, stride_w = self.strides
        dilation_h, dilation_w = self.dilations
        pad_top, pad_left, pad_bottom, pad_right = self.pads

        input_grad = np.zeros(self.input_grad_shape, dtype=np.float32)

        # Calculate channels per group
        in_ch_per_group = in_ch // self.group
        out_ch_per_group = out_ch // self.group

        for b in range(batch):
            for g in range(self.group):
                # Calculate channel ranges for this group
                ic_start = g * in_ch_per_group
                ic_end = (g + 1) * in_ch_per_group
                oc_start = g * out_ch_per_group
                oc_end = (g + 1) * out_ch_per_group

                for ic in range(ic_start, ic_end):
                    for ih in range(in_h):
                        for iw in range(in_w):
                            grad_sum = 0.0

                            for oc in range(oc_start, oc_end):
                                for kh in range(kernel_h):
                                    for kw in range(kernel_w):
                                        oh_candidate = ih + pad_top - kh * dilation_h
                                        ow_candidate = iw + pad_left - kw * dilation_w

                                        if (
                                            oh_candidate % stride_h == 0
                                            and ow_candidate % stride_w == 0
                                        ):
                                            oh = oh_candidate // stride_h
                                            ow = ow_candidate // stride_w

                                            if 0 <= oh < out_h and 0 <= ow < out_w:
                                                ic_in_group = ic - ic_start
                                                grad_sum += (
                                                    output_grad[b, oc, oh, ow]
                                                    * weight[oc, ic_in_group, kh, kw]
                                                )

                            input_grad[b, ic, ih, iw] = grad_sum

        return {"input_grad": input_grad}
