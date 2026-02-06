"""ConvGradW operator test implementation."""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class ConvGradWOperatorTest(BaseOperatorTest):
    """Test generator for ConvGradW operator (custom op - gradient w.r.t. weight)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.output_grad_shape = None
        self.input_shape = None
        self.weight_grad_shape = None
        self.kernel_shape = [2, 2]
        self.strides = [1, 1]
        self.pads = [1, 1, 1, 1]
        self.dilations = [1, 1]
        self.group = 1

    def get_operator_name(self) -> str:
        return "ConvGradW"

    def load_config(self) -> Dict[str, Any]:
        """Load ConvGradW-specific configuration."""
        config = super().load_config()

        cgw_config = config.get("convgradw", {})
        self.output_grad_shape = tuple(cgw_config.get("output_grad_shape", [1, 2, 5, 5]))
        self.input_shape = tuple(cgw_config.get("input_shape", [1, 1, 2, 2]))
        self.weight_grad_shape = tuple(cgw_config.get("weight_grad_shape", [2, 1, 2, 2]))
        self.kernel_shape = cgw_config.get("kernel_shape", [2, 2])
        self.strides = cgw_config.get("strides", [1, 1])
        self.pads = cgw_config.get("pads", [1, 1, 1, 1])
        self.dilations = cgw_config.get("dilations", [1, 1])
        self.group = cgw_config.get("group", 1)

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random gradient and input data."""
        return {
            "output_grad": np.random.randn(*self.output_grad_shape).astype(np.float32),
            "input_data": np.random.randn(*self.input_shape).astype(np.float32),
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for ConvGradW operator."""
        # Input tensors
        output_grad_tensor = helper.make_tensor_value_info(
            "output_grad", TensorProto.FLOAT, self.output_grad_shape
        )
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, self.input_shape)

        # Output tensor
        weight_grad_tensor = helper.make_tensor_value_info(
            "weight_grad", TensorProto.FLOAT, self.weight_grad_shape
        )

        # ConvGradW node
        convgradw_node = helper.make_node(
            "ConvGradW",
            inputs=["output_grad", "input"],
            outputs=["weight_grad"],
            name="convgradw_node",
            kernel_shape=self.kernel_shape,
            strides=self.strides,
            pads=self.pads,
            dilations=self.dilations,
            group=self.group,
        )

        # Graph
        graph = helper.make_graph(
            [convgradw_node],
            "convgradw_graph",
            [output_grad_tensor, input_tensor],
            [weight_grad_tensor],
        )

        return graph

    def create_model(self, graph, opset_version: int = 13):
        """Create ONNX model for ConvGradW."""
        model = helper.make_model(
            graph,
            producer_name=f"{self.get_operator_name().lower()}_test",
            opset_imports=[helper.make_opsetid("", opset_version)],
        )

        # Manually set output shape
        output_tensor = model.graph.output[0]
        del output_tensor.type.tensor_type.shape.dim[:]
        for dim in self.weight_grad_shape:
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
        input_data = inputs["input_data"]

        batch, c_out, out_h, out_w = self.output_grad_shape
        _, c_in, in_h, in_w = self.input_shape
        out_ch_total, in_ch_per_group, kernel_h, kernel_w = self.weight_grad_shape
        stride_h, stride_w = self.strides
        dilation_h, dilation_w = self.dilations
        pad_top, pad_left, pad_bottom, pad_right = self.pads

        weight_grad = np.zeros(self.weight_grad_shape, dtype=np.float32)

        c_out_per_group = c_out // self.group
        c_in_per_group = c_in // self.group

        for b in range(batch):
            for g in range(self.group):
                oc_start = g * c_out_per_group
                oc_end = (g + 1) * c_out_per_group

                ic_start = g * c_in_per_group
                ic_end = (g + 1) * c_in_per_group

                for oc_idx, oc in enumerate(range(oc_start, oc_end)):
                    for ic_idx, ic in enumerate(range(ic_start, ic_end)):
                        for kh in range(kernel_h):
                            for kw in range(kernel_w):
                                grad_sum = 0.0

                                for oh in range(out_h):
                                    for ow in range(out_w):
                                        ih = oh * stride_h + kh * dilation_h - pad_top
                                        iw = ow * stride_w + kw * dilation_w - pad_left

                                        if 0 <= ih < in_h and 0 <= iw < in_w:
                                            grad_sum += (
                                                output_grad[b, oc, oh, ow]
                                                * input_data[b, ic, ih, iw]
                                            )

                                weight_grad[oc, ic_idx, kh, kw] = grad_sum

        return {"weight_grad": weight_grad}
