# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""InPlaceAccumulatorV2 operator test implementation.

ORT custom operator from com.microsoft domain used in gradient accumulation
during training. Supports lazy (deferred) reset of the accumulation buffer.

Operator semantics:
    if lazy_reset_grad:
        output_buffer = gradient          # reset: start fresh accumulation
    else:
        output_buffer = buffer + gradient  # accumulate: add to existing buffer
"""

from typing import Any, Dict

import numpy as np
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class InPlaceAccumulatorV2OperatorTest(BaseOperatorTest):
    """Test generator for ORT InPlaceAccumulatorV2 operator (com.microsoft).

    ORT Specification:
    - Description: Gradient accumulation buffer update with optional lazy reset.
                   When lazy_reset_grad is non-zero, the buffer is overwritten
                   with the new gradient (reset). Otherwise the gradient is added
                   to the existing buffer (accumulate).
    - Inputs:
        0: buffer          (T)    - current accumulation buffer (float32 tensor)
        1: gradient        (T)    - new gradient to accumulate  (float32 tensor)
        2: lazy_reset_grad (bool) - reset flag, shape [1]; non-zero means reset
    - Output:
        0: output_buffer   (T)    - updated buffer (same shape as buffer/gradient)
    - Domain: com.microsoft
    - Reference: https://github.com/microsoft/onnxruntime/blob/main/
                 orttraining/orttraining/core/graph/training_op_defs.cc
    """

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.weight_shape = None
        self.test_reset = True  # Whether to test with lazy_reset_grad=True

    def get_operator_name(self) -> str:
        return "InPlaceAccumulatorV2"

    def load_config(self) -> Dict[str, Any]:
        """Load InPlaceAccumulatorV2-specific configuration."""
        config = super().load_config()

        op_config = config.get("inplace_accumulator_v2", {})
        self.weight_shape = tuple(op_config.get("weight_shape", (128, 784)))
        self.test_reset = bool(op_config.get("test_reset", True))

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate buffer, gradient, and lazy_reset_grad test data."""
        return {
            "buffer": np.random.randn(*self.weight_shape).astype(np.float32) * 0.1,
            "gradient": np.random.randn(*self.weight_shape).astype(np.float32) * 0.01,
            # lazy_reset_grad is bool[1]: 1 = reset (out=grad), 0 = accumulate (out=buf+grad)
            "lazy_reset_grad": np.array([1 if self.test_reset else 0], dtype=np.uint8),
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for InPlaceAccumulatorV2 operator.

        Graph structure:
            [buffer, gradient, lazy_reset_grad]
              -> InPlaceAccumulatorV2 -> output  (graph output)

        The Deeploy kernel template writes to both accum_buffer (in-place) and
        data_out (explicit output pointer). This lets us use the graph output
        directly without a wrapper node, and prevents graph.cleanup() from
        eliminating the node as dead code.
        """
        weight_shape = list(self.weight_shape)

        # Input tensors
        buffer_tensor = helper.make_tensor_value_info("buffer", TensorProto.FLOAT, weight_shape)
        gradient_tensor = helper.make_tensor_value_info("gradient", TensorProto.FLOAT, weight_shape)
        # lazy_reset_grad is a bool scalar stored as uint8 (ONNX BOOL type)
        reset_tensor = helper.make_tensor_value_info("lazy_reset_grad", TensorProto.UINT8, [1])

        # Graph output tensor (directly from InPlaceAccumulatorV2)
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, weight_shape)

        # InPlaceAccumulatorV2 node (com.microsoft custom domain)
        # output is the graph output: the Deeploy kernel writes result here via data_out
        accum_node = helper.make_node(
            "InPlaceAccumulatorV2",
            inputs=["buffer", "gradient", "lazy_reset_grad"],
            outputs=["output"],
            name="inplace_accumulator_v2_node",
            domain="com.microsoft",
        )

        # Graph
        graph = helper.make_graph(
            [accum_node],
            "inplace_accumulator_v2_graph",
            [buffer_tensor, gradient_tensor, reset_tensor],
            [output_tensor],
        )

        return graph

    def create_model(self, graph, opset_version: int = 14):
        """Create ONNX model with com.microsoft custom domain."""
        model = helper.make_model(
            graph,
            producer_name="inplace_accumulator_v2_test",
            opset_imports=[
                helper.make_opsetid("", opset_version),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

        # Set output shape for 'output' (graph output from InPlaceAccumulatorV2).
        # ONNX shape inference cannot infer shapes through custom-domain ops, so
        # we explicitly set dimensions on the graph output to satisfy
        # Deeploy's _assertTensorsHaveShape() check.
        output_tensor = model.graph.output[0]
        del output_tensor.type.tensor_type.shape.dim[:]
        for dim in self.weight_shape:
            output_tensor.type.tensor_type.shape.dim.add().dim_value = dim

        return model

    def run_inference(self, onnx_file: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Skip ONNX Runtime inference (com.microsoft op not supported in standard ORT).

        Compute output directly using the NumPy reference implementation.
        """
        return self.compute_expected_output(inputs)

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute expected output using NumPy.

        InPlaceAccumulatorV2 semantics:
            if lazy_reset_grad: output = gradient          (reset)
            else:               output = buffer + gradient  (accumulate)
        """
        buffer = inputs["buffer"]
        gradient = inputs["gradient"]
        lazy_reset_grad = inputs["lazy_reset_grad"]

        if lazy_reset_grad[0]:
            output = gradient.copy()
        else:
            output = buffer + gradient

        return {"output": output}
