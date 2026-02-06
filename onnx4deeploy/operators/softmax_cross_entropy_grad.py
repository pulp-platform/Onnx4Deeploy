# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""SoftmaxCrossEntropyGrad operator test implementation."""

from typing import Any, Dict

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class SoftmaxCrossEntropyGradOperatorTest(BaseOperatorTest):
    """Test generator for ONNX SoftmaxCrossEntropyLossGrad operator (custom/training op)."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.num_classes = None
        self.batch_size = None

    def get_operator_name(self) -> str:
        return "SoftmaxCrossEntropyLossGrad"

    def load_config(self) -> Dict[str, Any]:
        """Load SoftmaxCrossEntropyGrad-specific configuration."""
        config = super().load_config()

        sceg_config = config.get("softmax_cross_entropy_grad", {})
        self.input_shape = tuple(sceg_config["input_shape"])
        self.num_classes = self.input_shape[-1]
        self.batch_size = self.input_shape[0]

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random log_prob, labels, and loss_grad."""
        # Generate logits and convert to log probabilities
        logits = np.random.randn(*self.input_shape).astype(np.float32)
        softmax_prob = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        log_prob = np.log(softmax_prob)

        return {
            "log_prob": log_prob,
            "labels": np.random.randint(0, self.num_classes, size=(self.batch_size,)).astype(
                np.int64
            ),
            "loss_grad": np.ones((self.batch_size,), dtype=np.float32) / self.batch_size,
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for SoftmaxCrossEntropyLossGrad operator."""
        # Input tensors (without loss_grad for the final model)
        log_prob_tensor = helper.make_tensor_value_info(
            "log_prob", TensorProto.FLOAT, self.input_shape
        )
        labels_tensor = helper.make_tensor_value_info(
            "labels", TensorProto.INT64, [self.batch_size]
        )

        # Output tensor
        output_grad_tensor = helper.make_tensor_value_info(
            "output_grad", TensorProto.FLOAT, self.input_shape
        )

        # SoftmaxCrossEntropyLossGrad node (without loss_grad input)
        grad_node = helper.make_node(
            "SoftmaxCrossEntropyLossGrad",
            inputs=["log_prob", "labels"],
            outputs=["output_grad"],
            name="softmax_cross_entropy_grad_node",
            domain="com.microsoft",
            reduction="mean",
        )

        # Graph
        graph = helper.make_graph(
            [grad_node],
            "softmax_cross_entropy_grad_graph",
            [log_prob_tensor, labels_tensor],
            [output_grad_tensor],
        )

        return graph

    def create_model(self, graph, opset_version: int = 13):
        """Create ONNX model for SoftmaxCrossEntropyLossGrad with custom domain."""
        model = helper.make_model(
            graph,
            producer_name=f"{self.get_operator_name().lower()}_test",
            opset_imports=[
                helper.make_opsetid("", opset_version),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )

        return model

    def run_inference(self, onnx_file: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference using ONNX Runtime.

        Override to create complete model with loss_grad for inference.
        """
        # Create complete model with all inputs including loss_grad
        loss_grad_tensor = helper.make_tensor_value_info(
            "loss_grad", TensorProto.FLOAT, [self.batch_size]
        )
        log_prob_tensor = helper.make_tensor_value_info(
            "log_prob", TensorProto.FLOAT, self.input_shape
        )
        labels_tensor = helper.make_tensor_value_info(
            "labels", TensorProto.INT64, [self.batch_size]
        )
        output_grad_tensor = helper.make_tensor_value_info(
            "output_grad", TensorProto.FLOAT, self.input_shape
        )

        grad_node = helper.make_node(
            "SoftmaxCrossEntropyLossGrad",
            inputs=["loss_grad", "log_prob", "labels"],
            outputs=["output_grad"],
            name="softmax_cross_entropy_grad_node",
            domain="com.microsoft",
            reduction="mean",
        )

        complete_graph = helper.make_graph(
            [grad_node],
            "softmax_cross_entropy_grad_complete_graph",
            [loss_grad_tensor, log_prob_tensor, labels_tensor],
            [output_grad_tensor],
        )

        complete_model = helper.make_model(
            complete_graph,
            producer_name="softmax_cross_entropy_grad_complete",
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("com.microsoft", 1)],
        )

        # Run inference on complete model
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        session = ort.InferenceSession(complete_model.SerializeToString(), sess_options)
        outputs = session.run(None, inputs)

        return {"output_grad": outputs[0]}

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Return None to skip validation - this is a custom operator.
        """
        return None
