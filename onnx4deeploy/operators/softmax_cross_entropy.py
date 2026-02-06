# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""SoftmaxCrossEntropy operator test implementation."""

from typing import Any, Dict

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class SoftmaxCrossEntropyOperatorTest(BaseOperatorTest):
    """Test generator for ONNX SoftmaxCrossEntropyLoss operator."""

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.num_classes = None
        self.batch_size = None

    def get_operator_name(self) -> str:
        return "SoftmaxCrossEntropyLoss"

    def load_config(self) -> Dict[str, Any]:
        """Load SoftmaxCrossEntropy-specific configuration."""
        config = super().load_config()

        sce_config = config.get("softmax_cross_entropy", {})
        self.input_shape = tuple(sce_config["input_shape"])
        self.num_classes = self.input_shape[-1]
        self.batch_size = self.input_shape[0]

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        """Generate random logits and labels."""
        return {
            "logits": np.random.randn(*self.input_shape).astype(np.float32),
            "labels": np.random.randint(0, self.num_classes, size=(self.batch_size,)).astype(
                np.int64
            ),
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph for SoftmaxCrossEntropyLoss operator."""
        # Input tensors
        logits_tensor = helper.make_tensor_value_info("logits", TensorProto.FLOAT, self.input_shape)
        labels_tensor = helper.make_tensor_value_info(
            "labels", TensorProto.INT64, [self.batch_size]
        )

        # Output tensor (only log_prob, not loss)
        log_prob_tensor = helper.make_tensor_value_info(
            "log_prob", TensorProto.FLOAT, self.input_shape
        )

        # SoftmaxCrossEntropyLoss node
        # We create with both outputs, then remove loss output in create_model
        loss_node = helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["logits", "labels"],
            outputs=["log_prob"],  # Only log_prob output
            name="softmax_cross_entropy_node",
            reduction="mean",
        )

        # Graph
        graph = helper.make_graph(
            [loss_node],
            "softmax_cross_entropy_graph",
            [logits_tensor, labels_tensor],
            [log_prob_tensor],
        )

        return graph

    def run_inference(self, onnx_file: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference using ONNX Runtime.

        Override to handle the special case where we need both outputs for inference
        but only save log_prob.
        """
        # Create complete model with both outputs for inference
        logits_tensor = helper.make_tensor_value_info("logits", TensorProto.FLOAT, self.input_shape)
        labels_tensor = helper.make_tensor_value_info(
            "labels", TensorProto.INT64, [self.batch_size]
        )

        loss_tensor = helper.make_tensor_value_info("loss", TensorProto.FLOAT, [])
        log_prob_tensor = helper.make_tensor_value_info(
            "log_prob", TensorProto.FLOAT, self.input_shape
        )

        loss_node = helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["logits", "labels"],
            outputs=["loss", "log_prob"],
            name="softmax_cross_entropy_node",
            reduction="mean",
        )

        complete_graph = helper.make_graph(
            [loss_node],
            "softmax_cross_entropy_complete_graph",
            [logits_tensor, labels_tensor],
            [loss_tensor, log_prob_tensor],
        )

        complete_model = helper.make_model(
            complete_graph,
            producer_name="softmax_cross_entropy_complete",
            opset_imports=[helper.make_opsetid("", 13)],
        )

        # Run inference on complete model
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        session = ort.InferenceSession(complete_model.SerializeToString(), sess_options)
        outputs = session.run(None, inputs)

        # Return only log_prob
        return {"log_prob": outputs[1]}

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Return None to skip validation - we use ONNX Runtime as ground truth.
        """
        return None
