# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""SoftmaxCrossEntropyLoss dual-output operator test.

This tests the two-output variant of SoftmaxCrossEntropyLoss:
  (loss, log_prob) = SoftmaxCrossEntropyLoss(logits, labels, reduction='mean')

The single-output variant (softmax_cross_entropy.py) returns only log_prob.
This variant returns both the scalar mean CE loss AND the log-probability tensor.

ONNX Specification (opset 13):
  Inputs:
    - scores  (T):  [batch, num_classes] logit tensor
    - labels  (Tind): [batch] int64 class indices
    - weights (T):  optional per-class weights (not used here)
  Outputs:
    - output  (T):  [] scalar mean CE loss
    - log_prob (T): [batch, num_classes] log-softmax values
  Attributes:
    - reduction (string, default='mean'): 'none'|'mean'|'sum'
    - ignore_index (int, default=-1): label value to ignore
  Reference: https://onnx.ai/onnx/operators/onnx__SoftmaxCrossEntropyLoss.html

Deeploy implementation notes:
  - Forward: referenceDualOutputTemplate in SoftmaxCrossEntropyLossTemplate.py
  - Mapper:  SoftmaxCrossEntropyLossDualOutputMapper in Platform.py
  - Parser:  SoftmaxCrossEntropyLossParser (shared, accepts 1 or 2 outputs)
  - Checker: SoftmaxCrossEntropyLossChecker (handles both output counts)
"""

from typing import Any, Dict

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

from .base_operator import BaseOperatorTest


class SoftmaxCrossEntropyDualOutputOperatorTest(BaseOperatorTest):
    """Test generator for ONNX SoftmaxCrossEntropyLoss with both outputs.

    Generates a graph that exposes both the scalar mean-CE loss and the
    log-softmax tensor.  ONNX Runtime is used as the reference implementation
    for both outputs.
    """

    def __init__(self, config_path=None, save_path=None):
        super().__init__(config_path, save_path)
        self.input_shape = None
        self.num_classes = None
        self.batch_size = None

    def get_operator_name(self) -> str:
        return "SoftmaxCrossEntropyLossDualOutput"

    def load_config(self) -> Dict[str, Any]:
        config = super().load_config()

        sce_config = config.get("softmax_cross_entropy_dual_output", {})
        self.input_shape = tuple(sce_config["input_shape"])
        self.num_classes = self.input_shape[-1]
        self.batch_size = self.input_shape[0]

        return config

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        return {
            "logits": np.random.randn(*self.input_shape).astype(np.float32),
            "labels": np.random.randint(0, self.num_classes, size=(self.batch_size,)).astype(
                np.int64
            ),
        }

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]):
        """Create ONNX graph with both loss and log_prob outputs."""
        logits_tensor = helper.make_tensor_value_info(
            "logits", TensorProto.FLOAT, list(self.input_shape)
        )
        labels_tensor = helper.make_tensor_value_info(
            "labels", TensorProto.INT64, [self.batch_size]
        )

        # Both outputs exposed
        loss_tensor = helper.make_tensor_value_info("loss", TensorProto.FLOAT, [])
        log_prob_tensor = helper.make_tensor_value_info(
            "log_prob", TensorProto.FLOAT, list(self.input_shape)
        )

        loss_node = helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["logits", "labels"],
            outputs=["loss", "log_prob"],
            name="softmax_cross_entropy_dual_output_node",
            reduction="mean",
        )

        graph = helper.make_graph(
            [loss_node],
            "softmax_cross_entropy_dual_output_graph",
            [logits_tensor, labels_tensor],
            [loss_tensor, log_prob_tensor],
        )

        return graph

    def run_inference(self, onnx_file: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run ORT on the dual-output graph, return both outputs."""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        session = ort.InferenceSession(onnx_file, sess_options)
        loss_val, log_prob_val = session.run(None, inputs)

        return {
            "loss": loss_val,  # scalar float32
            "log_prob": log_prob_val,  # [batch, num_classes] float32
        }

    def compute_expected_output(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """NumPy reference for validation (optional cross-check).

        Forward formula (same as referenceDualOutputTemplate in Deeploy):
          log_prob[i,j] = logits[i,j] - max(logits[i]) - log(sum_j exp(logits[i,j]-max))
          loss           = mean_i(-log_prob[i, labels[i]])
        """
        logits = inputs["logits"].astype(np.float64)  # use float64 for precision
        labels = inputs["labels"]

        max_logit = logits.max(axis=-1, keepdims=True)
        shifted = logits - max_logit
        log_sum_exp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        log_prob = (shifted - log_sum_exp).astype(np.float32)

        nll = -log_prob[np.arange(self.batch_size), labels]
        loss = nll.mean().astype(np.float32).reshape([])

        return {"loss": loss, "log_prob": log_prob}
