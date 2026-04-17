# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Inference consistency tests: onnxruntime vs pure-Python run_onnx_graph.

For each model exported in inference mode, we verify that our pure-Python
ONNX executor (``run_onnx_graph``) produces numerically identical results
to onnxruntime's ``InferenceSession`` on the same random inputs.

Models tested:
  - LightweightCNN  (standard ops only → ORT + run_onnx_graph agree)
  - SleepConViT     (contains com.microsoft/Gelu and a Squeeze with opset-12
                     axes attribute; patched to opset-13 in-memory before ORT)
"""

import os
import subprocess
import sys

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
import pytest

from .onnx_node_implementations import run_onnx_graph
from .test_utils import load_and_check_onnx_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_CLI_SCRIPT = os.path.join(_PROJECT_ROOT, "Onnx4Deeploy.py")

_NUM_SAMPLES = 5
_TOLERANCE = 1e-5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _export_inference(model_name: str, output_dir: str) -> str:
    """Run the CLI in 'infer' mode and return the path to network.onnx."""
    cmd = [
        sys.executable, _CLI_SCRIPT,
        "-model", model_name,
        "-mode", "infer",
        "-o", output_dir,
    ]
    result = subprocess.run(
        cmd, cwd=_PROJECT_ROOT, capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.fail(
            f"CLI failed for '{model_name}' (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    onnx_file = os.path.join(output_dir, "network.onnx")
    assert os.path.exists(onnx_file), f"network.onnx not found in {output_dir}"
    return onnx_file


def _to_opset13_compatible(onnx_file: str) -> bytes:
    """
    Return a serialised ONNX model with all Squeeze/Unsqueeze nodes patched to
    opset-13 style: the ``axes`` *attribute* is moved to a constant *input*
    tensor so that onnxruntime (which validates opset-13+ rules) can load it.

    The original file on disk is not modified.
    """
    model = onnx.load(onnx_file)
    new_nodes = []
    extra_initializers = []

    for node in model.graph.node:
        if node.op_type not in ("Squeeze", "Unsqueeze"):
            new_nodes.append(node)
            continue

        # Find the axes attribute (opset-12 style)
        axes_attr = next((a for a in node.attribute if a.name == "axes"), None)
        if axes_attr is None:
            # Already opset-13 style (axes come as a second input) or no axes
            new_nodes.append(node)
            continue

        # Extract axes value(s)
        if axes_attr.type == onnx.AttributeProto.INT:
            axes = [int(axes_attr.i)]
        else:  # INTS
            axes = list(axes_attr.ints)

        # Create a constant initializer for the axes tensor
        axes_name = f"_axes_const_{node.name}"
        axes_tensor = onnx.helper.make_tensor(
            name=axes_name,
            data_type=onnx.TensorProto.INT64,
            dims=[len(axes)],
            vals=axes,
        )
        extra_initializers.append(axes_tensor)

        # Rebuild the node with axes as the second input, no axes attribute
        new_node = onnx.helper.make_node(
            op_type=node.op_type,
            inputs=[node.input[0], axes_name],
            outputs=list(node.output),
            name=node.name,
            domain=node.domain if node.domain else "",
        )
        # Copy over any other attributes (not axes)
        for attr in node.attribute:
            if attr.name != "axes":
                new_node.attribute.append(attr)

        new_nodes.append(new_node)

    # Rebuild graph with patched nodes and extra initializers
    new_graph = onnx.helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=list(model.graph.input),
        outputs=list(model.graph.output),
        initializer=list(model.graph.initializer) + extra_initializers,
    )
    for vi in model.graph.value_info:
        new_graph.value_info.append(vi)

    new_model = onnx.helper.make_model(
        new_graph,
        producer_name=model.producer_name,
        opset_imports=list(model.opset_import),
    )
    new_model.ir_version = model.ir_version
    return new_model.SerializeToString()


def _ort_session(onnx_file: str) -> ort.InferenceSession:
    """Create an onnxruntime InferenceSession, patching Squeeze axes to opset-13."""
    return ort.InferenceSession(_to_opset13_compatible(onnx_file))


def _compare_outputs(
    onnx_file: str,
    input_shape: tuple,
    num_samples: int = _NUM_SAMPLES,
    tolerance: float = _TOLERANCE,
) -> None:
    """
    Run *num_samples* random inputs through both onnxruntime and run_onnx_graph
    and assert the outputs agree within *tolerance*.

    ORT receives the opset-13-patched model bytes; run_onnx_graph reads the
    original file (our executor handles both styles).
    """
    sess = _ort_session(onnx_file)
    ort_input_name = sess.get_inputs()[0].name

    failures = []
    for seed in range(num_samples):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(input_shape).astype(np.float32)

        # onnxruntime reference
        ort_out = sess.run(None, {ort_input_name: x})[0]

        # pure-Python executor
        py_out = run_onnx_graph(onnx_file, {"input": x})

        max_diff = float(np.max(np.abs(ort_out - py_out)))
        if max_diff > tolerance:
            failures.append(
                f"  seed={seed}: max |ORT − run_onnx_graph| = {max_diff:.2e} "
                f"(limit {tolerance:.2e})"
            )

    if failures:
        pytest.fail(
            f"Outputs diverge on {len(failures)}/{num_samples} samples:\n"
            + "\n".join(failures)
        )


# ===========================================================================
# LightweightCNN
# ===========================================================================


class TestLightweightCNNInferenceConsistency:
    """
    LightweightCNN uses only standard ONNX ops (Conv, MaxPool, Relu, Gemm,
    Reshape), so both onnxruntime and run_onnx_graph can execute it.
    This test verifies they agree numerically.
    """

    _MODEL = "LightweightCNN"
    _INPUT_SHAPE = (1, 1, 28, 28)

    def test_export_produces_valid_onnx(self, model_test_dir):
        """Exported network.onnx loads and has correct structure."""
        onnx_file = _export_inference(self._MODEL, model_test_dir)
        model = load_and_check_onnx_model(onnx_file, skip_shape_check=True)
        assert len(model.graph.node) > 0
        assert any(o.name for o in model.graph.output)

    def test_ort_and_pure_python_agree(self, model_test_dir):
        """
        onnxruntime and run_onnx_graph produce identical outputs (within 1e-5)
        on 5 random inputs for LightweightCNN.
        """
        onnx_file = _export_inference(self._MODEL, model_test_dir)
        _compare_outputs(onnx_file, self._INPUT_SHAPE)

    def test_run_onnx_graph_produces_finite_output(self, model_test_dir):
        """run_onnx_graph produces finite float32 output for each random input."""
        onnx_file = _export_inference(self._MODEL, model_test_dir)
        for seed in range(_NUM_SAMPLES):
            rng = np.random.default_rng(seed)
            x = rng.standard_normal(self._INPUT_SHAPE).astype(np.float32)
            out = run_onnx_graph(onnx_file, {"input": x})
            assert out.shape == (1, 10), f"Unexpected output shape: {out.shape}"
            assert np.all(np.isfinite(out)), f"Non-finite output at seed={seed}"

    def test_run_onnx_graph_output_shape(self, model_test_dir):
        """Output shape is (1, num_classes=10)."""
        onnx_file = _export_inference(self._MODEL, model_test_dir)
        x = np.zeros(self._INPUT_SHAPE, dtype=np.float32)
        out = run_onnx_graph(onnx_file, {"input": x})
        assert out.shape == (1, 10)


# ===========================================================================
# SleepConViT
# ===========================================================================


class TestSleepConViTInferenceConsistency:
    """
    SleepConViT uses com.microsoft/Gelu and a Squeeze node with an ``axes``
    attribute (opset-12 style).  Before handing the model to ORT the test
    patches Squeeze/Unsqueeze nodes in-memory via ``_to_opset13_compatible``
    so that onnxruntime can load and execute it.
    """

    _MODEL = "SleepConViT"
    _INPUT_SHAPE = (1, 1, 1, 3000)
    _NUM_CLASSES = 4

    def test_export_produces_valid_onnx(self, model_test_dir):
        """Exported network.onnx loads without error."""
        onnx_file = _export_inference(self._MODEL, model_test_dir)
        model = load_and_check_onnx_model(onnx_file, skip_shape_check=True)
        assert len(model.graph.node) > 0

    def test_run_onnx_graph_produces_finite_output(self, model_test_dir):
        """
        run_onnx_graph produces finite float32 output for each random input.
        This exercises com.microsoft/Gelu and the Squeeze op in our executor.
        """
        onnx_file = _export_inference(self._MODEL, model_test_dir)
        for seed in range(_NUM_SAMPLES):
            rng = np.random.default_rng(seed)
            x = rng.standard_normal(self._INPUT_SHAPE).astype(np.float32)
            out = run_onnx_graph(onnx_file, {"input": x})
            assert np.all(np.isfinite(out)), f"Non-finite output at seed={seed}"

    def test_run_onnx_graph_output_shape(self, model_test_dir):
        """Output shape is (1, num_classes)."""
        onnx_file = _export_inference(self._MODEL, model_test_dir)
        x = np.zeros(self._INPUT_SHAPE, dtype=np.float32)
        out = run_onnx_graph(onnx_file, {"input": x})
        assert out.shape[0] == 1
        assert out.shape[-1] == self._NUM_CLASSES

    def test_run_onnx_graph_deterministic(self, model_test_dir):
        """run_onnx_graph produces the same result on repeated calls."""
        onnx_file = _export_inference(self._MODEL, model_test_dir)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(self._INPUT_SHAPE).astype(np.float32)
        out1 = run_onnx_graph(onnx_file, {"input": x})
        out2 = run_onnx_graph(onnx_file, {"input": x})
        np.testing.assert_array_equal(out1, out2)

    def test_ort_and_pure_python_agree(self, model_test_dir):
        """
        onnxruntime and run_onnx_graph produce identical outputs (within 1e-5)
        on 5 random inputs for SleepConViT.

        The model is patched in-memory via ``_to_opset13_compatible`` before
        being passed to ORT so that the opset-12-style Squeeze axes attribute
        does not cause a load failure.
        """
        onnx_file = _export_inference(self._MODEL, model_test_dir)
        _compare_outputs(onnx_file, self._INPUT_SHAPE)
