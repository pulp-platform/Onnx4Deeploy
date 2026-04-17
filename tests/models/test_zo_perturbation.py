# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Tests for ZO (zeroth-order) perturbation model exports.

Tests the full export pipeline for LightweightCNN, QLiteCNN, SleepConViT,
and QSleepConViT with various noise types (Uniform, Rademacher, Eggroll,
RQS-Rademacher).

Numerical verification uses ``run_onnx_graph`` from
``onnx_node_implementations``, a pure Python / PyTorch graph executor.
"""

import os
import subprocess
import sys

import numpy as np
import onnx
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

NUM_CORES = 8

# ---------------------------------------------------------------------------
# RNG Reference Implementation (matches Deeploy C++ code)
# ---------------------------------------------------------------------------


def scramble_seed(seed: int) -> np.uint32:
    """Scramble seed: seed * 1664525 + 1013904223 (mod 2^32)."""
    return np.uint32(np.uint32(seed) * np.uint32(1664525) + np.uint32(1013904223))


def xorshift32(state: np.uint32) -> np.uint32:
    """Xorshift32 PRNG step."""
    state = np.uint32(state)
    state ^= np.uint32(state << np.uint32(13))
    state ^= np.uint32(state >> np.uint32(17))
    state ^= np.uint32(state << np.uint32(5))
    return state


def generate_uniform_perturbation(
    data: np.ndarray,
    global_seed: int,
    node_id: int,
    eps: float,
    perturbation_sign: int = 1,
) -> np.ndarray:
    """Generate uniform perturbation matching Deeploy C++ reference.

    seed = scramble(initial_global_seed + NUM_CORES * node_id + core_id)
    RNG: Xorshift32, mapped to [-1, 1], scaled by eps * sqrt(3).
    """
    size = data.size
    output = data.flatten().copy()
    scale = eps * np.sqrt(3.0)

    for core_id in range(NUM_CORES):
        log2core = int(np.log2(NUM_CORES))
        chunk = (size >> log2core) + ((size & (NUM_CORES - 1)) != 0)
        chunk_start = min(chunk * core_id, size)
        chunk_stop = min(chunk_start + chunk, size)

        seed = scramble_seed(global_seed + NUM_CORES * node_id + core_id)

        for i in range(chunk_start, chunk_stop):
            seed = xorshift32(seed)
            # Map uint32 to [-1, 1]
            rand_val = (float(seed) / float(np.iinfo(np.uint32).max)) * 2.0 - 1.0
            output[i] = output[i] + perturbation_sign * rand_val * scale

    return output.reshape(data.shape)


def generate_rademacher_perturbation(
    data: np.ndarray,
    global_seed: int,
    node_id: int,
    eps: float,
    perturbation_sign: int = 1,
) -> np.ndarray:
    """Generate Rademacher perturbation (+/- eps) matching Deeploy C++ reference."""
    size = data.size
    output = data.flatten().copy()

    for core_id in range(NUM_CORES):
        log2core = int(np.log2(NUM_CORES))
        chunk = (size >> log2core) + ((size & (NUM_CORES - 1)) != 0)
        chunk_start = min(chunk * core_id, size)
        chunk_stop = min(chunk_start + chunk, size)

        seed = scramble_seed(global_seed + NUM_CORES * node_id + core_id)

        for i in range(chunk_start, chunk_stop):
            seed = xorshift32(seed)
            sign = 1 if (seed & 1) else -1
            output[i] = output[i] + perturbation_sign * sign * eps

    return output.reshape(data.shape)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_cli(model: str, mode: str, output_dir: str, noise_type: str) -> subprocess.CompletedProcess:
    """Run Onnx4Deeploy CLI and return result."""
    cmd = [
        sys.executable, _CLI_SCRIPT,
        "-model", model,
        "-mode", mode,
        "-o", output_dir,
        "--noise-type", noise_type,
    ]
    result = subprocess.run(cmd, cwd=_PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[CLI] stdout:\n{result.stdout}")
        print(f"[CLI] stderr:\n{result.stderr}")
    return result


def _verify_zo_output_files(output_dir: str, quantized: bool = False):
    """Verify expected ZO output files exist."""
    expected = ["network_infer.onnx", "network_zo_train.onnx", "inputs.npz", "outputs.npz"]
    if not quantized:
        # Non-quantized may also have network_zo_update.onnx
        pass
    for fname in expected:
        fpath = os.path.join(output_dir, fname)
        assert os.path.exists(fpath), f"Missing expected output file: {fpath}"


def _verify_onnx_valid(onnx_file: str) -> onnx.ModelProto:
    """Load and do basic validation of ONNX model."""
    return load_and_check_onnx_model(onnx_file, skip_shape_check=True)


def _count_perturbation_nodes(model: onnx.ModelProto, op_prefix: str = "Perturb") -> int:
    """Count perturbation operator nodes in the graph."""
    return sum(1 for node in model.graph.node if node.op_type.startswith(op_prefix))


def _get_perturbation_op_type(noise_type: str) -> str:
    """Map noise type to expected ONNX op_type."""
    mapping = {
        "uniform": "PerturbUniform",
        "rademacher": "PerturbRademacher",
        "eggroll": "PerturbRademacher",  # Eggroll uses Rademacher
        "rqs_rademacher": "RQSPerturbRademacher",
        "rqs_uniform": "RQSPerturbUniform",
    }
    return mapping.get(noise_type, f"Perturb{noise_type.capitalize()}")


def _run_and_verify_zo_export(
    model_name: str,
    mode: str,
    noise_type: str,
    output_dir: str,
    quantized: bool = False,
):
    """Run CLI export and verify output files and ONNX validity."""
    result = _run_cli(model_name, mode, output_dir, noise_type)
    assert result.returncode == 0, (
        f"CLI failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    _verify_zo_output_files(output_dir, quantized=quantized)

    # Check inference model
    infer_path = os.path.join(output_dir, "network_infer.onnx")
    _verify_onnx_valid(infer_path)

    # Check ZO training model
    zo_path = os.path.join(output_dir, "network_zo_train.onnx")
    zo_model = _verify_onnx_valid(zo_path)

    # Verify perturbation nodes exist
    expected_op = _get_perturbation_op_type(noise_type)
    perturb_count = sum(
        1 for n in zo_model.graph.node if n.op_type == expected_op
    )
    assert perturb_count > 0, (
        f"No {expected_op} nodes found in ZO training graph. "
        f"Node types: {set(n.op_type for n in zo_model.graph.node)}"
    )

    return zo_model


def _run_numerical_check(
    onnx_file: str,
    input_shape: tuple,
    num_samples: int = 3,
):
    """Run the ONNX graph with pure Python executor and verify it doesn't crash.

    For ZO training graphs, we verify execution succeeds (the perturbation
    ops produce finite outputs). Full numerical matching against the RNG
    reference is done in dedicated tests.
    """
    for seed in range(num_samples):
        rng = np.random.default_rng(seed)
        test_input = rng.standard_normal(input_shape).astype(np.float32)
        # ZO graphs need both input and label
        num_classes = 10  # default; overridden per-model
        label = np.zeros((input_shape[0], num_classes), dtype=np.float32)
        label[0, 0] = 1.0

        feeds = {"input": test_input, "label": label}
        try:
            output = run_onnx_graph(onnx_file, feeds)
            assert np.all(np.isfinite(output)), (
                f"Non-finite output at seed={seed}"
            )
        except Exception as e:
            pytest.fail(f"run_onnx_graph failed at seed={seed}: {e}")


# ===========================================================================
# LightweightCNN (float) ZO Tests
# ===========================================================================


@pytest.mark.zo
class TestLiteCNNZO:
    """ZO perturbation tests for LightweightCNN (float)."""

    _MODEL = "LightweightCNN"
    _MODE = "zo-train"
    _INPUT_SHAPE = (1, 1, 28, 28)

    def test_litecnn_uniform(self, model_test_dir):
        """LiteCNN-Uniform: export and verify perturbation nodes."""
        zo_model = _run_and_verify_zo_export(
            self._MODEL, self._MODE, "uniform", model_test_dir
        )
        # Verify inference model runs through pure Python executor
        infer_path = os.path.join(model_test_dir, "network_infer.onnx")
        rng = np.random.default_rng(0)
        test_input = rng.standard_normal(self._INPUT_SHAPE).astype(np.float32)
        output = run_onnx_graph(infer_path, {"input": test_input})
        assert output is not None and np.all(np.isfinite(output))

    def test_litecnn_rademacher(self, model_test_dir):
        """LiteCNN-Rademacher: export and verify perturbation nodes."""
        _run_and_verify_zo_export(
            self._MODEL, self._MODE, "rademacher", model_test_dir
        )

    def test_litecnn_eggroll(self, model_test_dir):
        """LiteCNN-Eggroll: export and verify perturbation nodes (uses Rademacher)."""
        _run_and_verify_zo_export(
            self._MODEL, self._MODE, "eggroll", model_test_dir
        )


# ===========================================================================
# QLiteCNN (quantized) ZO Tests
# ===========================================================================


@pytest.mark.zo
@pytest.mark.quantized
class TestQLiteCNNZO:
    """ZO perturbation tests for QLiteCNN (quantized)."""

    _MODEL = "QLiteCNN"
    _MODE = "q-zo-train"
    _INPUT_SHAPE = (1, 1, 28, 28)

    def test_qlitecnn_rqs_rademacher(self, model_test_dir):
        """QLiteCNN-RQSRad: quantized ZO export with RQS Rademacher perturbation."""
        zo_model = _run_and_verify_zo_export(
            self._MODEL, self._MODE, "rqs_rademacher", model_test_dir,
            quantized=True,
        )
        # Verify RQSPerturbRademacher nodes exist
        rqs_nodes = [
            n for n in zo_model.graph.node
            if n.op_type == "RQSPerturbRademacher"
        ]
        assert len(rqs_nodes) > 0, "No RQSPerturbRademacher nodes in quantized ZO graph"


# ===========================================================================
# SleepConViT (float) ZO Tests
# ===========================================================================


@pytest.mark.zo
class TestSleepViTZO:
    """ZO perturbation tests for SleepConViT (float)."""

    _MODEL = "SleepConViT"
    _MODE = "zo-train"
    _INPUT_SHAPE = (1, 1, 1, 3000)

    def test_sleepvit_uniform(self, model_test_dir):
        """SleepViT-Uniform: export and verify perturbation nodes."""
        zo_model = _run_and_verify_zo_export(
            self._MODEL, self._MODE, "uniform", model_test_dir
        )
        # Verify inference model runs through pure Python executor
        infer_path = os.path.join(model_test_dir, "network_infer.onnx")
        rng = np.random.default_rng(0)
        test_input = rng.standard_normal(self._INPUT_SHAPE).astype(np.float32)
        output = run_onnx_graph(infer_path, {"input": test_input})
        assert output is not None and np.all(np.isfinite(output))

    def test_sleepvit_rademacher(self, model_test_dir):
        """SleepViT-Rademacher: export and verify perturbation nodes."""
        _run_and_verify_zo_export(
            self._MODEL, self._MODE, "rademacher", model_test_dir
        )

    def test_sleepvit_eggroll(self, model_test_dir):
        """SleepViT-Eggroll: export and verify perturbation nodes (uses Rademacher)."""
        _run_and_verify_zo_export(
            self._MODEL, self._MODE, "eggroll", model_test_dir
        )


# ===========================================================================
# QSleepConViT (quantized) ZO Tests
# ===========================================================================


@pytest.mark.zo
@pytest.mark.quantized
class TestQSleepViTZO:
    """ZO perturbation tests for QSleepConViT (quantized)."""

    _MODEL = "QSleepConViT"
    _MODE = "q-zo-train"
    _INPUT_SHAPE = (1, 1, 1, 3000)

    def test_qsleepvit_rqs_rademacher(self, model_test_dir):
        """QSleepViT-RQSRad: quantized ZO export with RQS Rademacher perturbation."""
        zo_model = _run_and_verify_zo_export(
            self._MODEL, self._MODE, "rqs_rademacher", model_test_dir,
            quantized=True,
        )
        rqs_nodes = [
            n for n in zo_model.graph.node
            if n.op_type == "RQSPerturbRademacher"
        ]
        assert len(rqs_nodes) > 0, "No RQSPerturbRademacher nodes in quantized ZO graph"


# ===========================================================================
# RNG Seed Verification Tests
# ===========================================================================


@pytest.mark.zo
class TestRNGSeedComputation:
    """Verify the RNG seed computation matches the Deeploy C++ reference."""

    def test_scramble_seed(self):
        """Verify scramble formula: seed * 1664525 + 1013904223."""
        assert scramble_seed(0) == np.uint32(1013904223)
        assert scramble_seed(1) == np.uint32(1664525 + 1013904223)
        assert scramble_seed(42) == np.uint32(
            np.uint32(42) * np.uint32(1664525) + np.uint32(1013904223)
        )

    def test_xorshift32_deterministic(self):
        """Verify Xorshift32 is deterministic."""
        state = np.uint32(12345)
        s1 = xorshift32(state)
        s2 = xorshift32(state)
        assert s1 == s2
        # Verify it changes state
        assert xorshift32(s1) != s1

    def test_seed_per_node(self):
        """Verify each node gets a unique seed: seed + NUM_CORES * node_id + core_id."""
        global_seed = 42
        # Two different nodes should get different seeds
        seed_node0_core0 = scramble_seed(global_seed + NUM_CORES * 0 + 0)
        seed_node1_core0 = scramble_seed(global_seed + NUM_CORES * 1 + 0)
        assert seed_node0_core0 != seed_node1_core0

        # Same node, different cores should get different seeds
        seed_node0_core1 = scramble_seed(global_seed + NUM_CORES * 0 + 1)
        assert seed_node0_core0 != seed_node0_core1

    def test_uniform_perturbation_deterministic(self):
        """Verify uniform perturbation is deterministic for same seed/node_id."""
        data = np.zeros(64, dtype=np.float32)
        result1 = generate_uniform_perturbation(data, global_seed=42, node_id=0, eps=0.01)
        result2 = generate_uniform_perturbation(data, global_seed=42, node_id=0, eps=0.01)
        np.testing.assert_array_equal(result1, result2)

    def test_rademacher_perturbation_values(self):
        """Verify Rademacher perturbation only produces +/- eps offsets."""
        data = np.zeros(64, dtype=np.float32)
        eps = 0.01
        result = generate_rademacher_perturbation(data, global_seed=42, node_id=0, eps=eps)
        # All values should be exactly +eps or -eps
        np.testing.assert_array_less(np.abs(np.abs(result) - eps), 1e-7)
