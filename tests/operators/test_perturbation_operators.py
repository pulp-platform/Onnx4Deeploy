# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Tests for individual perturbation operators: PerturbUniform, PerturbRademacher,
and PerturbEggroll.

Each test class covers:
  - File generation (ONNX model, inputs.npz, outputs.npz) via the operator test
    generator classes in onnx4deeploy.operators.
  - Output shape correctness.
  - Output finiteness.
  - Determinism: the pure-Python executor produces the same result on two calls
    with the same inputs.
  - Reference-RNG consistency: the pure-Python executor result matches the
    reference _perturb_* helper functions in onnx4deeploy.utils directly.
"""

import os

import numpy as np
import pytest

from onnx4deeploy.operators import (
    PerturbEggrollOperatorTest,
    PerturbRademacherOperatorTest,
    PerturbUniformOperatorTest,
)
from onnx4deeploy.utils.onnx_node_implementations import (
    _perturb_rademacher,
    _perturb_uniform,
    run_onnx_graph,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_SEED = 42
_DEFAULT_EPS = 0.01
_SHAPES = [(1, 16), (2, 32), (1, 8, 4)]


def _write_uniform_config(path: str, shape) -> str:
    cfg_path = os.path.join(path, "config.yaml")
    with open(cfg_path, "w") as f:
        f.write(f"perturbuniform:\n  input_shape: {list(shape)}\n")
    return cfg_path


def _write_rademacher_config(path: str, shape) -> str:
    cfg_path = os.path.join(path, "config.yaml")
    with open(cfg_path, "w") as f:
        f.write(f"perturbrademacher:\n  input_shape: {list(shape)}\n")
    return cfg_path


def _write_eggroll_config(path: str, shape) -> str:
    cfg_path = os.path.join(path, "config.yaml")
    with open(cfg_path, "w") as f:
        f.write(f"perturbeggroll:\n  input_shape: {list(shape)}\n")
    return cfg_path


# ---------------------------------------------------------------------------
# PerturbUniform
# ---------------------------------------------------------------------------


class TestPerturbUniformOperator:
    """Tests for the PerturbUniform custom ONNX operator."""

    def test_files_generated(self, operator_test_dir):
        """Verify that generate() creates the ONNX model and data files."""
        cfg = _write_uniform_config(operator_test_dir, (1, 32))
        test = PerturbUniformOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        assert os.path.exists(onnx_file), "ONNX model file not created"
        assert os.path.exists(input_file), "inputs.npz not created"
        assert os.path.exists(output_file), "outputs.npz not created"

    def test_output_shape(self, operator_test_dir):
        """Output shape must match input shape."""
        shape = (2, 16)
        cfg = _write_uniform_config(operator_test_dir, shape)
        test = PerturbUniformOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        outputs = np.load(output_file)
        assert "perturbed_x" in outputs
        assert outputs["perturbed_x"].shape == shape

    def test_output_finite(self, operator_test_dir):
        """All output values must be finite."""
        cfg = _write_uniform_config(operator_test_dir, (1, 64))
        test = PerturbUniformOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        outputs = np.load(output_file)
        assert np.all(np.isfinite(outputs["perturbed_x"])), "PerturbUniform output contains non-finite values"

    @pytest.mark.parametrize("shape", _SHAPES)
    def test_pure_python_executor_runs(self, operator_test_dir, shape):
        """run_onnx_graph executes the PerturbUniform ONNX without errors."""
        cfg = _write_uniform_config(operator_test_dir, shape)
        test = PerturbUniformOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, _ = test.generate()

        x = np.load(input_file)["x"]
        result = run_onnx_graph(onnx_file, {"x": x})
        assert result is not None
        assert result.shape == x.shape

    def test_deterministic(self, operator_test_dir):
        """Two invocations of run_onnx_graph with the same input give the same result."""
        cfg = _write_uniform_config(operator_test_dir, (1, 32))
        test = PerturbUniformOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, _ = test.generate()

        x = np.load(input_file)["x"]
        out1 = run_onnx_graph(onnx_file, {"x": x})
        out2 = run_onnx_graph(onnx_file, {"x": x})
        np.testing.assert_array_equal(out1, out2, err_msg="PerturbUniform is not deterministic")

    def test_rng_reference_consistency(self, operator_test_dir):
        """run_onnx_graph output matches direct _perturb_uniform with seed=42, idx=0."""
        shape = (1, 32)
        cfg = _write_uniform_config(operator_test_dir, shape)
        test = PerturbUniformOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, _ = test.generate()

        x = np.load(input_file)["x"]
        graph_out = run_onnx_graph(onnx_file, {"x": x})

        # The ONNX node was created with seed=42, idx=0, eps=0.01*sqrt(3)
        eps = float(0.01 * np.sqrt(3))
        ref = _perturb_uniform(x, global_seed=42, node_id=0, eps=eps, sign=1)

        np.testing.assert_allclose(
            graph_out, ref, rtol=1e-6, atol=1e-6,
            err_msg="PerturbUniform graph result does not match reference RNG"
        )

    def test_perturbation_magnitude(self, operator_test_dir):
        """Perturbation magnitude is bounded by eps * sqrt(3) (uniform support [-sqrt(3), sqrt(3)])."""
        shape = (4, 64)
        cfg = _write_uniform_config(operator_test_dir, shape)
        test = PerturbUniformOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, _ = test.generate()

        x = np.load(input_file)["x"]
        out = run_onnx_graph(onnx_file, {"x": x})

        delta = np.abs(out - x)
        eps = float(0.01 * np.sqrt(3))
        assert np.all(delta <= eps * np.sqrt(3) + 1e-5), (
            f"PerturbUniform perturbation exceeds expected bound: max={delta.max():.6f}"
        )


# ---------------------------------------------------------------------------
# PerturbRademacher
# ---------------------------------------------------------------------------


class TestPerturbRademacherOperator:
    """Tests for the PerturbRademacher custom ONNX operator."""

    def test_files_generated(self, operator_test_dir):
        """Verify that generate() creates the ONNX model and data files."""
        cfg = _write_rademacher_config(operator_test_dir, (1, 32))
        test = PerturbRademacherOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)

    def test_output_shape(self, operator_test_dir):
        """Output shape must match input shape."""
        shape = (2, 16)
        cfg = _write_rademacher_config(operator_test_dir, shape)
        test = PerturbRademacherOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        outputs = np.load(output_file)
        assert "perturbed_x" in outputs
        assert outputs["perturbed_x"].shape == shape

    def test_output_finite(self, operator_test_dir):
        """All output values must be finite."""
        cfg = _write_rademacher_config(operator_test_dir, (1, 64))
        test = PerturbRademacherOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        outputs = np.load(output_file)
        assert np.all(np.isfinite(outputs["perturbed_x"]))

    @pytest.mark.parametrize("shape", _SHAPES)
    def test_pure_python_executor_runs(self, operator_test_dir, shape):
        """run_onnx_graph executes the PerturbRademacher ONNX without errors."""
        cfg = _write_rademacher_config(operator_test_dir, shape)
        test = PerturbRademacherOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, _ = test.generate()

        x = np.load(input_file)["x"]
        result = run_onnx_graph(onnx_file, {"x": x})
        assert result is not None
        assert result.shape == x.shape

    def test_deterministic(self, operator_test_dir):
        """Two invocations with same input produce identical results."""
        cfg = _write_rademacher_config(operator_test_dir, (1, 32))
        test = PerturbRademacherOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, _ = test.generate()

        x = np.load(input_file)["x"]
        out1 = run_onnx_graph(onnx_file, {"x": x})
        out2 = run_onnx_graph(onnx_file, {"x": x})
        np.testing.assert_array_equal(out1, out2)

    def test_rng_reference_consistency(self, operator_test_dir):
        """run_onnx_graph output matches direct _perturb_rademacher with seed=42, idx=0."""
        shape = (1, 32)
        cfg = _write_rademacher_config(operator_test_dir, shape)
        test = PerturbRademacherOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, _ = test.generate()

        x = np.load(input_file)["x"]
        graph_out = run_onnx_graph(onnx_file, {"x": x})

        # The ONNX node was created with seed=42, idx=0, eps=0.01
        ref = _perturb_rademacher(x, global_seed=42, node_id=0, eps=0.01, sign=1)

        np.testing.assert_allclose(
            graph_out, ref, rtol=1e-6, atol=1e-6,
            err_msg="PerturbRademacher graph result does not match reference RNG"
        )

    def test_perturbation_is_exactly_eps(self, operator_test_dir):
        """Every element of x should be perturbed by exactly ±eps=0.01."""
        shape = (4, 64)
        cfg = _write_rademacher_config(operator_test_dir, shape)
        test = PerturbRademacherOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, _ = test.generate()

        x = np.load(input_file)["x"]
        out = run_onnx_graph(onnx_file, {"x": x})

        delta = np.abs(out - x)
        np.testing.assert_allclose(
            delta, np.full_like(delta, 0.01), atol=1e-5,
            err_msg="PerturbRademacher: perturbation magnitude should be exactly eps=0.01"
        )

    def test_perturbation_values_binary(self, operator_test_dir):
        """Perturbation offsets must be exactly +eps or -eps (Rademacher property)."""
        cfg = _write_rademacher_config(operator_test_dir, (1, 128))
        test = PerturbRademacherOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, _ = test.generate()

        x = np.load(input_file)["x"]
        out = run_onnx_graph(onnx_file, {"x": x})

        noise = out - x
        # noise values should all be +0.01 or -0.01
        eps = np.float32(0.01)
        valid_mask = np.isclose(noise, eps, atol=1e-5) | np.isclose(noise, -eps, atol=1e-5)
        assert np.all(valid_mask), "PerturbRademacher noise contains values other than ±eps"


# ---------------------------------------------------------------------------
# PerturbEggroll
# ---------------------------------------------------------------------------


class TestPerturbEggrollOperator:
    """Tests for the PerturbEggroll custom ONNX operator."""

    def test_files_generated(self, operator_test_dir):
        """Verify that generate() creates the ONNX model and data files."""
        cfg = _write_eggroll_config(operator_test_dir, (4, 8))
        test = PerturbEggrollOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        assert os.path.exists(onnx_file)
        assert os.path.exists(input_file)
        assert os.path.exists(output_file)

    def test_output_shape(self, operator_test_dir):
        """Output shape must match input shape."""
        shape = (4, 8)
        cfg = _write_eggroll_config(operator_test_dir, shape)
        test = PerturbEggrollOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        outputs = np.load(output_file)
        assert "perturbed_x" in outputs
        assert outputs["perturbed_x"].shape == shape

    def test_output_finite(self, operator_test_dir):
        """All output values must be finite."""
        cfg = _write_eggroll_config(operator_test_dir, (4, 16))
        test = PerturbEggrollOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, output_file = test.generate()

        outputs = np.load(output_file)
        assert np.all(np.isfinite(outputs["perturbed_x"]))

    @pytest.mark.parametrize("shape", [(4, 8), (2, 16), (2, 4, 8)])
    def test_pure_python_executor_runs(self, operator_test_dir, shape):
        """run_onnx_graph executes the PerturbEggroll ONNX without errors."""
        cfg = _write_eggroll_config(operator_test_dir, shape)
        test = PerturbEggrollOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, _ = test.generate()

        x = np.load(input_file)["x"]
        result = run_onnx_graph(onnx_file, {"x": x})
        assert result is not None
        assert result.shape == x.shape

    def test_deterministic(self, operator_test_dir):
        """Two invocations with same input produce identical results."""
        cfg = _write_eggroll_config(operator_test_dir, (4, 8))
        test = PerturbEggrollOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, _ = test.generate()

        x = np.load(input_file)["x"]
        out1 = run_onnx_graph(onnx_file, {"x": x})
        out2 = run_onnx_graph(onnx_file, {"x": x})
        np.testing.assert_array_equal(out1, out2)

    def test_rng_reference_consistency_vectors(self, operator_test_dir):
        """
        The PerturbEggroll vectors (a and b) computed by run_onnx_graph match
        those from _perturb_rademacher applied to zero-filled column vectors.
        """
        shape = (4, 8)
        cfg = _write_eggroll_config(operator_test_dir, shape)
        test = PerturbEggrollOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, _ = test.generate()

        x = np.load(input_file)["x"]
        graph_out = run_onnx_graph(onnx_file, {"x": x})

        # The ONNX graph uses seed_a=13, idx=0 and seed_b=14, idx=1 for the two
        # PerturbEggroll nodes (as defined in perturbeggroll.py).
        a_shape = [shape[0], 1]
        b_shape = [int(np.prod(shape[1:])), 1]

        a_ref = _perturb_rademacher(
            np.zeros(a_shape, dtype=np.float32), global_seed=13, node_id=0, eps=1.0, sign=1
        )
        b_ref = _perturb_rademacher(
            np.zeros(b_shape, dtype=np.float32), global_seed=14, node_id=1, eps=1.0, sign=1
        )

        # PerturbEggroll output = eps * Gemm(a, b^T) with beta=0 (i.e. ignores x)
        # alpha in the graph is uniform_epsilon = 0.01 * sqrt(3)
        eps = float(0.01 * np.sqrt(3))
        expected = eps * (a_ref @ b_ref.T)
        expected = expected.reshape(shape)

        np.testing.assert_allclose(
            graph_out, expected, rtol=1e-5, atol=1e-5,
            err_msg="PerturbEggroll graph result does not match reference Rademacher vectors"
        )

    def test_low_rank_structure(self, operator_test_dir):
        """
        PerturbEggroll output = alpha * a @ b^T (Gemm with beta=0), which is
        exactly rank-1.  The output matrix itself should have matrix rank 1.
        """
        shape = (8, 16)
        cfg = _write_eggroll_config(operator_test_dir, shape)
        test = PerturbEggrollOperatorTest(config_path=cfg, save_path=operator_test_dir)
        onnx_file, input_file, _ = test.generate()

        x = np.load(input_file)["x"]
        out = run_onnx_graph(onnx_file, {"x": x})

        # The output IS alpha * a @ b^T (beta=0); it is rank-1.
        sv = np.linalg.svd(out.astype(np.float64), compute_uv=False)
        # Only one non-negligible singular value
        assert sv[0] > sv[1] * 1e3, (
            f"PerturbEggroll output does not appear rank-1: sv[0]={sv[0]:.4f}, sv[1]={sv[1]:.4f}"
        )
