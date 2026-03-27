# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Unit tests for individual passes in onnx4deeploy/transform/quant_transform.py.

Strategy
--------
Each test builds a *minimal* ONNX graph that exercises exactly one pass,
runs the *original* graph through ORT as the numerical reference, applies
the pass, then runs the *transformed* graph through ``run_onnx_graph``
(the pure-Python executor in onnx_node_implementations.py) and asserts
the outputs are identical (or within rounding for RequantShift).

Passes under test
-----------------
1. ``float_to_rqs_params``           pure arithmetic, no ONNX graph needed.
2. ``replace_qdq_with_deeploy``      QuantizeLinear → Quant
3. ``replace_qdq_with_deeploy``      DequantizeLinear → Dequant
4. ``replace_qdq_with_deeploy``      QDQ pair end-to-end
5. ``insert_rqs_from_map``           RequantShift insertion
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onph
import onnx_graphsurgeon as gs
import onnxruntime as ort
import pytest

from onnx4deeploy.transform.quant_transform import (
    float_to_rqs_params,
    insert_rqs_from_map,
    replace_qdq_with_deeploy,
)
from .onnx_node_implementations import run_onnx_graph, exec_requant_shift


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_tmp(model: onnx.ModelProto) -> str:
    """Save an ONNX model to a temporary file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.close()
    onnx.save(model, tmp.name)
    return tmp.name


def _ort_run(model_path: str, feed: dict) -> np.ndarray:
    """Run a model with ORT and return the first output."""
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return sess.run(None, feed)[0]


def _gs_to_tmp(graph: gs.Graph) -> str:
    """Export a graphsurgeon graph to a temp ONNX file."""
    graph.cleanup().toposort()
    return _save_tmp(gs.export_onnx(graph))


# ---------------------------------------------------------------------------
# 1. float_to_rqs_params – pure arithmetic
# ---------------------------------------------------------------------------


class TestFloatToRqsParams:
    """Verify that the integer RQS params reproduce the scale ratio."""

    def _check_ratio(self, ratio: np.ndarray) -> None:
        """
        For integer inputs spanning [-128, 127], verify that the RequantShift
        formula ``clip(((x * mul) + add) >> shift, -128, 127)`` gives the same
        result as the reference ``clip(round(x * ratio), -128, 127)``.
        """
        mul, add, div = float_to_rqs_params(ratio)
        shift = int(round(np.log2(div))) if div > 1 else 0

        x = np.arange(-128, 128, dtype=np.int64)

        # Reference: exact float multiplication, rounded and clipped
        ref = np.clip(np.round(x * ratio.flatten()[0]), -128, 127).astype(np.int64)

        # RequantShift formula — use int64 to avoid overflow (mul can be up to 2^30)
        mul64 = np.asarray(mul, dtype=np.int64).flatten()[0]
        rqs = np.clip(((x * mul64) + np.int64(add)) >> shift, -128, 127).astype(np.int64)

        # Allow ±1 LSB rounding difference
        assert np.all(np.abs(rqs - ref) <= 1), (
            f"RQS params mismatch for ratio={ratio}: "
            f"max diff = {np.max(np.abs(rqs - ref))}"
        )

    def test_scalar_ratio_halving(self):
        """scale_src / scale_dst = 0.5 → requantisation halves the value."""
        self._check_ratio(np.array(0.5))

    def test_scalar_ratio_doubling(self):
        """scale_src / scale_dst = 2.0."""
        self._check_ratio(np.array(2.0))

    def test_scalar_ratio_identity(self):
        """scale_src == scale_dst → identity (ratio = 1.0)."""
        self._check_ratio(np.array(1.0))

    def test_scalar_ratio_arbitrary(self):
        """Non-power-of-two ratio."""
        self._check_ratio(np.array(0.75))

    def test_per_channel_vector(self):
        """Per-channel scale vector: check that mul is a vector."""
        ratio = np.array([0.5, 1.0, 2.0, 0.25], dtype=np.float64)
        mul, add, div = float_to_rqs_params(ratio)
        assert mul.shape == ratio.shape, "mul must be per-channel"

    def test_div_is_power_of_two(self):
        """div must always be a power of two (log2(div) is integer)."""
        for r in [0.1, 0.333, 1.5, 3.7, 0.001]:
            _, _, div = float_to_rqs_params(np.array(r))
            log2_div = np.log2(div)
            assert abs(log2_div - round(log2_div)) < 1e-9, (
                f"div={div} is not a power of two for ratio={r}"
            )

    def test_mul_fits_int32(self):
        """mul must fit in int32 for all tested ratios."""
        for r in [0.001, 0.5, 1.0, 100.0, 1234.56]:
            mul, _, _ = float_to_rqs_params(np.array(r))
            assert np.all(np.abs(mul) <= 2**31 - 1)


# ---------------------------------------------------------------------------
# 2. replace_qdq_with_deeploy – QuantizeLinear → Quant
# ---------------------------------------------------------------------------


class TestReplaceQuantizeLinear:
    """
    QuantizeLinear → Deeploy Quant.

    Reference: ORT runs the original QuantizeLinear node.
    Transformed: run_onnx_graph runs the Quant node.
    Expected: identical int8 output.
    """

    def _make_quantize_graph(self, scale: float, zero_point: int = 0) -> str:
        """Build Input(float32) → QuantizeLinear(scale, zp) → Output(int8)."""
        scale_t = oh.make_tensor("scale", onnx.TensorProto.FLOAT, [], [scale])
        zp_t = oh.make_tensor("zp", onnx.TensorProto.INT8, [], [zero_point])

        node = oh.make_node(
            "QuantizeLinear",
            inputs=["input", "scale", "zp"],
            outputs=["output"],
        )
        graph = oh.make_graph(
            [node],
            "quant_graph",
            [oh.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 8])],
            [oh.make_tensor_value_info("output", onnx.TensorProto.INT8, [1, 8])],
            initializer=[scale_t, zp_t],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
        return _save_tmp(model)

    def test_quantize_per_tensor_zero_zp(self):
        """Per-tensor quantization with zero_point=0."""
        original_path = self._make_quantize_graph(scale=0.5, zero_point=0)
        x = np.array([[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -2.0]], dtype=np.float32)

        reference = _ort_run(original_path, {"input": x})

        graph = gs.import_onnx(onnx.load(original_path))
        replace_qdq_with_deeploy(graph)
        transformed_path = _gs_to_tmp(graph)

        result = run_onnx_graph(transformed_path, {"input": x})

        np.testing.assert_array_equal(
            result.astype(np.int8), reference,
            err_msg="Quant output differs from ORT QuantizeLinear (zero_point=0)",
        )

    def test_quantize_per_tensor_nonzero_zp(self):
        """Per-tensor quantization with non-zero zero_point."""
        original_path = self._make_quantize_graph(scale=0.25, zero_point=10)
        x = np.array([[-1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 0.25, 0.75]], dtype=np.float32)

        reference = _ort_run(original_path, {"input": x})

        graph = gs.import_onnx(onnx.load(original_path))
        replace_qdq_with_deeploy(graph)
        transformed_path = _gs_to_tmp(graph)

        result = run_onnx_graph(transformed_path, {"input": x})

        np.testing.assert_array_equal(
            result.astype(np.int8), reference,
            err_msg="Quant output differs from ORT QuantizeLinear (nonzero zp)",
        )

    def test_quantize_saturation(self):
        """Values outside the quantization range must saturate to ±127."""
        original_path = self._make_quantize_graph(scale=0.1, zero_point=0)
        x = np.array([[-200.0, -100.0, 0.0, 100.0, 200.0, 12.7, -12.8, 0.05]],
                     dtype=np.float32)

        reference = _ort_run(original_path, {"input": x})

        graph = gs.import_onnx(onnx.load(original_path))
        replace_qdq_with_deeploy(graph)
        transformed_path = _gs_to_tmp(graph)

        result = run_onnx_graph(transformed_path, {"input": x})

        np.testing.assert_array_equal(
            result.astype(np.int8), reference,
            err_msg="Quant saturation differs from ORT",
        )


# ---------------------------------------------------------------------------
# 3. replace_qdq_with_deeploy – DequantizeLinear → Dequant
# ---------------------------------------------------------------------------


class TestReplaceDequantizeLinear:
    """
    DequantizeLinear → Deeploy Dequant.

    Reference: ORT runs the original DequantizeLinear node.
    Transformed: run_onnx_graph runs the Dequant node.
    Expected: identical float32 output.
    """

    def _make_dequantize_graph(self, scale: float, zero_point: int = 0) -> str:
        """Build Input(int8) → DequantizeLinear(scale, zp) → Output(float32)."""
        scale_t = oh.make_tensor("scale", onnx.TensorProto.FLOAT, [], [scale])
        zp_t = oh.make_tensor("zp", onnx.TensorProto.INT8, [], [zero_point])

        node = oh.make_node(
            "DequantizeLinear",
            inputs=["input", "scale", "zp"],
            outputs=["output"],
        )
        graph = oh.make_graph(
            [node],
            "dequant_graph",
            [oh.make_tensor_value_info("input", onnx.TensorProto.INT8, [1, 8])],
            [oh.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 8])],
            initializer=[scale_t, zp_t],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
        return _save_tmp(model)

    def test_dequantize_zero_zp(self):
        """Per-tensor dequantization with zero_point=0."""
        original_path = self._make_dequantize_graph(scale=0.5, zero_point=0)
        x = np.array([[-128, -64, -1, 0, 1, 64, 127, -100]], dtype=np.int8)

        reference = _ort_run(original_path, {"input": x})

        graph = gs.import_onnx(onnx.load(original_path))
        replace_qdq_with_deeploy(graph)
        transformed_path = _gs_to_tmp(graph)

        result = run_onnx_graph(transformed_path, {"input": x})

        np.testing.assert_allclose(
            result, reference, rtol=0, atol=1e-6,
            err_msg="Dequant output differs from ORT DequantizeLinear (zero_point=0)",
        )

    def test_dequantize_nonzero_zp(self):
        """Per-tensor dequantization with non-zero zero_point."""
        original_path = self._make_dequantize_graph(scale=0.25, zero_point=10)
        x = np.array([[-128, -64, -10, 0, 10, 64, 127, -1]], dtype=np.int8)

        reference = _ort_run(original_path, {"input": x})

        graph = gs.import_onnx(onnx.load(original_path))
        replace_qdq_with_deeploy(graph)
        transformed_path = _gs_to_tmp(graph)

        result = run_onnx_graph(transformed_path, {"input": x})

        np.testing.assert_allclose(
            result, reference, rtol=0, atol=1e-6,
            err_msg="Dequant output differs from ORT DequantizeLinear (nonzero zp)",
        )


# ---------------------------------------------------------------------------
# 4. replace_qdq_with_deeploy – full QDQ pair end-to-end
# ---------------------------------------------------------------------------


class TestReplaceQDQPair:
    """
    Input(float32) → QuantizeLinear → DequantizeLinear → Output(float32).

    Reference: ORT runs the original QDQ graph.
    Transformed: run_onnx_graph runs the Quant → Dequant graph.
    Expected: identical float32 output (same rounding as ORT).
    """

    def _make_qdq_graph(self, scale: float, zero_point: int = 0) -> str:
        scale_t = oh.make_tensor("scale", onnx.TensorProto.FLOAT, [], [scale])
        zp_t = oh.make_tensor("zp", onnx.TensorProto.INT8, [], [zero_point])

        q_node = oh.make_node(
            "QuantizeLinear",
            inputs=["input", "scale", "zp"],
            outputs=["quantized"],
        )
        dq_node = oh.make_node(
            "DequantizeLinear",
            inputs=["quantized", "scale", "zp"],
            outputs=["output"],
        )
        graph = oh.make_graph(
            [q_node, dq_node],
            "qdq_graph",
            [oh.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 8])],
            [oh.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 8])],
            initializer=[scale_t, zp_t],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
        return _save_tmp(model)

    def test_qdq_roundtrip(self):
        """QDQ roundtrip: float → quantize → dequantize → float."""
        original_path = self._make_qdq_graph(scale=0.5, zero_point=0)
        x = np.array([[-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]], dtype=np.float32)

        reference = _ort_run(original_path, {"input": x})

        graph = gs.import_onnx(onnx.load(original_path))
        replace_qdq_with_deeploy(graph)
        transformed_path = _gs_to_tmp(graph)

        result = run_onnx_graph(transformed_path, {"input": x})

        np.testing.assert_allclose(
            result, reference, rtol=0, atol=1e-6,
            err_msg="QDQ roundtrip differs from ORT",
        )

    def test_qdq_roundtrip_small_scale(self):
        """QDQ with a small scale (high precision)."""
        original_path = self._make_qdq_graph(scale=0.00390625, zero_point=0)
        x = np.random.default_rng(0).uniform(-0.5, 0.5, (1, 8)).astype(np.float32)

        reference = _ort_run(original_path, {"input": x})

        graph = gs.import_onnx(onnx.load(original_path))
        replace_qdq_with_deeploy(graph)
        transformed_path = _gs_to_tmp(graph)

        result = run_onnx_graph(transformed_path, {"input": x})

        np.testing.assert_allclose(
            result, reference, rtol=0, atol=1e-6,
            err_msg="QDQ roundtrip (small scale) differs from ORT",
        )


# ---------------------------------------------------------------------------
# 5. insert_rqs_from_map – RequantShift insertion
# ---------------------------------------------------------------------------


class TestInsertRqs:
    """
    RequantShift: rescale int8 tensor from scale s_src to scale s_dst.

    Reference: ORT runs QuantizeLinear(s_src) → DequantizeLinear(s_src) →
               QuantizeLinear(s_dst) → DequantizeLinear(s_dst) to get the
               exact floating-point result after double-quantisation.
    Transformed: run_onnx_graph runs a graph where RQS replaces the second
               QuantizeLinear, followed by Dequant(s_dst).
    Tolerance: ±1 LSB at scale s_dst (RequantShift introduces ±1 rounding vs ORT).
    """

    def _make_double_qdq_graph(
        self, scale_src: float, scale_dst: float
    ) -> tuple:
        """
        Build:  float → QDQ(s_src) → QDQ(s_dst) → float

        Returns (ort_path, src_tensor_name, dst_tensor_name, gs_graph)
        so that insert_rqs_from_map can be applied to the gs_graph.
        """
        s_src_t = oh.make_tensor("s_src", onnx.TensorProto.FLOAT, [], [scale_src])
        zp_src_t = oh.make_tensor("zp_src", onnx.TensorProto.INT8, [], [0])
        s_dst_t = oh.make_tensor("s_dst", onnx.TensorProto.FLOAT, [], [scale_dst])
        zp_dst_t = oh.make_tensor("zp_dst", onnx.TensorProto.INT8, [], [0])

        nodes = [
            oh.make_node("QuantizeLinear", ["input", "s_src", "zp_src"], ["q_src"]),
            oh.make_node("DequantizeLinear", ["q_src", "s_src", "zp_src"], ["dq_src"]),
            oh.make_node("QuantizeLinear", ["dq_src", "s_dst", "zp_dst"], ["q_dst"]),
            oh.make_node("DequantizeLinear", ["q_dst", "s_dst", "zp_dst"], ["output"]),
        ]
        graph = oh.make_graph(
            nodes, "double_qdq",
            [oh.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 8])],
            [oh.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 8])],
            initializer=[s_src_t, zp_src_t, s_dst_t, zp_dst_t],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
        ort_path = _save_tmp(model)
        return ort_path

    def _make_rqs_target_graph(
        self, scale_src: float, scale_dst: float
    ) -> str:
        """
        Build the graph that insert_rqs_from_map will operate on:
          float → Quant(s_src) → <gap where RQS goes> → Dequant(s_dst) → float

        We represent the "gap" as two tensors q_src and q_dst of the same dtype,
        connected via an Identity (which will be replaced by RQS after the pass).
        """
        s_src_t = oh.make_tensor("s_src", onnx.TensorProto.FLOAT, [], [scale_src])
        zp_src_t = oh.make_tensor("zp_src", onnx.TensorProto.INT8, [], [0])
        s_dst_t = oh.make_tensor("s_dst", onnx.TensorProto.FLOAT, [], [scale_dst])
        zp_dst_t = oh.make_tensor("zp_dst", onnx.TensorProto.INT8, [], [0])

        nodes = [
            oh.make_node("QuantizeLinear", ["input", "s_src", "zp_src"], ["q_src"]),
            oh.make_node("Identity", ["q_src"], ["q_dst"]),
            oh.make_node("DequantizeLinear", ["q_dst", "s_dst", "zp_dst"], ["output"]),
        ]
        graph_proto = oh.make_graph(
            nodes, "rqs_target",
            [oh.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 8])],
            [oh.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 8])],
            initializer=[s_src_t, zp_src_t, s_dst_t, zp_dst_t],
        )
        model = oh.make_model(graph_proto, opset_imports=[oh.make_opsetid("", 13)])
        return _save_tmp(model)

    def _run_rqs_test(self, scale_src: float, scale_dst: float, seed: int = 42):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1.0, 1.0, (1, 8)).astype(np.float32)

        # Reference: ORT double-QDQ (quantize at s_src, requantize to s_dst)
        ort_path = self._make_double_qdq_graph(scale_src, scale_dst)
        reference = _ort_run(ort_path, {"input": x})

        # Transformed: insert RequantShift between q_src and q_dst
        gs_model_path = self._make_rqs_target_graph(scale_src, scale_dst)
        graph = gs.import_onnx(onnx.load(gs_model_path))

        rqs_map = {
            "edges": [{
                "src_tensor": "q_src",
                "dst_tensor": "q_dst",
                "src_scale": scale_src,
                "dst_scale": scale_dst,
            }]
        }
        # First replace QDQ → Deeploy so the graph has Quant/Dequant nodes
        replace_qdq_with_deeploy(graph)
        insert_rqs_from_map(graph, rqs_map)
        transformed_path = _gs_to_tmp(graph)

        result = run_onnx_graph(transformed_path, {"input": x})

        # Allow ±1 LSB tolerance at scale_dst (RQS introduces ±1 rounding vs ORT)
        atol = scale_dst
        np.testing.assert_allclose(
            result, reference, rtol=0, atol=atol,
            err_msg=(
                f"RQS output differs from ORT double-QDQ "
                f"(s_src={scale_src}, s_dst={scale_dst}) "
                f"by more than 1 LSB.\n"
                f"  ORT:    {reference}\n"
                f"  RQS:    {result}\n"
                f"  diff:   {np.abs(result - reference)}"
            ),
        )

    def test_rqs_halving(self):
        """Requantise from s=0.5 to s=0.25 (factor 2 downscale)."""
        self._run_rqs_test(scale_src=0.5, scale_dst=0.25)

    def test_rqs_doubling(self):
        """Requantise from s=0.25 to s=0.5 (factor 2 upscale)."""
        self._run_rqs_test(scale_src=0.25, scale_dst=0.5)

    def test_rqs_identity(self):
        """Same scale: RequantShift should be a no-op (±1 LSB)."""
        self._run_rqs_test(scale_src=0.5, scale_dst=0.5)

    def test_rqs_arbitrary_ratio(self):
        """Non-power-of-two scale ratio."""
        self._run_rqs_test(scale_src=0.03125, scale_dst=0.06395246833562851)

    def test_rqs_params_roundtrip_via_exec(self):
        """
        Direct unit test: exec_requant_shift with params from float_to_rqs_params
        must match the reference integer requantisation.
        """
        scale_src, scale_dst = 0.5, 0.25
        ratio = np.array(scale_src / scale_dst)
        mul, add, div = float_to_rqs_params(ratio)

        x = np.arange(-128, 128, dtype=np.int8)
        result = exec_requant_shift(x, mul, np.array(add, np.int32), np.array(div, np.int32))

        reference = np.clip(np.round(x.astype(np.float64) * ratio), -128, 127).astype(np.int8)

        assert np.all(np.abs(result.astype(np.int32) - reference.astype(np.int32)) <= 1), (
            f"exec_requant_shift deviates from reference by more than 1 LSB: "
            f"max diff = {np.max(np.abs(result.astype(np.int32) - reference.astype(np.int32)))}"
        )
