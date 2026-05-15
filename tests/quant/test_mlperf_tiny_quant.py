# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Smoke tests for `-mode quant` on the MLperf Tiny benchmark suite.

Each test instantiates the registered exporter, swaps ``create_model`` for the
Brevitas-quantized factory, runs the ``DeepQuant.exportBrevitas`` → 12-pass
adapter pipeline, and asserts the resulting ONNX is structurally
Deeploy-compatible (only Conv/Gemm/Add/ReduceMean/Flatten/RequantShift ops;
int8 input/output dtype; no Quant/Dequant nodes left in the graph).

Skip-conditions:
- ``brevitas`` not installed → skip
- ``DeepQuant`` not importable → skip
"""

from collections import Counter

import pytest

# Hard skip for the entire module if brevitas or DeepQuant aren't available.
brevitas = pytest.importorskip("brevitas")
DeepQuant = pytest.importorskip("DeepQuant.ExportBrevitas")


import onnx  # noqa: E402

# These op types are the only ones expected in a Deeploy-compatible quantized
# graph after the 12-pass adapter pipeline (see
# ``onnx4deeploy.optimization.qcdq_to_deeploy``). Anything else — in particular
# leftover ``QuantizeLinear`` / ``DequantizeLinear`` — indicates a regression.
_ALLOWED_OPS = {
    "Conv",
    "Gemm",
    "MatMul",
    "Add",
    "ReduceMean",
    "Flatten",
    "Reshape",
    "Transpose",
    "Squeeze",
    "Unsqueeze",
    "RequantShift",
}

_DTYPE_INT8 = 3  # onnx TensorProto.INT8

# (registry_name, expected_min_node_count) — the lower bound guards against
# accidental over-folding to an empty graph.
_MLPERF_TINY_QUANT_MODELS = [
    ("ResNet8", 20),
    ("MobileNetV2-VWW", 50),
    ("DSCNN", 20),
    ("DSCNN-S", 20),
    ("Autoencoder", 10),
    ("Autoencoder-MLPerf", 10),
]


@pytest.fixture(scope="module")
def model_registry():
    """Pull the CLI's model registry dict (defined inside ``list_available_models``)."""
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Onnx4Deeploy import list_available_models  # noqa: PLC0415

    return list_available_models()


@pytest.mark.parametrize("model_name,min_nodes", _MLPERF_TINY_QUANT_MODELS)
def test_quant_pipeline_is_deeploy_compatible(tmp_path, model_registry, model_name, min_nodes):
    """End-to-end smoke: -mode quant produces a Deeploy-shaped int8 ONNX."""
    entry = model_registry[model_name]
    exporter_cls = entry["class"]

    out_dir = tmp_path / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    exporter = exporter_cls(save_path=str(out_dir))
    exporter._config_overrides = entry.get("config", {})
    exporter.config = exporter.load_config()

    onnx_path = exporter.export_quantized()
    model = onnx.load(str(onnx_path))

    op_counter = Counter(n.op_type for n in model.graph.node)

    unknown_ops = set(op_counter) - _ALLOWED_OPS
    assert not unknown_ops, (
        f"{model_name}: unexpected op types remain after adapter pipeline: "
        f"{sorted(unknown_ops)} (full histogram: {dict(op_counter)})"
    )

    assert sum(op_counter.values()) >= min_nodes, (
        f"{model_name}: only {sum(op_counter.values())} nodes after adapter "
        f"(expected ≥ {min_nodes}); pipeline likely over-folded"
    )

    inp_dtype = model.graph.input[0].type.tensor_type.elem_type
    out_dtype = model.graph.output[0].type.tensor_type.elem_type
    assert (
        inp_dtype == _DTYPE_INT8
    ), f"{model_name}: input dtype is {inp_dtype}, expected INT8 ({_DTYPE_INT8})"
    assert (
        out_dtype == _DTYPE_INT8
    ), f"{model_name}: output dtype is {out_dtype}, expected INT8 ({_DTYPE_INT8})"
