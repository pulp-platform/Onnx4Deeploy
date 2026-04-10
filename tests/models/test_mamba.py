# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Tests for Mamba model export (SelectiveSSM-based).

Currently only inference mode is tested. The Mamba exporter does not yet
implement the `training_mode` argument required for training graph export.
"""

import onnx
import pytest

from onnx4deeploy.models.mamba_exporter import MambaExporter

from .test_utils import load_and_check_onnx_model, verify_trainable_params, verifyonnx_file_exists


@pytest.mark.inference
class TestMambaInference:
    """Test Mamba model inference mode export."""

    def test_mamba_inference_export(self, model_test_dir, mamba_config):
        """Test Mamba inference export produces a valid ONNX file."""
        exporter = MambaExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in mamba_config.items()}

        onnx_file = exporter.export(mode="infer")
        verifyonnx_file_exists(onnx_file)

        # Mamba uses custom SelectiveSSM operators so skip strict ONNX shape
        # inference; we only verify basic structural integrity.
        model = load_and_check_onnx_model(onnx_file, skip_shape_check=True)
        assert len(model.graph.input) >= 1
        assert len(model.graph.output) >= 1

    def test_mamba_inference_input_shape(self, model_test_dir, mamba_config):
        """Test Mamba export produces the expected input tensor shape."""
        exporter = MambaExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in mamba_config.items()}

        onnx_file = exporter.export(mode="infer")
        model = onnx.load(onnx_file)

        input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
        expected = [
            mamba_config["batch_size"],
            mamba_config["max_seq_len"],
            mamba_config["d_model"],
        ]
        assert (
            input_shape == expected
        ), f"Input shape mismatch: expected {expected}, got {input_shape}"

    def test_mamba_custom_selective_ssm_op(self, model_test_dir, mamba_config):
        """Test Mamba export uses the custom SelectiveSSM operator."""
        exporter = MambaExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in mamba_config.items()}

        onnx_file = exporter.export(mode="infer")
        model = onnx.load(onnx_file)

        op_types = {node.op_type for node in model.graph.node}
        assert (
            "SelectiveSSM" in op_types
        ), f"Expected SelectiveSSM op in Mamba graph, got ops: {sorted(op_types)}"

    def test_mamba_trainable_params(self, model_test_dir, mamba_config):
        """Test Mamba trainable parameters identification."""
        exporter = MambaExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in mamba_config.items()}
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        verify_trainable_params(exporter, model)
