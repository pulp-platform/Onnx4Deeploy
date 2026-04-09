# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Tests for DS-CNN model export (MLperf Tiny Keyword Spotting benchmark).

Tests inference and training graph generation for DS-CNN-XS (25×10 MFCC input).
"""

import os

import numpy as np
import pytest

from onnx4deeploy.models.dscnn_exporter import DSCNNExporter

from .test_utils import (
    create_random_input,
    verify_inference_export,
    verify_onnxruntime_compatibility,
    verify_trainable_params,
    verify_training_export,
)


@pytest.mark.inference
class TestDSCNNInference:
    """Test DS-CNN model inference mode export."""

    def test_dscnn_inference_export(self, model_test_dir, dscnn_config):
        """Test DS-CNN model inference export with full verification."""
        exporter = DSCNNExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in dscnn_config.items()}

        # DS-CNN input: (batch, 1, n_time, n_freq)
        expected_input_shape = [
            dscnn_config["batch_size"],
            1,
            dscnn_config["n_time"],
            dscnn_config["n_freq"],
        ]

        verify_inference_export(
            exporter,
            model_test_dir,
            expected_input_shape=expected_input_shape,
            expected_batch_size=dscnn_config["batch_size"],
            expected_output_classes=dscnn_config["num_classes"],
        )

    def test_dscnn_onnxruntime_inference(self, model_test_dir, dscnn_config):
        """Test exported DS-CNN model runs correctly with ONNX Runtime."""
        exporter = DSCNNExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in dscnn_config.items()}
        onnx_file = exporter.export(mode="infer")

        input_shape = (
            dscnn_config["batch_size"],
            1,
            dscnn_config["n_time"],
            dscnn_config["n_freq"],
        )
        test_input = create_random_input(input_shape)
        expected_output_shape = (dscnn_config["batch_size"], dscnn_config["num_classes"])
        verify_onnxruntime_compatibility(
            onnx_file, test_input, expected_output_shape, input_name="input"
        )

    def test_dscnn_trainable_params(self, model_test_dir, dscnn_config):
        """Test DS-CNN trainable parameter identification."""
        exporter = DSCNNExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in dscnn_config.items()}
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        verify_trainable_params(exporter, model)

    def test_dscnn_last_layer_strategy(self, model_test_dir, dscnn_config):
        """Test DS-CNN last-layer-only training strategy."""
        exporter = DSCNNExporter(save_path=model_test_dir)
        cfg = {k: v for k, v in dscnn_config.items()}
        cfg["training_strategy"] = "last_layer"
        exporter._config_overrides = cfg
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        all_params = [name for name, _ in model.named_parameters()]
        trainable = exporter.get_trainable_params(all_params)

        assert len(trainable) > 0
        assert len(trainable) < len(all_params)


@pytest.mark.training
class TestDSCNNTraining:
    """Test DS-CNN model training mode export."""

    def test_dscnn_training_export(self, model_test_dir, dscnn_config):
        """Test DS-CNN training graph generation (smoke test with random data)."""
        exporter = DSCNNExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            **dscnn_config,
            "n_batches": 2,
            "n_accum": 1,
            "dataset": "random",
        }
        onnx_file = verify_training_export(exporter, model_test_dir)
        assert os.path.exists(onnx_file)

    def test_dscnn_training_artifacts(self, model_test_dir, dscnn_config):
        """Test DS-CNN training artifacts are generated correctly."""
        exporter = DSCNNExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            **dscnn_config,
            "n_batches": 2,
            "n_accum": 1,
            "dataset": "random",
        }
        verify_training_export(
            exporter,
            model_test_dir,
            required_artifacts=["checkpoint", "optimizer_model.onnx", "eval_model.onnx"],
        )

    def test_dscnn_training_npz_layout(self, model_test_dir, dscnn_config):
        """Test inputs.npz / outputs.npz are generated with the correct layout."""
        n_batches = 4
        exporter = DSCNNExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            **dscnn_config,
            "n_batches": n_batches,
            "n_accum": 1,
            "dataset": "random",
        }
        onnx_file = verify_training_export(exporter, model_test_dir)
        output_dir = os.path.dirname(onnx_file)

        npz_in = np.load(os.path.join(output_dir, "inputs.npz"), allow_pickle=True)
        assert "meta_data_size" in npz_in
        assert "meta_n_batches" in npz_in
        assert int(npz_in["meta_n_batches"]) == n_batches

        npz_out = np.load(os.path.join(output_dir, "outputs.npz"), allow_pickle=True)
        assert "loss" in npz_out
        assert len(npz_out["loss"]) == n_batches
