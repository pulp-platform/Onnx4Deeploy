# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Tests for Lightweight CNN model export."""

import os

import numpy as np
import pytest

from onnx4deeploy.models.lightweight_cnn_exporter import LightweightCnnExporter

from .test_utils import (
    create_random_input,
    verify_inference_export,
    verify_onnxruntime_compatibility,
    verify_trainable_params,
    verify_training_export,
)


@pytest.mark.inference
class TestLightweightCNNInference:
    """Test Lightweight CNN model inference mode export."""

    def test_lightweight_cnn_inference_export(self, model_test_dir, lightweight_cnn_config):
        """Test Lightweight CNN model inference export with full verification."""
        exporter = LightweightCnnExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in lightweight_cnn_config.items()}

        expected_input_shape = [
            lightweight_cnn_config["batch_size"],
            lightweight_cnn_config["input_channels"],
            lightweight_cnn_config["input_height"],
            lightweight_cnn_config["input_width"],
        ]

        verify_inference_export(
            exporter,
            model_test_dir,
            expected_input_shape=expected_input_shape,
            expected_batch_size=lightweight_cnn_config["batch_size"],
            expected_output_classes=lightweight_cnn_config["num_classes"],
        )

    def test_lightweight_cnn_onnxruntime_inference(self, model_test_dir, lightweight_cnn_config):
        """Test exported Lightweight CNN model runs correctly with ONNX Runtime."""
        exporter = LightweightCnnExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in lightweight_cnn_config.items()}
        onnx_file = exporter.export(mode="infer")

        input_shape = (
            lightweight_cnn_config["batch_size"],
            lightweight_cnn_config["input_channels"],
            lightweight_cnn_config["input_height"],
            lightweight_cnn_config["input_width"],
        )
        test_input = create_random_input(input_shape)
        expected_output_shape = (
            lightweight_cnn_config["batch_size"],
            lightweight_cnn_config["num_classes"],
        )
        verify_onnxruntime_compatibility(
            onnx_file, test_input, expected_output_shape, input_name="input"
        )

    def test_lightweight_cnn_trainable_params(self, model_test_dir, lightweight_cnn_config):
        """Test Lightweight CNN trainable parameter identification."""
        exporter = LightweightCnnExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in lightweight_cnn_config.items()}
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        verify_trainable_params(exporter, model)

    def test_lightweight_cnn_conv_only_strategy(self, model_test_dir, lightweight_cnn_config):
        """Test Lightweight CNN conv-only training strategy (freeze fc layer)."""
        exporter = LightweightCnnExporter(save_path=model_test_dir)
        cfg = {k: v for k, v in lightweight_cnn_config.items()}
        cfg["training_strategy"] = "conv_only"
        exporter._config_overrides = cfg
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        all_params = [name for name, _ in model.named_parameters()]
        trainable = exporter.get_trainable_params(all_params)

        assert len(trainable) > 0
        assert all(not p.startswith("fc") for p in trainable)


@pytest.mark.training
class TestLightweightCNNTraining:
    """Test Lightweight CNN model training mode export."""

    def test_lightweight_cnn_training_export(self, model_test_dir, lightweight_cnn_config):
        """Test Lightweight CNN training graph generation (smoke test)."""
        exporter = LightweightCnnExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            **lightweight_cnn_config,
            "n_batches": 2,
            "n_accum": 1,
            "dataset": "random",
        }
        onnx_file = verify_training_export(exporter, model_test_dir)
        assert os.path.exists(onnx_file)

    def test_lightweight_cnn_training_artifacts(self, model_test_dir, lightweight_cnn_config):
        """Test Lightweight CNN training artifacts are generated correctly."""
        exporter = LightweightCnnExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            **lightweight_cnn_config,
            "n_batches": 2,
            "n_accum": 1,
            "dataset": "random",
        }
        verify_training_export(
            exporter,
            model_test_dir,
            required_artifacts=["checkpoint", "optimizer_model.onnx", "eval_model.onnx"],
        )

    def test_lightweight_cnn_training_npz_layout(self, model_test_dir, lightweight_cnn_config):
        """Test inputs.npz / outputs.npz are generated with the correct layout."""
        n_batches = 4
        exporter = LightweightCnnExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            **lightweight_cnn_config,
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
