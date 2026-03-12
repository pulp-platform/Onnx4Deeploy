# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Tests for FC Autoencoder model export (MLperf Tiny Anomaly Detection benchmark).

Key difference from classification models:
- Loss: MSELoss (reconstruction), not CrossEntropyLoss
- No int64 label input; ORT receives the input features as both data and "labels"
- outputs.npz contains MSE reconstruction losses (lower = better reconstruction)
"""

import os

import numpy as np
import pytest
import torch

from onnx4deeploy.models.autoencoder_exporter import AutoencoderExporter

from .test_utils import (
    create_random_input,
    verify_inference_export,
    verify_onnxruntime_compatibility,
    verify_trainable_params,
    verify_training_export,
)


@pytest.mark.inference
class TestAutoencoderInference:
    """Test Autoencoder model inference mode export."""

    def test_autoencoder_inference_export(self, model_test_dir, autoencoder_config):
        """Test Autoencoder model inference export with full verification."""
        exporter = AutoencoderExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in autoencoder_config.items()}

        # Autoencoder input shape: (batch_size, input_dim)
        expected_input_shape = [autoencoder_config["batch_size"], autoencoder_config["input_dim"]]

        verify_inference_export(
            exporter,
            model_test_dir,
            expected_input_shape=expected_input_shape,
            expected_batch_size=autoencoder_config["batch_size"],
            expected_output_classes=autoencoder_config["input_dim"],  # output == input dim
        )

    def test_autoencoder_onnxruntime_inference(self, model_test_dir, autoencoder_config):
        """Test exported Autoencoder model runs correctly with ONNX Runtime."""
        exporter = AutoencoderExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in autoencoder_config.items()}
        onnx_file = exporter.export(mode="infer")

        input_shape = (autoencoder_config["batch_size"], autoencoder_config["input_dim"])
        test_input = create_random_input(input_shape)
        expected_output_shape = (autoencoder_config["batch_size"], autoencoder_config["input_dim"])
        verify_onnxruntime_compatibility(
            onnx_file, test_input, expected_output_shape, input_name="input"
        )

    def test_autoencoder_reconstruction_output_shape(self, model_test_dir, autoencoder_config):
        """Test that Autoencoder output shape matches input shape (reconstruction)."""
        torch.manual_seed(42)
        np.random.seed(42)

        exporter = AutoencoderExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in autoencoder_config.items()}
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        model.eval()

        input_shape = (autoencoder_config["batch_size"], autoencoder_config["input_dim"])
        test_input = torch.randn(*input_shape)

        with torch.no_grad():
            output = model(test_input)

        assert (
            output.shape == test_input.shape
        ), f"Autoencoder output shape {output.shape} != input shape {test_input.shape}"

    def test_autoencoder_trainable_params(self, model_test_dir, autoencoder_config):
        """Test Autoencoder trainable parameter identification."""
        exporter = AutoencoderExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in autoencoder_config.items()}
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        verify_trainable_params(exporter, model)

    def test_autoencoder_mseloss_type(self, model_test_dir, autoencoder_config):
        """Test that Autoencoder uses MSELoss (not CrossEntropyLoss)."""
        from onnxruntime.training import artifacts

        exporter = AutoencoderExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in autoencoder_config.items()}
        exporter.config = exporter.load_config()

        loss_type = exporter.get_loss_type()
        assert loss_type == artifacts.LossType.MSELoss, f"Expected MSELoss, got {loss_type}"


@pytest.mark.training
class TestAutoencoderTraining:
    """Test Autoencoder model training mode export."""

    def test_autoencoder_training_export(self, model_test_dir, autoencoder_config):
        """Test Autoencoder training graph generation (smoke test)."""
        exporter = AutoencoderExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            **autoencoder_config,
            "n_batches": 2,
            "n_accum": 1,
        }
        onnx_file = verify_training_export(exporter, model_test_dir)
        assert os.path.exists(onnx_file)

    def test_autoencoder_training_artifacts(self, model_test_dir, autoencoder_config):
        """Test Autoencoder training artifacts are generated correctly."""
        exporter = AutoencoderExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            **autoencoder_config,
            "n_batches": 2,
            "n_accum": 1,
        }
        verify_training_export(
            exporter,
            model_test_dir,
            required_artifacts=["checkpoint", "optimizer_model.onnx", "eval_model.onnx"],
        )

    def test_autoencoder_training_npz_layout(self, model_test_dir, autoencoder_config):
        """Test inputs.npz / outputs.npz are generated with the correct layout."""
        n_batches = 4
        exporter = AutoencoderExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            **autoencoder_config,
            "n_batches": n_batches,
            "n_accum": 1,
        }
        onnx_file = verify_training_export(exporter, model_test_dir)
        output_dir = os.path.dirname(onnx_file)

        npz_in = np.load(os.path.join(output_dir, "inputs.npz"), allow_pickle=True)
        assert "meta_data_size" in npz_in
        assert "meta_n_batches" in npz_in
        assert int(npz_in["meta_n_batches"]) == n_batches

        npz_out = np.load(os.path.join(output_dir, "outputs.npz"), allow_pickle=True)
        assert "loss" in npz_out, "outputs.npz missing reconstruction loss"
        # MSE loss should be non-negative
        assert all(v >= 0 for v in npz_out["loss"]), "MSE loss values should be non-negative"
