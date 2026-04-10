# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Tests for Simple CNN model export.

Tests both inference and training mode exports.
"""

import os

import numpy as np
import pytest
import torch

from onnx4deeploy.models.simple_cnn_exporter import SimpleCnnExporter

from .test_utils import (
    create_random_input,
    run_onnxruntime_inference,
    verify_inference_export,
    verify_onnxruntime_compatibility,
    verify_trainable_params,
    verify_training_export,
)


@pytest.mark.inference
class TestSimpleCnnInference:
    """Test Simple CNN model inference mode export."""

    def test_simple_cnn_inference_export(self, model_test_dir, simple_cnn_config):
        """Test Simple CNN model inference export with full verification."""
        exporter = SimpleCnnExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in simple_cnn_config.items()}

        expected_input_shape = [
            simple_cnn_config["batch_size"],
            simple_cnn_config["input_channels"],
            simple_cnn_config["input_height"],
            simple_cnn_config["input_width"],
        ]

        verify_inference_export(
            exporter,
            model_test_dir,
            expected_input_shape=expected_input_shape,
            expected_batch_size=simple_cnn_config["batch_size"],
            expected_output_classes=simple_cnn_config["num_classes"],
        )

    def test_simple_cnn_inference_correctness(self, model_test_dir, simple_cnn_config):
        """Test Simple CNN inference output correctness."""
        torch.manual_seed(42)
        np.random.seed(42)

        exporter = SimpleCnnExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in simple_cnn_config.items()}
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        model.eval()

        input_shape = (
            simple_cnn_config["batch_size"],
            simple_cnn_config["input_channels"],
            simple_cnn_config["input_height"],
            simple_cnn_config["input_width"],
        )
        test_input = create_random_input(input_shape)

        with torch.no_grad():
            torch_output = model(torch.from_numpy(test_input))

        onnx_file = exporter.export(mode="infer")
        onnx_output = run_onnxruntime_inference(onnx_file, test_input)

        assert onnx_output.shape == torch_output.numpy().shape

    def test_simple_cnn_onnxruntime_inference(self, model_test_dir, simple_cnn_config):
        """Test that exported Simple CNN model can be run with ONNX Runtime."""
        exporter = SimpleCnnExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in simple_cnn_config.items()}
        onnx_file = exporter.export(mode="infer")

        input_shape = (
            simple_cnn_config["batch_size"],
            simple_cnn_config["input_channels"],
            simple_cnn_config["input_height"],
            simple_cnn_config["input_width"],
        )
        test_input = create_random_input(input_shape)

        expected_output_shape = (
            simple_cnn_config["batch_size"],
            simple_cnn_config["num_classes"],
        )
        verify_onnxruntime_compatibility(onnx_file, test_input, expected_output_shape)


@pytest.mark.training
class TestSimpleCnnTraining:
    """Test Simple CNN model training mode export."""

    def test_simple_cnn_training_export(self, model_test_dir, simple_cnn_config):
        """Test Simple CNN training export with random data (fast smoke test)."""
        exporter = SimpleCnnExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            **simple_cnn_config,
            "n_batches": 4,
            "n_accum": 1,
            "dataset": "random",
        }
        onnx_file = verify_training_export(exporter, model_test_dir)
        assert os.path.exists(onnx_file)

    def test_simple_cnn_training_artifacts(self, model_test_dir, simple_cnn_config):
        """Test Simple CNN training artifacts generation."""
        exporter = SimpleCnnExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            **simple_cnn_config,
            "n_batches": 4,
            "n_accum": 1,
            "dataset": "random",
        }
        verify_training_export(
            exporter,
            model_test_dir,
            required_artifacts=["checkpoint", "optimizer_model.onnx", "eval_model.onnx"],
        )

    def test_simple_cnn_trainable_params(self, model_test_dir, simple_cnn_config):
        """Test Simple CNN trainable parameters identification."""
        exporter = SimpleCnnExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in simple_cnn_config.items()}
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        verify_trainable_params(exporter, model)
