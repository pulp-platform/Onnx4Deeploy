# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Tests for SleepConViT model export (sleep stage classification).

Tests both inference and training mode exports.
"""

import os

import pytest

from onnx4deeploy.models.sleep_convit_exporter import SleepConViTExporter

from .test_utils import (
    create_random_input,
    verify_inference_export,
    verify_onnxruntime_compatibility,
    verify_trainable_params,
    verify_training_export,
)


def _expected_input_shape(cfg):
    # SleepConViT accepts shape (B, C, 1, input_length)
    return [
        cfg["batch_size"],
        cfg["input_channels"],
        1,
        cfg["input_length"],
    ]


@pytest.mark.inference
class TestSleepConViTInference:
    """Test SleepConViT model inference mode export."""

    def test_sleep_convit_inference_export(self, model_test_dir, sleep_convit_config):
        """Test SleepConViT inference export with full verification."""
        exporter = SleepConViTExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in sleep_convit_config.items()}

        verify_inference_export(
            exporter,
            model_test_dir,
            expected_input_shape=_expected_input_shape(sleep_convit_config),
            expected_batch_size=sleep_convit_config["batch_size"],
            expected_output_classes=sleep_convit_config["num_classes"],
        )

    def test_sleep_convit_onnxruntime_inference(self, model_test_dir, sleep_convit_config):
        """Test exported SleepConViT model runs with ONNX Runtime."""
        exporter = SleepConViTExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in sleep_convit_config.items()}
        onnx_file = exporter.export(mode="infer")

        input_shape = tuple(_expected_input_shape(sleep_convit_config))
        test_input = create_random_input(input_shape)

        expected_output_shape = (
            sleep_convit_config["batch_size"],
            sleep_convit_config["num_classes"],
        )
        verify_onnxruntime_compatibility(onnx_file, test_input, expected_output_shape)


@pytest.mark.training
class TestSleepConViTTraining:
    """Test SleepConViT model training mode export."""

    def test_sleep_convit_training_export(self, model_test_dir, sleep_convit_config):
        """Test SleepConViT training export."""
        exporter = SleepConViTExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in sleep_convit_config.items()}
        onnx_file = verify_training_export(exporter, model_test_dir)
        assert os.path.exists(onnx_file)

    def test_sleep_convit_training_artifacts(self, model_test_dir, sleep_convit_config):
        """Test SleepConViT training artifacts generation."""
        exporter = SleepConViTExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in sleep_convit_config.items()}
        verify_training_export(
            exporter,
            model_test_dir,
            required_artifacts=["checkpoint", "optimizer_model.onnx", "eval_model.onnx"],
        )

    def test_sleep_convit_trainable_params(self, model_test_dir, sleep_convit_config):
        """Test SleepConViT trainable parameters identification."""
        exporter = SleepConViTExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in sleep_convit_config.items()}
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        verify_trainable_params(exporter, model)
