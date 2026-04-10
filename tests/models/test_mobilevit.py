# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Tests for MobileViT model export.

Tests both inference and training mode exports for the XXS variant
with a reduced input size (64x64) for CI speed.
"""

import os

import pytest

from onnx4deeploy.models.mobilevit_exporter import MobileViTExporter

from .test_utils import (
    create_random_input,
    verify_inference_export,
    verify_onnxruntime_compatibility,
    verify_trainable_params,
    verify_training_export,
)


def _expected_input_shape(cfg):
    return [cfg["batch_size"], cfg["input_channels"], cfg["img_size"], cfg["img_size"]]


@pytest.mark.inference
class TestMobileViTInference:
    """Test MobileViT model inference mode export."""

    def test_mobilevit_inference_export(self, model_test_dir, mobilevit_config):
        """Test MobileViT inference export with full verification."""
        exporter = MobileViTExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in mobilevit_config.items()}

        verify_inference_export(
            exporter,
            model_test_dir,
            expected_input_shape=_expected_input_shape(mobilevit_config),
            expected_batch_size=mobilevit_config["batch_size"],
            expected_output_classes=mobilevit_config["num_classes"],
        )

    def test_mobilevit_onnxruntime_inference(self, model_test_dir, mobilevit_config):
        """Test exported MobileViT model runs with ONNX Runtime."""
        exporter = MobileViTExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in mobilevit_config.items()}
        onnx_file = exporter.export(mode="infer")

        input_shape = tuple(_expected_input_shape(mobilevit_config))
        test_input = create_random_input(input_shape)

        expected_output_shape = (
            mobilevit_config["batch_size"],
            mobilevit_config["num_classes"],
        )
        verify_onnxruntime_compatibility(onnx_file, test_input, expected_output_shape)


@pytest.mark.training
class TestMobileViTTraining:
    """Test MobileViT model training mode export."""

    def test_mobilevit_training_export(self, model_test_dir, mobilevit_config):
        """Test MobileViT training export."""
        exporter = MobileViTExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in mobilevit_config.items()}
        onnx_file = verify_training_export(exporter, model_test_dir)
        assert os.path.exists(onnx_file)

    def test_mobilevit_training_artifacts(self, model_test_dir, mobilevit_config):
        """Test MobileViT training artifacts generation."""
        exporter = MobileViTExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in mobilevit_config.items()}
        verify_training_export(
            exporter,
            model_test_dir,
            required_artifacts=["checkpoint", "optimizer_model.onnx", "eval_model.onnx"],
        )

    def test_mobilevit_trainable_params(self, model_test_dir, mobilevit_config):
        """Test MobileViT trainable parameters identification."""
        exporter = MobileViTExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in mobilevit_config.items()}
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        verify_trainable_params(exporter, model)
