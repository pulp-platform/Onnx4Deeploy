# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Tests for TinyViT model export.

Tests both inference and training mode exports for the 5M-parameter variant
with a reduced input size (32x32) for CI speed.
"""

import os

import pytest

from onnx4deeploy.models.tinyvit_exporter import TinyViTExporter

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
class TestTinyViTInference:
    """Test TinyViT model inference mode export."""

    def test_tinyvit_inference_export(self, model_test_dir, tinyvit_config):
        """Test TinyViT inference export with full verification."""
        exporter = TinyViTExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in tinyvit_config.items()}

        verify_inference_export(
            exporter,
            model_test_dir,
            expected_input_shape=_expected_input_shape(tinyvit_config),
            expected_batch_size=tinyvit_config["batch_size"],
            expected_output_classes=tinyvit_config["num_classes"],
        )

    def test_tinyvit_onnxruntime_inference(self, model_test_dir, tinyvit_config):
        """Test exported TinyViT model runs with ONNX Runtime."""
        exporter = TinyViTExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in tinyvit_config.items()}
        onnx_file = exporter.export(mode="infer")

        input_shape = tuple(_expected_input_shape(tinyvit_config))
        test_input = create_random_input(input_shape)

        expected_output_shape = (
            tinyvit_config["batch_size"],
            tinyvit_config["num_classes"],
        )
        verify_onnxruntime_compatibility(onnx_file, test_input, expected_output_shape)


@pytest.mark.training
class TestTinyViTTraining:
    """Test TinyViT model training mode export."""

    def test_tinyvit_training_export(self, model_test_dir, tinyvit_config):
        """Test TinyViT training export."""
        exporter = TinyViTExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in tinyvit_config.items()}
        onnx_file = verify_training_export(exporter, model_test_dir)
        assert os.path.exists(onnx_file)

    def test_tinyvit_training_artifacts(self, model_test_dir, tinyvit_config):
        """Test TinyViT training artifacts generation."""
        exporter = TinyViTExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in tinyvit_config.items()}
        verify_training_export(
            exporter,
            model_test_dir,
            required_artifacts=["checkpoint", "optimizer_model.onnx", "eval_model.onnx"],
        )

    def test_tinyvit_trainable_params(self, model_test_dir, tinyvit_config):
        """Test TinyViT trainable parameters identification."""
        exporter = TinyViTExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in tinyvit_config.items()}
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        verify_trainable_params(exporter, model)
