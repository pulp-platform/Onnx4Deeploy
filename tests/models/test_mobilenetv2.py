# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Tests for MobileNetV2 model export (MLperf Tiny Visual Wake Words benchmark).

Tests inference mode export for MobileNetV2-0.35.
"""

import pytest

from onnx4deeploy.models.mobilenetv2_exporter import MobileNetV2Exporter

from .test_utils import (
    create_random_input,
    verify_inference_export,
    verify_onnxruntime_compatibility,
    verify_trainable_params,
)


@pytest.mark.inference
class TestMobileNetV2Inference:
    """Test MobileNetV2 model inference mode export."""

    def test_mobilenetv2_inference_export(self, model_test_dir, mobilenetv2_config):
        """Test MobileNetV2 model inference export with full verification."""
        exporter = MobileNetV2Exporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in mobilenetv2_config.items()}

        expected_input_shape = [
            mobilenetv2_config["batch_size"],
            mobilenetv2_config["input_channels"],
            mobilenetv2_config["img_size"],
            mobilenetv2_config["img_size"],
        ]

        verify_inference_export(
            exporter,
            model_test_dir,
            expected_input_shape=expected_input_shape,
            expected_batch_size=mobilenetv2_config["batch_size"],
            expected_output_classes=mobilenetv2_config["num_classes"],
        )

    def test_mobilenetv2_onnxruntime_inference(self, model_test_dir, mobilenetv2_config):
        """Test exported MobileNetV2 model runs correctly with ONNX Runtime."""
        exporter = MobileNetV2Exporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in mobilenetv2_config.items()}
        onnx_file = exporter.export(mode="infer")

        input_shape = (
            mobilenetv2_config["batch_size"],
            mobilenetv2_config["input_channels"],
            mobilenetv2_config["img_size"],
            mobilenetv2_config["img_size"],
        )
        test_input = create_random_input(input_shape)
        expected_output_shape = (
            mobilenetv2_config["batch_size"],
            mobilenetv2_config["num_classes"],
        )
        verify_onnxruntime_compatibility(
            onnx_file, test_input, expected_output_shape, input_name="input"
        )

    def test_mobilenetv2_trainable_params(self, model_test_dir, mobilenetv2_config):
        """Test MobileNetV2 trainable parameter identification."""
        exporter = MobileNetV2Exporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in mobilenetv2_config.items()}
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        verify_trainable_params(exporter, model)

    def test_mobilenetv2_last_layer_strategy(self, model_test_dir, mobilenetv2_config):
        """Test MobileNetV2 last-layer-only training strategy."""
        exporter = MobileNetV2Exporter(save_path=model_test_dir)
        cfg = {k: v for k, v in mobilenetv2_config.items()}
        cfg["training_strategy"] = "last_layer"
        exporter._config_overrides = cfg
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        all_params = [name for name, _ in model.named_parameters()]
        trainable = exporter.get_trainable_params(all_params)

        assert len(trainable) > 0, "No trainable params with last_layer strategy"
        assert len(trainable) < len(all_params), "last_layer should freeze some params"
        assert all(
            "classifier" in p for p in trainable
        ), "last_layer should only include classifier params"
