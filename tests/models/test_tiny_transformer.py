# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Tests for Tiny Transformer model export.

Tests both inference and training mode exports.
"""

import os

import numpy as np
import pytest
import torch

from onnx4deeploy.models.tiny_transformer_exporter import TinyTransformerExporter

from .test_utils import (
    create_random_input,
    run_onnxruntime_inference,
    verify_inference_export,
    verify_onnxruntime_compatibility,
    verify_trainable_params,
    verify_training_export,
)


def _expected_input_shape(cfg):
    num_patches = (cfg["img_size"] // cfg["patch_size"]) ** 2
    patch_dim = cfg["patch_size"] * cfg["patch_size"]
    return [cfg["batch_size"], num_patches, patch_dim]


@pytest.mark.inference
class TestTinyTransformerInference:
    """Test Tiny Transformer model inference mode export."""

    def test_tiny_transformer_inference_export(self, model_test_dir, tiny_transformer_config):
        """Test Tiny Transformer inference export with full verification."""
        exporter = TinyTransformerExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in tiny_transformer_config.items()}

        verify_inference_export(
            exporter,
            model_test_dir,
            expected_input_shape=_expected_input_shape(tiny_transformer_config),
            expected_batch_size=tiny_transformer_config["batch_size"],
            expected_output_classes=tiny_transformer_config["num_classes"],
        )

    def test_tiny_transformer_inference_correctness(
        self, model_test_dir, tiny_transformer_config
    ):
        """Test Tiny Transformer inference output correctness."""
        torch.manual_seed(42)
        np.random.seed(42)

        exporter = TinyTransformerExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in tiny_transformer_config.items()}
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        model.eval()

        input_shape = tuple(_expected_input_shape(tiny_transformer_config))
        test_input = create_random_input(input_shape)

        with torch.no_grad():
            torch_output = model(torch.from_numpy(test_input))

        onnx_file = exporter.export(mode="infer")
        onnx_output = run_onnxruntime_inference(onnx_file, test_input)

        assert onnx_output.shape == torch_output.numpy().shape

    def test_tiny_transformer_onnxruntime_inference(
        self, model_test_dir, tiny_transformer_config
    ):
        """Test exported Tiny Transformer model runs with ONNX Runtime."""
        exporter = TinyTransformerExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in tiny_transformer_config.items()}
        onnx_file = exporter.export(mode="infer")

        input_shape = tuple(_expected_input_shape(tiny_transformer_config))
        test_input = create_random_input(input_shape)

        expected_output_shape = (
            tiny_transformer_config["batch_size"],
            tiny_transformer_config["num_classes"],
        )
        verify_onnxruntime_compatibility(onnx_file, test_input, expected_output_shape)


@pytest.mark.training
class TestTinyTransformerTraining:
    """Test Tiny Transformer model training mode export."""

    def test_tiny_transformer_training_export(self, model_test_dir, tiny_transformer_config):
        """Test Tiny Transformer training export with random data (fast smoke test)."""
        exporter = TinyTransformerExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            **tiny_transformer_config,
            "n_batches": 4,
            "n_accum": 1,
            "dataset": "random",
        }
        onnx_file = verify_training_export(exporter, model_test_dir)
        assert os.path.exists(onnx_file)

    def test_tiny_transformer_training_artifacts(
        self, model_test_dir, tiny_transformer_config
    ):
        """Test Tiny Transformer training artifacts generation."""
        exporter = TinyTransformerExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            **tiny_transformer_config,
            "n_batches": 4,
            "n_accum": 1,
            "dataset": "random",
        }
        verify_training_export(
            exporter,
            model_test_dir,
            required_artifacts=["checkpoint", "optimizer_model.onnx", "eval_model.onnx"],
        )

    def test_tiny_transformer_trainable_params(
        self, model_test_dir, tiny_transformer_config
    ):
        """Test Tiny Transformer trainable parameters identification."""
        exporter = TinyTransformerExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in tiny_transformer_config.items()}
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        verify_trainable_params(exporter, model)
