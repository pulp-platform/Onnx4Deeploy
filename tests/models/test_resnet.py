# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Tests for ResNet model export (MLperf Tiny Image Classification benchmark).

Tests inference and training graph generation for ResNet-8 (CIFAR-10 / MLperf Tiny IC).
"""

import os

import numpy as np
import pytest
import torch

from onnx4deeploy.models.resnet_exporter import ResNetExporter

from .test_utils import (
    create_random_input,
    run_onnxruntime_inference,
    verify_inference_export,
    verify_onnxruntime_compatibility,
    verify_trainable_params,
    verify_training_export,
)


@pytest.mark.inference
class TestResNetInference:
    """Test ResNet-8 model inference mode export."""

    def test_resnet8_inference_export(self, model_test_dir, resnet_config):
        """Test ResNet-8 model inference export with full verification."""
        exporter = ResNetExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in resnet_config.items()}

        expected_input_shape = [
            resnet_config["batch_size"],
            resnet_config["input_channels"],
            resnet_config["img_size"],
            resnet_config["img_size"],
        ]

        verify_inference_export(
            exporter,
            model_test_dir,
            expected_input_shape=expected_input_shape,
            expected_batch_size=resnet_config["batch_size"],
            expected_output_classes=resnet_config["num_classes"],
        )

    def test_resnet8_onnxruntime_inference(self, model_test_dir, resnet_config):
        """Test exported ResNet-8 model runs correctly with ONNX Runtime."""
        exporter = ResNetExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in resnet_config.items()}
        onnx_file = exporter.export(mode="infer")

        input_shape = (
            resnet_config["batch_size"],
            resnet_config["input_channels"],
            resnet_config["img_size"],
            resnet_config["img_size"],
        )
        test_input = create_random_input(input_shape)
        expected_output_shape = (resnet_config["batch_size"], resnet_config["num_classes"])
        verify_onnxruntime_compatibility(
            onnx_file, test_input, expected_output_shape, input_name="input"
        )

    def test_resnet8_inference_correctness(self, model_test_dir, resnet_config):
        """Test ResNet-8 PyTorch vs ONNX output agreement."""
        torch.manual_seed(42)
        np.random.seed(42)

        exporter = ResNetExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in resnet_config.items()}
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        model.eval()

        input_shape = (
            resnet_config["batch_size"],
            resnet_config["input_channels"],
            resnet_config["img_size"],
            resnet_config["img_size"],
        )
        test_input = create_random_input(input_shape)

        with torch.no_grad():
            torch_output = model(torch.from_numpy(test_input))

        onnx_file = exporter.export(mode="infer")
        onnx_output = run_onnxruntime_inference(onnx_file, test_input)

        assert onnx_output.shape == torch_output.numpy().shape

    def test_resnet8_trainable_params(self, model_test_dir, resnet_config):
        """Test ResNet-8 trainable parameter identification."""
        exporter = ResNetExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in resnet_config.items()}
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        verify_trainable_params(exporter, model)

    def test_resnet8_last_layer_strategy(self, model_test_dir, resnet_config):
        """Test ResNet-8 last-layer-only training strategy."""
        exporter = ResNetExporter(save_path=model_test_dir)
        cfg = {k: v for k, v in resnet_config.items()}
        cfg["training_strategy"] = "last_layer"
        exporter._config_overrides = cfg
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        all_params = [name for name, _ in model.named_parameters()]
        trainable = exporter.get_trainable_params(all_params)

        assert len(trainable) > 0, "No trainable params with last_layer strategy"
        assert len(trainable) < len(all_params), "last_layer should freeze some params"
        assert all("fc" in p for p in trainable), "last_layer should only include fc params"


@pytest.mark.training
class TestResNetTraining:
    """Test ResNet-8 model training mode export."""

    def test_resnet8_training_export(self, model_test_dir, resnet_config):
        """Test ResNet-8 training graph generation (smoke test with random data)."""
        exporter = ResNetExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            **resnet_config,
            "n_batches": 2,
            "n_accum": 1,
            "dataset": "random",
        }
        onnx_file = verify_training_export(exporter, model_test_dir)
        assert os.path.exists(onnx_file)

    def test_resnet8_training_artifacts(self, model_test_dir, resnet_config):
        """Test ResNet-8 training artifacts are generated correctly."""
        exporter = ResNetExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            **resnet_config,
            "n_batches": 2,
            "n_accum": 1,
            "dataset": "random",
        }
        verify_training_export(
            exporter,
            model_test_dir,
            required_artifacts=["checkpoint", "optimizer_model.onnx", "eval_model.onnx"],
        )

    def test_resnet8_training_npz_layout(self, model_test_dir, resnet_config):
        """Test inputs.npz / outputs.npz are generated with the correct layout."""
        n_batches = 4
        exporter = ResNetExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            **resnet_config,
            "n_batches": n_batches,
            "n_accum": 1,
            "dataset": "random",
        }
        onnx_file = verify_training_export(exporter, model_test_dir)
        output_dir = os.path.dirname(onnx_file)

        npz_in = np.load(os.path.join(output_dir, "inputs.npz"), allow_pickle=True)
        assert "meta_data_size" in npz_in, "inputs.npz missing meta_data_size"
        assert "meta_n_batches" in npz_in, "inputs.npz missing meta_n_batches"
        assert int(npz_in["meta_n_batches"]) == n_batches

        npz_out = np.load(os.path.join(output_dir, "outputs.npz"), allow_pickle=True)
        assert "loss" in npz_out, "outputs.npz missing loss"
        assert len(npz_out["loss"]) == n_batches
