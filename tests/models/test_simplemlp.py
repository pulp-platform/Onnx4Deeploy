# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Tests for Simple MLP model export.

Tests both inference and training mode exports, including training with the
MNIST dataset (mirrors the canonical CI command):

    python Onnx4Deeploy.py --model Simplemlp --mode train \
        --dataset mnist --data-size 5 --n-epochs 20 --n-accum 8
"""

import math
import os

import numpy as np
import pytest
import torch

from onnx4deeploy.models.simple_mlp_exporter import SimpleMlpExporter

from .test_utils import (
    create_random_input,
    run_onnxruntime_inference,
    verify_inference_export,
    verify_onnxruntime_compatibility,
    verify_trainable_params,
    verify_training_export,
)


@pytest.mark.inference
class TestSimpleMlpInference:
    """Test Simple MLP model inference mode export."""

    def test_simplemlp_inference_export(self, model_test_dir, simplemlp_config):
        """Test Simple MLP model inference export with full verification."""
        exporter = SimpleMlpExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            "input_height": simplemlp_config["input_height"],
            "input_width": simplemlp_config["input_width"],
            "hidden_size": simplemlp_config["hidden_size"],
            "num_classes": simplemlp_config["num_classes"],
        }

        input_size = simplemlp_config["input_height"] * simplemlp_config["input_width"]
        expected_input_shape = [simplemlp_config["batch_size"], input_size]

        verify_inference_export(
            exporter,
            model_test_dir,
            expected_input_shape=expected_input_shape,
            expected_batch_size=simplemlp_config["batch_size"],
            expected_output_classes=simplemlp_config["num_classes"],
        )

    def test_simplemlp_inference_correctness(self, model_test_dir, simplemlp_config):
        """Test Simple MLP inference output correctness."""
        torch.manual_seed(42)
        np.random.seed(42)

        exporter = SimpleMlpExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            "input_height": simplemlp_config["input_height"],
            "input_width": simplemlp_config["input_width"],
            "hidden_size": simplemlp_config["hidden_size"],
            "num_classes": simplemlp_config["num_classes"],
        }
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        model.eval()

        input_size = simplemlp_config["input_height"] * simplemlp_config["input_width"]
        input_shape = (simplemlp_config["batch_size"], input_size)
        test_input = create_random_input(input_shape)

        with torch.no_grad():
            torch_output = model(torch.from_numpy(test_input))

        onnx_file = exporter.export(mode="infer")
        onnx_output = run_onnxruntime_inference(onnx_file, test_input)

        assert onnx_output.shape == torch_output.numpy().shape

    def test_simplemlp_onnxruntime_inference(self, model_test_dir, simplemlp_config):
        """Test that exported Simple MLP model can be run with ONNX Runtime."""
        exporter = SimpleMlpExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            "input_height": simplemlp_config["input_height"],
            "input_width": simplemlp_config["input_width"],
            "hidden_size": simplemlp_config["hidden_size"],
            "num_classes": simplemlp_config["num_classes"],
        }
        onnx_file = exporter.export(mode="infer")

        input_size = simplemlp_config["input_height"] * simplemlp_config["input_width"]
        input_shape = (simplemlp_config["batch_size"], input_size)
        test_input = create_random_input(input_shape)

        expected_output_shape = (simplemlp_config["batch_size"], simplemlp_config["num_classes"])
        verify_onnxruntime_compatibility(onnx_file, test_input, expected_output_shape)


@pytest.mark.training
class TestSimpleMlpTraining:
    """Test Simple MLP model training mode export."""

    def test_simplemlp_training_export(self, model_test_dir, simplemlp_config):
        """Test Simple MLP training export with random data (fast smoke test)."""
        exporter = SimpleMlpExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            "n_batches": 4,
            "n_accum": 1,
            "batch_size": 1,
            "dataset": "random",
            "input_height": simplemlp_config["input_height"],
            "input_width": simplemlp_config["input_width"],
            "hidden_size": simplemlp_config["hidden_size"],
            "num_classes": simplemlp_config["num_classes"],
        }
        onnx_file = verify_training_export(exporter, model_test_dir)
        assert os.path.exists(onnx_file)

    def test_simplemlp_training_artifacts(self, model_test_dir, simplemlp_config):
        """Test Simple MLP training artifacts generation."""
        exporter = SimpleMlpExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            "n_batches": 4,
            "n_accum": 1,
            "batch_size": 1,
            "dataset": "random",
            "input_height": simplemlp_config["input_height"],
            "input_width": simplemlp_config["input_width"],
            "hidden_size": simplemlp_config["hidden_size"],
            "num_classes": simplemlp_config["num_classes"],
        }
        verify_training_export(
            exporter,
            model_test_dir,
            required_artifacts=["checkpoint", "optimizer_model.onnx", "eval_model.onnx"],
        )

    def test_simplemlp_trainable_params(self, model_test_dir, simplemlp_config):
        """Test Simple MLP trainable parameters identification."""
        exporter = SimpleMlpExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            "input_height": simplemlp_config["input_height"],
            "input_width": simplemlp_config["input_width"],
            "hidden_size": simplemlp_config["hidden_size"],
            "num_classes": simplemlp_config["num_classes"],
        }
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        verify_trainable_params(exporter, model)

    @pytest.mark.slow
    def test_simplemlp_training_mnist(self, model_test_dir, simplemlp_config):
        """Test Simple MLP training export with MNIST dataset.

        Replicates the canonical CI command:
            python Onnx4Deeploy.py --model Simplemlp --mode train
                --dataset mnist --data-size 5 --n-epochs 20 --n-accum 8

        n_batches = ceil(20 * 5 / 8) * 8 = 104
        """
        n_epochs = 20
        data_size = 5
        n_accum = 8
        n_batches = math.ceil(n_epochs * data_size / n_accum) * n_accum  # 104

        exporter = SimpleMlpExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            "n_batches": n_batches,
            "n_accum": n_accum,
            "batch_size": 1,
            "dataset": "mnist",
            "data_size": data_size,
        }

        onnx_file = verify_training_export(exporter, model_test_dir)
        assert os.path.exists(onnx_file)

        # Verify inputs.npz was generated with the right data layout
        import numpy as np

        npz_path = os.path.join(os.path.dirname(onnx_file), "inputs.npz")
        assert os.path.exists(npz_path), f"inputs.npz not found at {npz_path}"

        npz = np.load(npz_path, allow_pickle=True)
        keys = list(npz.keys())

        # meta_data_size encodes the number of unique samples stored
        assert "meta_data_size" in keys, "inputs.npz missing meta_data_size"
        assert int(npz["meta_data_size"]) == data_size

        # meta_n_batches encodes the total training steps
        assert "meta_n_batches" in keys, "inputs.npz missing meta_n_batches"
        assert int(npz["meta_n_batches"]) == n_batches
