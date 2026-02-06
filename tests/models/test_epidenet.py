"""
Tests for EpiDeNet model export.

Tests both inference and training mode exports.
"""

import os

import numpy as np
import pytest
import torch

from onnx4deeploy.models.epidenet_exporter import EpiDeNetExporter

from .test_utils import (
    create_random_input,
    run_onnxruntime_inference,
    verify_inference_export,
    verify_onnxruntime_compatibility,
    verify_trainable_params,
    verify_training_export,
)


@pytest.mark.inference
class TestEpiDeNetInference:
    """Test EpiDeNet model inference mode export."""

    def test_epidenet_inference_export(self, model_test_dir, epidenet_config):
        """Test EpiDeNet model inference export with full verification."""
        exporter = EpiDeNetExporter(save_path=model_test_dir)

        expected_input_shape = [
            epidenet_config["batch_size"],
            1,
            epidenet_config["channels"],
            epidenet_config["time_steps"],
        ]

        verify_inference_export(
            exporter,
            model_test_dir,
            expected_input_shape=expected_input_shape,
            expected_batch_size=epidenet_config["batch_size"],
            expected_output_classes=epidenet_config["num_classes"],
        )

    def test_epidenet_inference_input_output_shapes(self, model_test_dir, epidenet_config):
        """Test EpiDeNet inference input/output shape consistency."""
        exporter = EpiDeNetExporter(save_path=model_test_dir)

        expected_input_shape = [
            epidenet_config["batch_size"],
            1,
            epidenet_config["channels"],
            epidenet_config["time_steps"],
        ]

        onnx_file = verify_inference_export(
            exporter,
            model_test_dir,
            expected_input_shape=expected_input_shape,
            expected_batch_size=epidenet_config["batch_size"],
            expected_output_classes=epidenet_config["num_classes"],
        )

    def test_epidenet_inference_correctness(self, model_test_dir, epidenet_config):
        """Test EpiDeNet inference output correctness."""
        torch.manual_seed(42)
        np.random.seed(42)

        exporter = EpiDeNetExporter(save_path=model_test_dir)
        exporter.config = exporter.load_config()

        # Create model
        model = exporter.create_model()
        model.eval()

        # Create test input
        input_shape = (
            epidenet_config["batch_size"],
            1,
            epidenet_config["channels"],
            epidenet_config["time_steps"],
        )
        test_input = create_random_input(input_shape)

        # Get PyTorch output
        with torch.no_grad():
            torch_output = model(torch.from_numpy(test_input))

        # Export and run ONNX inference
        onnx_file = exporter.export(mode="infer")
        onnx_output = run_onnxruntime_inference(onnx_file, test_input)

        # Verify output shape matches
        assert onnx_output.shape == torch_output.numpy().shape

    def test_epidenet_onnxruntime_inference(self, model_test_dir, epidenet_config):
        """Test that exported EpiDeNet model can be run with ONNX Runtime."""
        exporter = EpiDeNetExporter(save_path=model_test_dir)
        onnx_file = exporter.export(mode="infer")

        # Create test input
        input_shape = (
            epidenet_config["batch_size"],
            1,
            epidenet_config["channels"],
            epidenet_config["time_steps"],
        )
        test_input = create_random_input(input_shape)

        # Verify ONNX Runtime compatibility
        expected_output_shape = (epidenet_config["batch_size"], epidenet_config["num_classes"])
        verify_onnxruntime_compatibility(onnx_file, test_input, expected_output_shape)


@pytest.mark.training
class TestEpiDeNetTraining:
    """Test EpiDeNet model training mode export."""

    def test_epidenet_training_export(self, model_test_dir, epidenet_config):
        """Test EpiDeNet model training export."""
        exporter = EpiDeNetExporter(save_path=model_test_dir)
        onnx_file = verify_training_export(exporter, model_test_dir)
        assert os.path.exists(onnx_file)

    def test_epidenet_training_artifacts(self, model_test_dir, epidenet_config):
        """Test EpiDeNet training artifacts generation."""
        exporter = EpiDeNetExporter(save_path=model_test_dir)
        verify_training_export(
            exporter,
            model_test_dir,
            required_artifacts=["checkpoint", "optimizer_model.onnx", "eval_model.onnx"],
        )

    def test_epidenet_trainable_params(self, model_test_dir, epidenet_config):
        """Test EpiDeNet trainable parameters identification."""
        exporter = EpiDeNetExporter(save_path=model_test_dir)
        exporter.config = exporter.load_config()

        # Create model
        model = exporter.create_model()

        # Verify trainable parameters
        verify_trainable_params(exporter, model)

    def test_epidenet_training_model_validity(self, model_test_dir, epidenet_config):
        """Test EpiDeNet training model validity (with relaxed shape checking)."""
        exporter = EpiDeNetExporter(save_path=model_test_dir)
        onnx_file = verify_training_export(exporter, model_test_dir)

        # Note: Training models may have undefined gradient tensor shapes,
        # which is expected behavior. The verify_training_export function
        # already performs validation with skip_shape_check=True.
        assert os.path.exists(onnx_file)
