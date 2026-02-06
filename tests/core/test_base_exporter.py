# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Tests for onnx4deeploy.core.base_exporter module.

These tests verify the BaseONNXExporter abstract class and export functionality.
"""

import os

import onnx
import pytest
import torch
import torch.nn as nn

from onnx4deeploy.core.base_exporter import BaseONNXExporter, ExportMode


class SimpleTestModel(nn.Module):
    """Simple PyTorch model for testing."""

    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimpleTestExporter(BaseONNXExporter):
    """Concrete implementation of BaseONNXExporter for testing."""

    def load_config(self):
        """Load test configuration."""
        return {
            "input_size": 10,
            "hidden_size": 20,
            "output_size": 5,
            "batch_size": 1,
            "opset_version": 12,
        }

    def create_model(self):
        """Create simple test model."""
        return SimpleTestModel(
            self.config["input_size"], self.config["hidden_size"], self.config["output_size"]
        )

    def get_input_shape(self):
        """Get input shape for test model."""
        return (self.config["batch_size"], self.config["input_size"])

    def get_trainable_params(self, all_param_names):
        """Get trainable parameters - make only fc2 trainable for testing."""
        return [name for name in all_param_names if "fc2" in name]


class TestBaseONNXExporter:
    """Test BaseONNXExporter functionality."""

    def test_exporter_initialization(self, tmp_path):
        """Test that exporter can be initialized."""
        exporter = SimpleTestExporter(save_path=str(tmp_path))
        assert exporter.save_path == str(tmp_path)
        assert exporter.config is None
        assert exporter.paths is None

    def test_get_model_name(self):
        """Test model name extraction."""
        exporter = SimpleTestExporter()
        assert exporter.get_model_name() == "SimpleTest"

    def test_setup_paths_training(self, tmp_path):
        """Test path setup for training mode."""
        exporter = SimpleTestExporter(save_path=str(tmp_path))
        exporter.config = {"batch_size": 1, "opset_version": 12}
        paths = exporter.setup_paths(ExportMode.TRAINING)

        assert "output_dir" in paths
        assert "network" in paths
        assert "network_infer" in paths
        assert "network_train" in paths
        assert "network_train_optim" in paths
        assert "network_pre_sgd" in paths

    def test_setup_paths_inference(self, tmp_path):
        """Test path setup for inference mode."""
        exporter = SimpleTestExporter(save_path=str(tmp_path))
        exporter.config = {"batch_size": 1, "opset_version": 12}
        paths = exporter.setup_paths(ExportMode.INFERENCE)

        assert "output_dir" in paths
        assert "network" in paths
        # Training-specific paths should not be present
        assert "network_train" not in paths

    def test_export_to_onnx(self, tmp_path):
        """Test basic ONNX export functionality."""
        exporter = SimpleTestExporter(save_path=str(tmp_path))
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        model.eval()

        input_shape = exporter.get_input_shape()
        input_tensor = torch.randn(*input_shape, dtype=torch.float32)

        onnx_model = exporter._export_to_onnx(model, input_tensor, opset_version=12)

        # Verify ONNX model
        assert isinstance(onnx_model, onnx.ModelProto)
        assert len(onnx_model.graph.node) > 0
        assert len(onnx_model.graph.input) == 1
        assert len(onnx_model.graph.output) == 1

    def test_export_inference_mode(self, tmp_path):
        """Test full inference mode export."""
        exporter = SimpleTestExporter(save_path=str(tmp_path))
        output_file = exporter.export_inference()

        # Verify output file exists
        assert os.path.exists(output_file)
        assert output_file.endswith("network.onnx")

        # Verify ONNX model is valid
        model = onnx.load(output_file)
        onnx.checker.check_model(model)

        # Verify model structure
        assert len(model.graph.node) > 0
        assert model.graph.input[0].name == "input"
        assert model.graph.output[0].name == "output"

    def test_export_mode_parameter(self, tmp_path):
        """Test export with mode parameter."""
        exporter = SimpleTestExporter(save_path=str(tmp_path))

        # Test inference mode
        output_infer = exporter.export(mode="infer")
        assert os.path.exists(output_infer)

        # Verify the file is valid ONNX
        model = onnx.load(output_infer)
        onnx.checker.check_model(model)

    def test_invalid_export_mode(self, tmp_path):
        """Test that invalid mode raises error."""
        exporter = SimpleTestExporter(save_path=str(tmp_path))

        with pytest.raises(ValueError, match="Invalid mode"):
            exporter.export(mode="invalid")

    def test_get_trainable_params(self):
        """Test trainable parameters selection."""
        exporter = SimpleTestExporter()
        exporter.config = exporter.load_config()

        all_params = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]

        trainable = exporter.get_trainable_params(all_params)

        # Should only include fc2 parameters
        assert "fc2.weight" in trainable
        assert "fc2.bias" in trainable
        assert "fc1.weight" not in trainable
        assert "fc1.bias" not in trainable


class TestExportMode:
    """Test ExportMode enum."""

    def test_export_mode_values(self):
        """Test ExportMode enum values."""
        assert ExportMode.TRAINING.value == "train"
        assert ExportMode.INFERENCE.value == "infer"

    def test_export_mode_comparison(self):
        """Test ExportMode comparison."""
        assert ExportMode.TRAINING == ExportMode.TRAINING
        assert ExportMode.INFERENCE == ExportMode.INFERENCE
        assert ExportMode.TRAINING != ExportMode.INFERENCE
