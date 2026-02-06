"""
Tests for onnx4deeploy.core.onnx_utils module.

These tests verify the shared utility functions used across different models.
"""

import os

import numpy as np
import yaml
from onnx import TensorProto, helper, numpy_helper

from onnx4deeploy.core.onnx_utils import (
    ensure_output_dir,
    load_train_config,
    randomize_onnx_initializers,
    setup_output_paths,
)


class TestLoadTrainConfig:
    """Test load_train_config function."""

    def test_load_train_config_default(self, tmp_path):
        """Test loading training config with default learning rate."""
        config_file = tmp_path / "config.yaml"
        config_data = {"training": {"learning_rate": 0.001}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        os.chdir(tmp_path)
        lr = load_train_config("config.yaml")
        assert lr == 0.001

    def test_load_train_config_custom_lr(self, tmp_path):
        """Test loading training config with custom learning rate."""
        config_file = tmp_path / "config.yaml"
        config_data = {"training": {"learning_rate": 0.01}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        os.chdir(tmp_path)
        lr = load_train_config("config.yaml")
        assert lr == 0.01

    def test_load_train_config_missing_section(self, tmp_path):
        """Test loading config when training section is missing."""
        config_file = tmp_path / "config.yaml"
        config_data = {"model": {"name": "test"}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        os.chdir(tmp_path)
        lr = load_train_config("config.yaml")
        assert lr == 0.001  # Should return default


class TestRandomizeONNXInitializers:
    """Test randomize_onnx_initializers function."""

    def test_randomize_initializers(self):
        """Test that initializers are randomized."""
        # Create a simple ONNX model with initializers
        W = np.ones((3, 2), dtype=np.float32)
        b = np.zeros((3,), dtype=np.float32)

        W_init = numpy_helper.from_array(W, name="W")
        b_init = numpy_helper.from_array(b, name="b")

        input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
        output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

        node = helper.make_node(
            "Gemm",
            inputs=["input", "W", "b"],
            outputs=["output"],
        )

        graph = helper.make_graph(
            [node], "test_graph", [input_info], [output_info], [W_init, b_init]
        )

        model = helper.make_model(graph)

        # Randomize
        randomized_model = randomize_onnx_initializers(model)

        # Check that initializers exist
        assert len(randomized_model.graph.initializer) == 2

        # Check that values are different from original
        W_randomized = numpy_helper.to_array(randomized_model.graph.initializer[0])
        b_randomized = numpy_helper.to_array(randomized_model.graph.initializer[1])

        # Should not be all ones or all zeros anymore (with very high probability)
        assert not np.allclose(W_randomized, W)
        assert not np.allclose(b_randomized, b)

        # Check shapes are preserved
        assert W_randomized.shape == W.shape
        assert b_randomized.shape == b.shape


class TestSetupOutputPaths:
    """Test setup_output_paths function."""

    def test_setup_output_paths_default(self, tmp_path):
        """Test setting up output paths with default location."""
        os.chdir(tmp_path)
        paths = setup_output_paths(None, "TestModel", "_config1")

        assert "output_dir" in paths
        assert "network_infer" in paths
        assert "network_train" in paths
        assert "network" in paths

        # Check directory was created
        assert os.path.exists(paths["output_dir"])
        assert "TestModel_train_config1" in paths["output_dir"]

    def test_setup_output_paths_custom_base(self, tmp_path):
        """Test setting up output paths with custom base path."""
        custom_path = tmp_path / "custom"
        paths = setup_output_paths(str(custom_path), "TestModel", "_config1")

        assert paths["output_dir"] == str(custom_path)
        assert os.path.exists(paths["output_dir"])

    def test_setup_output_paths_no_config_string(self, tmp_path):
        """Test setting up output paths without config string."""
        os.chdir(tmp_path)
        paths = setup_output_paths(None, "TestModel", "")

        assert "TestModel_train" in paths["output_dir"]
        assert "_config" not in paths["output_dir"]


class TestEnsureOutputDir:
    """Test ensure_output_dir function."""

    def test_ensure_output_dir_creates_directory(self, tmp_path):
        """Test that ensure_output_dir creates directory if it doesn't exist."""
        test_dir = tmp_path / "test" / "nested" / "dir"
        assert not test_dir.exists()

        ensure_output_dir(str(test_dir))
        assert test_dir.exists()
        assert test_dir.is_dir()

    def test_ensure_output_dir_existing_directory(self, tmp_path):
        """Test that ensure_output_dir works with existing directory."""
        test_dir = tmp_path / "existing"
        test_dir.mkdir()
        assert test_dir.exists()

        # Should not raise error
        ensure_output_dir(str(test_dir))
        assert test_dir.exists()
