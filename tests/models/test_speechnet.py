# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Tests for SpeechNet model export (SilentWear EMG silent speech recognition).

Tests inference and training graph generation for the 5-block SpeechNet CNN
(14 EMG channels, 700 time samples, 9 classes).
"""

import os

import numpy as np
import pytest

from onnx4deeploy.models.speechnet_exporter import SpeechNetExporter

from .test_utils import (
    create_random_input,
    verify_inference_export,
    verify_onnxruntime_compatibility,
    verify_trainable_params,
    verify_training_export,
)


@pytest.mark.inference
class TestSpeechNetInference:
    """Test SpeechNet model inference mode export."""

    def test_speechnet_inference_export(self, model_test_dir, speechnet_config):
        """Test SpeechNet model inference export with full verification."""
        exporter = SpeechNetExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in speechnet_config.items()}

        expected_input_shape = [
            speechnet_config["batch_size"],
            1,
            speechnet_config["num_channels"],
            speechnet_config["time_steps"],
        ]

        verify_inference_export(
            exporter,
            model_test_dir,
            expected_input_shape=expected_input_shape,
            expected_batch_size=speechnet_config["batch_size"],
            expected_output_classes=speechnet_config["num_classes"],
        )

    def test_speechnet_onnxruntime_inference(self, model_test_dir, speechnet_config):
        """Test exported SpeechNet model runs correctly with ONNX Runtime."""
        exporter = SpeechNetExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in speechnet_config.items()}
        onnx_file = exporter.export(mode="infer")

        input_shape = (
            speechnet_config["batch_size"],
            1,
            speechnet_config["num_channels"],
            speechnet_config["time_steps"],
        )
        test_input = create_random_input(input_shape)
        expected_output_shape = (speechnet_config["batch_size"], speechnet_config["num_classes"])
        verify_onnxruntime_compatibility(
            onnx_file, test_input, expected_output_shape, input_name="input"
        )

    def test_speechnet_trainable_params(self, model_test_dir, speechnet_config):
        """Test SpeechNet trainable parameter identification."""
        exporter = SpeechNetExporter(save_path=model_test_dir)
        exporter._config_overrides = {k: v for k, v in speechnet_config.items()}
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        verify_trainable_params(exporter, model)

    def test_speechnet_last_layer_strategy(self, model_test_dir, speechnet_config):
        """Test SpeechNet last-layer-only training strategy."""
        exporter = SpeechNetExporter(save_path=model_test_dir)
        cfg = {k: v for k, v in speechnet_config.items()}
        cfg["training_strategy"] = "last_layer"
        exporter._config_overrides = cfg
        exporter.config = exporter.load_config()

        model = exporter.create_model()
        all_params = [name for name, _ in model.named_parameters()]
        trainable = exporter.get_trainable_params(all_params)

        assert len(trainable) > 0
        assert len(trainable) < len(all_params)


@pytest.mark.training
class TestSpeechNetTraining:
    """Test SpeechNet model training mode export."""

    def test_speechnet_training_export(self, model_test_dir, speechnet_config):
        """Test SpeechNet training graph generation (smoke test with random data)."""
        exporter = SpeechNetExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            **speechnet_config,
            "n_batches": 2,
            "n_accum": 1,
            "dataset": "random",
        }
        onnx_file = verify_training_export(exporter, model_test_dir)
        assert os.path.exists(onnx_file)

    def test_speechnet_training_npz_layout(self, model_test_dir, speechnet_config):
        """Test inputs.npz / outputs.npz are generated with the correct layout."""
        n_batches = 4
        exporter = SpeechNetExporter(save_path=model_test_dir)
        exporter._config_overrides = {
            **speechnet_config,
            "n_batches": n_batches,
            "n_accum": 1,
            "dataset": "random",
        }
        onnx_file = verify_training_export(exporter, model_test_dir)
        output_dir = os.path.dirname(onnx_file)

        npz_in = np.load(os.path.join(output_dir, "inputs.npz"), allow_pickle=True)
        assert "meta_data_size" in npz_in
        assert "meta_n_batches" in npz_in
        assert int(npz_in["meta_n_batches"]) == n_batches

        npz_out = np.load(os.path.join(output_dir, "outputs.npz"), allow_pickle=True)
        assert "loss" in npz_out
        assert len(npz_out["loss"]) == n_batches
