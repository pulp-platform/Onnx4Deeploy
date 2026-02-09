# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Configuration loading utilities.

This module provides functions for loading and parsing YAML configuration files.
"""

import os
from typing import Tuple

import yaml


def load_config(
    config_filename: str = "../config.yaml",
) -> Tuple[bool, int, int, int, int, int, int, int]:
    """
    Load and parse config.yaml, returning CCT-specific parameters in a single return statement.

    Args:
        config_filename: Relative path to the config file

    Returns:
        A tuple containing:
        - pretrained (bool): Whether to use pretrained weights
        - img_size (int): Input image size
        - num_classes (int): Number of output classes
        - embedding_dim (int): Embedding dimension
        - num_heads (int): Number of attention heads
        - num_layers (int): Number of transformer layers
        - batch_size (int): Batch size for training
        - opset_version (int): ONNX opset version (default: 12)
    """
    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, config_filename)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f).get("cct", {})

    return (
        config["pretrained"],
        config["img_size"],
        config["num_classes"],
        config["embedding_dim"],
        config["num_heads"],
        config["num_layers"],
        config["batch_size"],
        config.get("opset_version", 12),  # Default value for opset_version
    )


def load_train_config(config_filename: str = "../config.yaml") -> float:
    """
    Load and parse config.yaml, returning CCT-specific parameters in a single return statement.

    Args:
        config_filename: Relative path to the config file

    Returns:
        Learning rate from the training configuration (default: 0.01)
    """
    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, config_filename)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f).get("training", {})

    return config.get("learning_rate", 0.01)