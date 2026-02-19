# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Shared ONNX utility functions for model export and training.

This module contains common functions used across different model export scripts,
extracted from duplicate code in CCT, EpiDeNet, and MI-BMInet.
"""

import os
import re
from typing import Dict

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from onnx import numpy_helper


def load_train_config(config_filename="config.yaml") -> float:
    """
    Load training configuration from config.yaml.

    Args:
        config_filename: Name of the YAML config file

    Returns:
        Learning rate value

    Note:
        This function was duplicated in CCT, EpiDeNet, and MI-BMInet.
    """
    script_dir = os.getcwd()  # Will be called from script directory
    config_file = os.path.join(script_dir, config_filename)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f).get("training", {})

    return config.get("learning_rate", 0.001)


def create_test_input_output(
    base_path: str,
    batch_size: int,
    C: int,
    T: int,
    N: int,
    input_data: np.ndarray = None,
    seed: int = 42,
) -> str:
    """
    Create test input/output data for ONNX training model.

    This function:
    1. Generates random input data and labels
    2. Saves them to inputs.npz
    3. Runs the training model to get gradients
    4. Applies SGD updates to parameters
    5. Saves updated parameters to outputs.npz

    Args:
        base_path: Directory to save input/output files
        batch_size: Batch size for input data
        C: Number of channels
        T: Number of time steps
        N: Number of classes
        input_data: Optional pre-generated input data
        seed: Random seed for reproducibility

    Returns:
        Path to the input file

    Note:
        This function was duplicated in EpiDeNet and MI-BMInet (86 lines each).
    """
    np.random.seed(seed)

    # Generate input data if not provided
    if input_data is None:
        input_data = np.random.randn(batch_size, 1, C, T).astype(np.float32)

    # Generate labels: class indices for CrossEntropyLoss
    labels = np.random.randint(0, N, size=(batch_size,)).astype(np.int64)

    # Save inputs
    input_file = os.path.join(base_path, "inputs.npz")
    np.savez(input_file, input=input_data, labels=labels)
    print(f"✅ Test input data saved to {input_file}")
    print(f"   - Input shape: {input_data.shape}")
    print(f"   - Labels shape: {labels.shape}")

    # Run the training model to get outputs (gradients)
    network_train_path = os.path.join(base_path, "network_train.onnx")
    output_file = os.path.join(base_path, "outputs.npz")

    if not os.path.exists(network_train_path):
        print(f"⚠️ Training model not found: {network_train_path}")
        print("   Outputs will be generated after model creation")
        return input_file

    # Run inference to get gradients
    try:
        ort_session = ort.InferenceSession(network_train_path, providers=["CPUExecutionProvider"])
        output_names = [output.name for output in ort_session.get_outputs()]
        print(f"Training model has {len(output_names)} outputs")

        outputs = ort_session.run(None, {"input": input_data, "labels": labels})
        outputs_dict = {name: outputs[i] for i, name in enumerate(output_names)}

        # Load model to get parameter names
        model = onnx.load(network_train_path)
        param_names = [init.name for init in model.graph.initializer]

        # Map gradients to parameters
        learning_rate = load_train_config()
        grad_to_param_map = {}

        for output_name in output_names:
            if "grad" in output_name.lower():
                # Try to match gradient to parameter
                potential_param = re.sub(r"_grad.*$", "", output_name, flags=re.IGNORECASE)
                if potential_param in param_names:
                    grad_to_param_map[potential_param] = output_name
                    print(f"Mapped gradient {output_name} to parameter {potential_param}")

        # Apply SGD updates to parameters
        sgd_outputs = {}
        for param_name, grad_name in grad_to_param_map.items():
            # Get original parameter
            param_value = None
            for initializer in model.graph.initializer:
                if initializer.name == param_name:
                    param_value = numpy_helper.to_array(initializer)
                    break

            if param_value is not None:
                grad_value = outputs_dict[grad_name]
                # Apply SGD: param = param - lr * grad
                param_updated = param_value - learning_rate * grad_value

                updated_name = f"{param_name}_updated"
                sgd_outputs[updated_name] = param_updated
                print(f"✅ Applied SGD update to {param_name}")

        if sgd_outputs:
            np.savez(output_file, **sgd_outputs)
            print(f"✅ Updated parameters saved to {output_file}")
        else:
            print("⚠️ No parameters were updated")

    except Exception as e:
        print(f"⚠️ Error running training model: {e}")
        print("   Outputs will be generated later")

    return input_file


def randomize_onnx_initializers(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Randomize initializers in ONNX model for testing.

    Args:
        onnx_model: ONNX model to randomize

    Returns:
        ONNX model with randomized initializers

    Note:
        This is used across different models for testing purposes.
    """
    for initializer in onnx_model.graph.initializer:
        # Get the current tensor as numpy array
        tensor = numpy_helper.to_array(initializer)

        # Randomize the values
        print(F"Randomizing initializer '{initializer.name}' with shape {tensor.shape}")
    
        randomized_tensor = np.random.randn(*tensor.shape)
        if not isinstance(randomized_tensor, np.ndarray):
            randomized_tensor = np.array(randomized_tensor)
        randomized_tensor = randomized_tensor.astype(tensor.dtype)

        # Update the initializer
        new_initializer = numpy_helper.from_array(randomized_tensor, initializer.name)
        initializer.CopyFrom(new_initializer)

    return onnx_model


def print_model_info(onnx_path: str):
    """
    Print information about ONNX model.

    Args:
        onnx_path: Path to ONNX model file
    """
    model = onnx.load(onnx_path)
    print(f"\n📊 Model Info: {os.path.basename(onnx_path)}")
    print(f"   Inputs: {len(model.graph.input)}")
    print(f"   Outputs: {len(model.graph.output)}")
    print(f"   Initializers: {len(model.graph.initializer)}")
    print(f"   Nodes: {len(model.graph.node)}")


def setup_output_paths(base_path: str, model_name: str, config_str: str = "") -> Dict[str, str]:
    """
    Setup standard output directory and file paths.

    Args:
        base_path: Base directory for output
        model_name: Name of the model (e.g., "CCT", "EpiDeNet")
        config_str: Configuration string to append to folder name

    Returns:
        Dictionary of file paths
    """
    folder_name = f"{model_name}_train{config_str}" if config_str else f"{model_name}_train"
    output_dir = base_path if base_path else os.path.join(os.getcwd(), "onnx", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    return {
        "output_dir": output_dir,
        "network_infer": os.path.join(output_dir, "network_infer.onnx"),
        "network_train": os.path.join(output_dir, "network_train.onnx"),
        "network": os.path.join(output_dir, "network.onnx"),
        "network_train_optim": os.path.join(output_dir, "network_train_optim.onnx"),
        "network_pre_sgd": os.path.join(output_dir, "network_pre_sgd.onnx"),
    }


def ensure_output_dir(path: str):
    """
    Ensure output directory exists.

    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)
    print(f"📁 Output directory: {path}")
