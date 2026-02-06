# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
PyTorch testing and debugging utilities.

This module provides functions for creating test inputs/outputs and debugging
PyTorch models, particularly for gradient computation verification.
"""

import os
import sys
from typing import List

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper

# Global debug data storage
debug_data = {"forward_output": None, "backward_grad_output": None, "backward_grad_input": None}


def debug_hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
    """
    Forward hook for debugging PyTorch module outputs.

    Args:
        module: The PyTorch module
        input: Input tensors to the module
        output: Output tensor from the module
    """
    debug_data["forward_output"] = output.detach().cpu().numpy()


def debug_grad_hook(module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
    """
    Backward hook for debugging PyTorch module gradients.

    Args:
        module: The PyTorch module
        grad_input: Gradient tensors w.r.t. inputs
        grad_output: Gradient tensors w.r.t. outputs
    """
    debug_data["backward_grad_output"] = grad_output[0].detach().cpu().numpy()
    debug_data["backward_grad_input"] = (
        grad_input[0].detach().cpu().numpy() if grad_input[0] is not None else None
    )


def create_test_input_output_pytorch(
    base_path: str,
    batch_size: int,
    C: int,
    T: int,
    N: int,
    model_pytorch: nn.Module,
    requires_grad_list: List[str],
) -> str:
    """
    Generate reference test inputs and outputs using GroupNorm model for gradient compatibility.

    Uses MIBMINetDeployGroupNorm internally so that gradient shapes match Deeploy's GroupNormGradW.
    Loads weights from ONNX file to ensure randomized biases match between ONNX and PyTorch.

    Args:
        base_path: Base directory path for saving inputs/outputs
        batch_size: Batch size for test data
        C: Number of channels
        T: Temporal dimension size
        N: Number of output classes
        model_pytorch: PyTorch model to generate test data from
        requires_grad_list: List of parameter names that require gradients

    Returns:
        Path to the saved input file
    """
    # base_path is like /app/.../MI-BMInet/onnx/xxx, go up two levels to reach MI-BMInet
    mi_bminet_root = os.path.dirname(os.path.dirname(base_path))
    mi_bminet_model_path = os.path.join(mi_bminet_root, "mi_bminet_model")
    if mi_bminet_model_path not in sys.path:
        sys.path.insert(0, mi_bminet_model_path)
    from mi_bminet import MIBMINetDeployGroupNorm

    # 1. Prepare data
    input_data = torch.randn(batch_size, 1, C, T)
    labels = torch.randint(0, N, (batch_size,))

    # Save input
    input_file = os.path.join(base_path, "inputs.npz")
    np.savez(input_file, input=input_data.numpy(), labels=labels.numpy())
    print(f"✅ Input data saved to {input_file}")

    # 2. Load weights from ONNX file (including randomized biases)
    # This ensures PyTorch and ONNX use the same weights
    onnx_infer_path = os.path.join(base_path, "network_infer.onnx")
    print(f"📥 Loading weights from ONNX: {onnx_infer_path}")
    onnx_model = onnx.load(onnx_infer_path)

    # Convert ONNX weights to dictionary
    onnx_weights = {}
    for init in onnx_model.graph.initializer:
        onnx_weights[init.name] = numpy_helper.to_array(init)

    # 3. Create GroupNorm version of model and load ONNX weights
    model_gn = MIBMINetDeployGroupNorm(
        F1=model_pytorch.F1,
        D=model_pytorch.D,
        F2=model_pytorch.F2,
        C=model_pytorch.C,
        T=model_pytorch.T,
        N=model_pytorch.N,
        Nf=model_pytorch.Nf,
        Nf2=model_pytorch.Nf2,
    )

    # Load ONNX weights into GroupNorm model
    # Conv weights are directly copied, LayerNorm weights need reshape and mean
    model_gn.conv1.weight.data = torch.from_numpy(onnx_weights["conv1_weight"].copy())
    model_gn.conv2.weight.data = torch.from_numpy(onnx_weights["conv2_weight"].copy())
    model_gn.sep_conv1.weight.data = torch.from_numpy(onnx_weights["sep_conv1_weight"].copy())
    model_gn.sep_conv2.weight.data = torch.from_numpy(onnx_weights["sep_conv2_weight"].copy())
    model_gn.fc.weight.data = torch.from_numpy(onnx_weights["fc_weight"].copy())
    model_gn.fc.bias.data = torch.from_numpy(onnx_weights["fc_bias"].copy())

    # LayerNorm weights [C, H, W] -> GroupNorm weights [C] via mean
    for ln_name, gn_module in [
        ("layer_norm1", model_gn.layer_norm1),
        ("layer_norm2", model_gn.layer_norm2),
        ("layer_norm3", model_gn.layer_norm3),
    ]:
        w = onnx_weights[f"{ln_name}_weight"]  # [C, H, W]
        b = onnx_weights[f"{ln_name}_bias"]  # [C, H, W]
        # Convert to [C] by averaging over spatial dimensions
        gn_module.weight.data = torch.from_numpy(w.reshape(w.shape[0], -1).mean(axis=1).copy())
        gn_module.bias.data = torch.from_numpy(b.reshape(b.shape[0], -1).mean(axis=1).copy())

    print("✅ Loaded weights from ONNX into GroupNorm model")

    # 3. Execute PyTorch computation (using GroupNorm model)
    model_gn.train()
    criterion = nn.CrossEntropyLoss()

    # Import load_train_config from the io module
    from ..io.config_loader import load_train_config

    learning_rate = load_train_config()

    model_gn.zero_grad()
    output = model_gn(input_data)
    loss = criterion(output, labels)
    loss.backward()

    # 4. Selectively extract gradients and perform SGD update
    sgd_outputs = {}

    # Map ONNX-style names back to PyTorch style (for matching)
    # Example: 'fc_weight' -> 'fc.weight'
    # Note: Assumes conversion rule is '_' to '.'; deeper hierarchies may need more complex mapping
    print(f"\n📋 Filtering gradients based on selected list ({len(requires_grad_list)} params):")
    print("   Using GroupNorm model for gradient-compatible reference output")

    for name, param in model_gn.named_parameters():
        # Convert PyTorch name to ONNX format for matching
        onnx_name = name.replace(".", "_")

        if onnx_name in requires_grad_list:
            if param.grad is not None:
                # SGD update: param_new = param_old - lr * grad
                param_updated = param.data - learning_rate * param.grad.data
                param_updated_np = param_updated.cpu().numpy()

                output_key = f"{onnx_name}_updated"
                sgd_outputs[output_key] = param_updated_np
                print(f"   - [Selected] Updated {onnx_name} (shape: {param_updated_np.shape})")
    output_file = os.path.join(base_path, "outputs.npz")
    if sgd_outputs:
        np.savez(output_file, **sgd_outputs)
        print(f"✅ Selected SGD updated parameters saved to {output_file}")
    else:
        print("⚠️ Warning: No matching parameters from requires_grad_list were found or updated!")

    return input_file
