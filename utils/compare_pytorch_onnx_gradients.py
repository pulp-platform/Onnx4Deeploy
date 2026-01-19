#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare PyTorch and ONNX Runtime gradients for a given model and node

This script performs:
1. Extract weights from ONNX model to npz
2. Run PyTorch grad_at_node.py to compute gradients
3. Run ONNX Runtime to get gradients
4. Compare and report differences

Usage:
    python compare_pytorch_onnx_gradients.py \
        --model_py path/to/model.py \
        --model_class ModelClass \
        --model_kwargs_json '{"param":"value"}' \
        --onnx_dir path/to/onnx/directory \
        --node node_name \
        [--lr learning_rate]
        

Example:
    python /app/Onnx4Deeploy/utils/compare_pytorch_onnx_gradients.py \
        --model_py /app/Onnx4Deeploy/Tests/Models/MI-BMInet/mi_bminet_model/mi_bminet.py \
        --model_class MIBMINetDeploy \
        --model_kwargs_json '{"C":8,"T":2000,"N":4}' \
        --onnx_dir Tests/Models/MI-BMInet/onnx/MI_BMINet_train_C8_T2000_F1_8_D2_N4_fc_sep_conv2 \
        --node sep_conv2 \
        --lr 0.01
"""

import argparse
import json
import os
import sys
import importlib.util
import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from typing import Dict, Any


def import_from_path(py_path: str):
    """Import a module from a file path"""
    py_path = os.path.abspath(py_path)
    mod_name = os.path.splitext(os.path.basename(py_path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_weights_to_npz(onnx_model_path: str, inputs_npz_path: str, output_npz_path: str):
    """Extract weights from ONNX model and merge with inputs.npz"""
    print("\n" + "=" * 70)
    print("📦 Step 1: Extracting weights from ONNX model")
    print("=" * 70)

    # Load existing inputs.npz
    data = np.load(inputs_npz_path)
    new_data = {key: data[key] for key in data.files}
    print(f"Loaded inputs.npz with keys: {list(data.keys())}")

    # Load ONNX model
    model = onnx.load(onnx_model_path)

    # Extract weights
    weight_count = 0
    for init in model.graph.initializer:
        tensor = numpy_helper.to_array(init)

        # Convert ONNX naming to PyTorch naming
        parts = init.name.split("_")
        if len(parts) >= 2 and parts[-1] in ["weight", "bias"]:
            param_type = parts[-1]
            module_name = "_".join(parts[:-1])
            pytorch_name = f"{module_name}.{param_type}"
            new_data[pytorch_name] = tensor
            weight_count += 1
            print(f"  ✓ {pytorch_name}: shape={tensor.shape}")

    # Save merged npz
    np.savez(output_npz_path, **new_data)
    print(f"\n✅ Saved to: {output_npz_path}")
    print(f"   Total keys: {len(new_data)} ({len(data.files)} inputs + {weight_count} weights)")

    return output_npz_path


def run_pytorch_gradients(model_py: str, model_class: str, model_kwargs: Dict[str, Any],
                          npz_path: str, node: str, output_npz: str) -> Dict[str, np.ndarray]:
    """Run PyTorch to compute gradients"""
    print("\n" + "=" * 70)
    print("🔥 Step 2: Computing gradients with PyTorch")
    print("=" * 70)

    # Import model
    module = import_from_path(model_py)
    ModelCls = getattr(module, model_class)
    model = ModelCls(**model_kwargs)

    device = torch.device("cpu")
    model = model.to(device)
    model.train()

    # Load data
    npz_dict = np.load(npz_path)
    x_np = npz_dict['input']
    labels_np = npz_dict['labels']

    x = torch.from_numpy(x_np).to(device)
    x.requires_grad_(True)
    labels = torch.from_numpy(labels_np).to(device)

    # Load weights
    sd = model.state_dict()
    loadable = {}
    for k in sd.keys():
        if k in npz_dict.files:
            loadable[k] = torch.from_numpy(npz_dict[k])

    new_sd = {k: loadable.get(k, v) for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=False)
    print(f"Loaded {len(loadable)} weights from npz")

    # Get target module
    target_module = model.get_submodule(node)

    # Hook to capture node output
    captured = {}
    def hook_fn(_m, _inp, out):
        captured["node_output"] = out[0] if isinstance(out, (tuple, list)) else out

    h = target_module.register_forward_hook(hook_fn)

    # Forward pass
    model.zero_grad(set_to_none=True)
    final_output = model(x)
    h.remove()

    # Compute loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(final_output, labels)
    print(f"PyTorch Loss: {loss.item():.10f}")

    # Backward
    loss.backward()

    # Collect gradients
    results = {
        "grad_x": x.grad.detach().cpu().numpy(),
        "loss": np.array(loss.item())
    }

    for name, param in target_module.named_parameters(recurse=False):
        if param.grad is not None:
            results[f"grad_w_{name}"] = param.grad.detach().cpu().numpy()
            print(f"  ✓ grad_w_{name}: shape={tuple(param.grad.shape)}")

    # Save
    np.savez(output_npz, **results)
    print(f"\n✅ Saved PyTorch gradients to: {output_npz}")

    return results


def run_onnx_gradients(onnx_model_path: str, inputs_npz_path: str, output_npz: str = None) -> Dict[str, np.ndarray]:
    """Run ONNX Runtime to get gradients"""
    print("\n" + "=" * 70)
    print("⚡ Step 3: Getting gradients from ONNX Runtime")
    print("=" * 70)

    # Load inputs
    inputs_data = np.load(inputs_npz_path)
    input_data = inputs_data['input']
    labels = inputs_data['labels']

    # Load and run ONNX model
    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

    print("ONNX Model Outputs:")
    for out in session.get_outputs():
        print(f"  - {out.name}")

    outputs = session.run(None, {"input": input_data, "labels": labels})
    output_names = [out.name for out in session.get_outputs()]

    # Organize results
    results = {}
    for i, name in enumerate(output_names):
        results[name] = outputs[i]
        if outputs[i].size == 1:
            print(f"  ✓ {name}: {outputs[i].item():.10f}")
        else:
            print(f"  ✓ {name}: shape={outputs[i].shape}")

    # Save to file if path provided
    if output_npz:
        np.savez(output_npz, **results)
        print(f"\n✅ Saved ONNX gradients to: {output_npz}")

    return results


def compare_gradients(pytorch_results: Dict[str, np.ndarray], onnx_results: Dict[str, np.ndarray], node: str):
    """Compare PyTorch and ONNX gradients"""
    print("\n" + "=" * 70)
    print("📊 Step 4: Comparing Results")
    print("=" * 70)

    # Compare loss
    pytorch_loss = pytorch_results.get("loss", None)
    onnx_loss = None
    for key in onnx_results.keys():
        if "loss" in key.lower():
            onnx_loss = onnx_results[key].item() if onnx_results[key].size == 1 else onnx_results[key]
            break

    if pytorch_loss is not None and onnx_loss is not None:
        pytorch_loss_val = pytorch_loss.item() if hasattr(pytorch_loss, 'item') else pytorch_loss
        onnx_loss_val = onnx_loss.item() if hasattr(onnx_loss, 'item') else onnx_loss
        loss_diff = abs(pytorch_loss_val - onnx_loss_val)
        print(f"\n1. Loss Comparison")
        print(f"   PyTorch:  {pytorch_loss_val:.10f}")
        print(f"   ONNX:     {onnx_loss_val:.10f}")
        print(f"   Diff:     {loss_diff:.2e}")
        print(f"   Status:   {'✅ MATCH' if loss_diff < 1e-6 else '⚠️ DIFFER'}")

    # Compare weight gradients
    print(f"\n2. Weight Gradients for node '{node}'")

    for pytorch_key in pytorch_results.keys():
        if not pytorch_key.startswith("grad_w_"):
            continue

        param_name = pytorch_key.replace("grad_w_", "")
        pytorch_grad = pytorch_results[pytorch_key]

        # Find corresponding ONNX gradient
        onnx_grad = None
        onnx_key = None
        for key in onnx_results.keys():
            if node in key and param_name in key and "grad" in key:
                onnx_grad = onnx_results[key]
                onnx_key = key
                break

        print(f"\n   {param_name}:")
        print(f"   PyTorch key: {pytorch_key}")
        print(f"   ONNX key:    {onnx_key if onnx_key else 'NOT FOUND'}")

        if onnx_grad is not None:
            if pytorch_grad.shape != onnx_grad.shape:
                print(f"   ⚠️ Shape mismatch: PyTorch {pytorch_grad.shape} vs ONNX {onnx_grad.shape}")
            else:
                diff = np.abs(pytorch_grad - onnx_grad)
                max_diff = diff.max()
                mean_diff = diff.mean()
                rel_error = max_diff / (np.abs(pytorch_grad).max() + 1e-10)

                print(f"   Shape:        {pytorch_grad.shape}")
                print(f"   Max diff:     {max_diff:.2e}")
                print(f"   Mean diff:    {mean_diff:.2e}")
                print(f"   Rel error:    {rel_error:.2e}")

                if max_diff < 1e-5:
                    print(f"   Status:       ✅ EXCELLENT MATCH")
                elif max_diff < 1e-3:
                    print(f"   Status:       ✅ GOOD MATCH (minor numerical diff)")
                else:
                    print(f"   Status:       ⚠️ SIGNIFICANT DIFFERENCE")
                    print(f"   PyTorch sample: {pytorch_grad.flat[:5]}")
                    print(f"   ONNX sample:    {onnx_grad.flat[:5]}")
        else:
            print(f"   Status:       ❌ NOT FOUND IN ONNX")

    print("\n" + "=" * 70)
    print("✅ Comparison Complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX gradients")
    parser.add_argument("--model_py", required=True, help="Path to model Python file")
    parser.add_argument("--model_class", required=True, help="Model class name")
    parser.add_argument("--model_kwargs_json", required=True, help="Model kwargs as JSON")
    parser.add_argument("--onnx_dir", required=True, help="ONNX directory containing models and inputs")
    parser.add_argument("--node", required=True, help="Node name to compute gradients for")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (not used, for reference)")

    args = parser.parse_args()

    # Parse paths - handle both absolute and relative paths
    # If relative path, resolve from /app/Onnx4Deeploy (not from cwd)
    if os.path.isabs(args.onnx_dir):
        onnx_dir = args.onnx_dir
    elif os.path.exists(args.onnx_dir):
        # Relative path exists from current directory
        onnx_dir = os.path.abspath(args.onnx_dir)
    else:
        # Try from /app/Onnx4Deeploy base directory
        base_dir = "/app/Onnx4Deeploy"
        onnx_dir = os.path.join(base_dir, args.onnx_dir)
        if not os.path.exists(onnx_dir):
            raise FileNotFoundError(
                f"ONNX directory not found. Tried:\n"
                f"  1. {os.path.abspath(args.onnx_dir)}\n"
                f"  2. {onnx_dir}"
            )

    onnx_dir = os.path.abspath(onnx_dir)
    onnx_infer = os.path.join(onnx_dir, "network_infer.onnx")
    onnx_train = os.path.join(onnx_dir, "network_train.onnx")
    inputs_npz = os.path.join(onnx_dir, "inputs.npz")
    inputs_with_weights_npz = os.path.join(onnx_dir, "inputs_with_weights.npz")
    pytorch_grad_npz = os.path.join(onnx_dir, "pytorch_gradients.npz")
    onnx_grad_npz = os.path.join(onnx_dir, "onnx_gradients.npz")

    # Parse model kwargs
    model_kwargs = json.loads(args.model_kwargs_json)

    print("=" * 70)
    print("🚀 PyTorch vs ONNX Gradient Comparison")
    print("=" * 70)
    print(f"Model:     {args.model_class}")
    print(f"Node:      {args.node}")
    print(f"ONNX dir:  {onnx_dir}")

    # Step 1: Extract weights
    if not os.path.exists(inputs_with_weights_npz):
        extract_weights_to_npz(onnx_infer, inputs_npz, inputs_with_weights_npz)
    else:
        print(f"\n✓ inputs_with_weights.npz already exists, skipping extraction")

    # Step 2: Run PyTorch
    pytorch_results = run_pytorch_gradients(
        args.model_py, args.model_class, model_kwargs,
        inputs_with_weights_npz, args.node, pytorch_grad_npz
    )

    # Step 3: Run ONNX
    onnx_results = run_onnx_gradients(onnx_train, inputs_npz, onnx_grad_npz)

    # Step 4: Compare
    compare_gradients(pytorch_results, onnx_results, args.node)


if __name__ == "__main__":
    main()
