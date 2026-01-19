#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract weights from ONNX model and merge with inputs.npz

Usage:
    python extract_onnx_weights_to_npz.py \
        --onnx_model path/to/network_infer.onnx \
        --inputs_npz path/to/inputs.npz \
        --output_npz path/to/inputs_with_weights.npz

python /app/Onnx4Deeploy/utils/extract_onnx_weights_to_npz.py \
  --onnx_model Tests/Models/MI-BMInet/onnx/MI_BMINet_train_C8_T2000_F1_8_D2_N4_fc_sep_conv2/network_infer.onnx \
  --inputs_npz Tests/Models/MI-BMInet/onnx/MI_BMINet_train_C8_T2000_F1_8_D2_N4_fc_sep_conv2/inputs.npz \
  --output_npz Tests/Models/MI-BMInet/onnx/MI_BMINet_train_C8_T2000_F1_8_D2_N4_fc_sep_conv2/inputs_with_weights.npz

This script:
1. Loads an existing inputs.npz (containing input data and labels)
2. Extracts weights from ONNX model initializers
3. Maps ONNX weight names to PyTorch naming convention
4. Saves everything to a new npz file with all weights included
"""

import argparse
import os
import numpy as np
import onnx
from onnx import numpy_helper


def main():
    parser = argparse.ArgumentParser(description="Extract ONNX weights and merge with inputs.npz")
    parser.add_argument("--onnx_model", required=True, help="Path to ONNX model (e.g., network_infer.onnx)")
    parser.add_argument("--inputs_npz", required=True, help="Path to existing inputs.npz")
    parser.add_argument("--output_npz", required=True, help="Path to output npz file")

    args = parser.parse_args()

    # Check files exist
    if not os.path.exists(args.onnx_model):
        raise FileNotFoundError(f"ONNX model not found: {args.onnx_model}")
    if not os.path.exists(args.inputs_npz):
        raise FileNotFoundError(f"Inputs npz not found: {args.inputs_npz}")

    # Load existing inputs.npz
    print(f"📂 Loading {args.inputs_npz}...")
    data = np.load(args.inputs_npz)
    new_data = {key: data[key] for key in data.files}
    print(f"   Found keys: {list(data.keys())}")

    # Load ONNX model
    print(f"\n📂 Loading {args.onnx_model}...")
    model = onnx.load(args.onnx_model)

    # Extract weights from ONNX model
    print(f"\n🔍 Extracting weights from ONNX model...")
    weight_count = 0

    for init in model.graph.initializer:
        tensor = numpy_helper.to_array(init)

        # Convert ONNX naming to PyTorch naming
        # ONNX uses underscore: conv1_weight -> PyTorch uses dot: conv1.weight
        pytorch_name = init.name.replace("_weight", ".weight").replace("_bias", ".bias")

        # Handle special cases like sep_conv1_weight -> sep_conv1.weight
        # This assumes a simple pattern, may need adjustment for complex models
        parts = init.name.split("_")
        if len(parts) >= 2:
            # Find where 'weight' or 'bias' appears
            if parts[-1] in ["weight", "bias"]:
                param_type = parts[-1]
                module_name = "_".join(parts[:-1])
                pytorch_name = f"{module_name}.{param_type}"

        new_data[pytorch_name] = tensor
        weight_count += 1
        print(f"   ✓ {pytorch_name}: shape={tensor.shape}, dtype={tensor.dtype}")

    # Save merged npz
    print(f"\n💾 Saving to {args.output_npz}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)) or ".", exist_ok=True)
    np.savez(args.output_npz, **new_data)

    print(f"\n✅ Success!")
    print(f"   Total keys in output: {len(new_data)}")
    print(f"   - Input/label keys: {len(data.files)}")
    print(f"   - Weight keys added: {weight_count}")
    print(f"\n📋 All keys in output npz:")
    for key in sorted(new_data.keys()):
        print(f"   - {key}")


if __name__ == "__main__":
    main()
