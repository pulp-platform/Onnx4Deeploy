#!/usr/bin/env python3
"""
Diagnostic script to compare network_train.onnx and network.onnx
and identify any missing nodes or outputs.
"""

import os
import sys

# Add utils to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from utils.utils import compare_onnx_models


def main():
    # Check for existing ONNX files
    base_dir = script_dir

    # Find the latest generated model directory
    onnx_dir = os.path.join(base_dir, "onnx")

    if not os.path.exists(onnx_dir):
        print(f"❌ No onnx directory found at {onnx_dir}")
        print("Please run testtraingenerate.py first to generate the models.")
        return

    # List all subdirectories
    subdirs = [d for d in os.listdir(onnx_dir) if os.path.isdir(os.path.join(onnx_dir, d))]

    if not subdirs:
        print(f"❌ No model directories found in {onnx_dir}")
        print("Please run testtraingenerate.py first to generate the models.")
        return

    # Use the first (or only) subdirectory
    model_dir = os.path.join(onnx_dir, subdirs[0])
    print(f"📂 Using model directory: {model_dir}")

    network_train = os.path.join(model_dir, "network_train.onnx")
    network = os.path.join(model_dir, "network.onnx")
    network_pre_sgd = os.path.join(model_dir, "network_pre_sgd.onnx")

    # Check which files exist
    files_to_check = [
        ("network_train.onnx", network_train),
        ("network.onnx", network),
        ("network_pre_sgd.onnx", network_pre_sgd)
    ]

    existing_files = {}
    for name, path in files_to_check:
        if os.path.exists(path):
            existing_files[name] = path
            print(f"✅ Found {name}")
        else:
            print(f"❌ Missing {name}")

    if len(existing_files) < 2:
        print("\n❌ Need at least 2 model files to compare")
        return

    print("\n" + "="*70)
    print("DIAGNOSTIC ANALYSIS")
    print("="*70)

    # Compare network_train.onnx vs network.onnx
    if "network_train.onnx" in existing_files and "network.onnx" in existing_files:
        print("\n🔍 Comparison 1: network_train.onnx → network.onnx")
        result = compare_onnx_models(
            existing_files["network_train.onnx"],
            existing_files["network.onnx"],
            name1="network_train.onnx",
            name2="network.onnx"
        )

        if result['missing_outputs']:
            print(f"\n⚠️  ISSUE DETECTED: {len(result['missing_outputs'])} outputs missing!")
            print("This suggests the optimization pipeline is removing necessary outputs.")
        else:
            print("\n✅ No output loss detected in this comparison")

    # Compare network_train.onnx vs network_pre_sgd.onnx
    if "network_train.onnx" in existing_files and "network_pre_sgd.onnx" in existing_files:
        print("\n🔍 Comparison 2: network_train.onnx → network_pre_sgd.onnx (before SGD)")
        result = compare_onnx_models(
            existing_files["network_train.onnx"],
            existing_files["network_pre_sgd.onnx"],
            name1="network_train.onnx",
            name2="network_pre_sgd.onnx"
        )

        if result['missing_outputs']:
            print(f"\n⚠️  ISSUE DETECTED: {len(result['missing_outputs'])} outputs missing!")
            print("This suggests the optimization pipeline (before SGD) is removing necessary outputs.")
        else:
            print("\n✅ No output loss detected in this comparison")

    # Compare network_pre_sgd.onnx vs network.onnx
    if "network_pre_sgd.onnx" in existing_files and "network.onnx" in existing_files:
        print("\n🔍 Comparison 3: network_pre_sgd.onnx → network.onnx (SGD addition)")
        result = compare_onnx_models(
            existing_files["network_pre_sgd.onnx"],
            existing_files["network.onnx"],
            name1="network_pre_sgd.onnx",
            name2="network.onnx"
        )

        if result['missing_outputs']:
            print(f"\n⚠️  ISSUE DETECTED: {len(result['missing_outputs'])} outputs missing!")
            print("This suggests the SGD addition step is removing necessary outputs.")
        else:
            print("\n✅ No output loss detected in this comparison")

    print("\n" + "="*70)
    print("END OF DIAGNOSTIC ANALYSIS")
    print("="*70)


if __name__ == "__main__":
    main()
