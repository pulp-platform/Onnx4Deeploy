#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
List all modules in a PyTorch model to find the correct node name for grad_at_node.py

Usage:
    python list_model_modules.py \
        --model_py path/to/model.py \
        --model_class ModelClass \
        --model_kwargs_json '{"param":"value"}'

Example:
    python /app/Onnx4Deeploy/utils/list_model_modules.py \
        --model_py /app/Onnx4Deeploy/Tests/Models/MI-BMInet/mi_bminet_model/mi_bminet.py \
        --model_class MIBMINetDeploy \
        --model_kwargs_json '{"C":8,"T":2000,"N":4}'
"""

import argparse
import importlib.util
import json
import os
import sys


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


def main():
    parser = argparse.ArgumentParser(description="List all modules in a PyTorch model")
    parser.add_argument("--model_py", required=True, help="Path to model Python file")
    parser.add_argument("--model_class", required=True, help="Model class name")
    parser.add_argument("--model_kwargs_json", default="{}", help="Model kwargs as JSON")

    args = parser.parse_args()

    # Import model
    module = import_from_path(args.model_py)
    if not hasattr(module, args.model_class):
        raise AttributeError(f"Class '{args.model_class}' not found in {args.model_py}")

    ModelCls = getattr(module, args.model_class)
    model_kwargs = json.loads(args.model_kwargs_json)
    model = ModelCls(**model_kwargs)

    print("=" * 80)
    print(f"🔍 Modules in {args.model_class}")
    print("=" * 80)
    print(f"\nTotal modules: {sum(1 for _ in model.named_modules())}\n")

    # List all modules with their types
    print("Module Name".ljust(40) + "Module Type")
    print("-" * 80)

    for name, mod in model.named_modules():
        if name == "":
            name = "<root>"
        module_type = type(mod).__name__

        # Highlight common layers
        emoji = ""
        if "Conv" in module_type:
            emoji = "🔲"
        elif "Linear" in module_type or "FC" in name.upper():
            emoji = "📊"
        elif "Pool" in module_type:
            emoji = "🏊"
        elif "BatchNorm" in module_type or "LayerNorm" in module_type:
            emoji = "📏"
        elif "Dropout" in module_type:
            emoji = "💧"
        elif "ReLU" in module_type or "Activation" in module_type:
            emoji = "⚡"

        print(f"{emoji} {name.ljust(38)} {module_type}")

    print("\n" + "=" * 80)
    print("\n💡 Usage:")
    print(f"   Use any module name above with grad_at_node.py --node <name>")
    print(f"\n   Example:")
    print(f"   python grad_at_node.py --node pool2 ...")
    print("=" * 80)


if __name__ == "__main__":
    main()
