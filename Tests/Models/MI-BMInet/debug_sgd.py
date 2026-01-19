#!/usr/bin/env python3
"""
Debug script to see why only 3 SGD nodes are created instead of 5.
"""

import onnx
from onnx import numpy_helper
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "onnx/MI_BMINet_train_C8_T2000_F1_8_D2_N4")
network_pre_sgd = os.path.join(model_dir, "network_pre_sgd.onnx")

if not os.path.exists(network_pre_sgd):
    print(f"❌ Model not found: {network_pre_sgd}")
    sys.exit(1)

print(f"Loading model: {network_pre_sgd}")
model = onnx.load(network_pre_sgd)
graph = model.graph

# Get all output names
output_names = [output.name for output in graph.output]
print(f"\n📊 Graph outputs ({len(output_names)}):")
for name in output_names:
    print(f"  - {name}")

# Get all initializer names (parameters)
param_names = [init.name for init in graph.initializer]
param_shapes = {init.name: numpy_helper.to_array(init).shape for init in graph.initializer}

print(f"\n📦 Initializers/Parameters ({len(param_names)}):")
for name in sorted(param_names):
    print(f"  - {name}: shape={param_shapes[name]}")

# Simulate the NEW FIXED matching logic from add_sgd_nodes
print(f"\n🔍 Matching gradients to parameters (FIXED LOGIC):")
grad_to_param_map = {}

for output_name in output_names:
    if "grad" in output_name.lower():
        print(f"\n  Checking gradient output: {output_name}")

        # Regular case: param_name_grad -> param_name
        # Remove _grad suffix and match exactly
        if output_name.lower().endswith("_grad"):
            param_name_candidate = output_name[:-5]  # Remove "_grad"
            print(f"    Candidate parameter name: {param_name_candidate}")

            # Try exact match first
            if param_name_candidate in param_names:
                grad_to_param_map[param_name_candidate] = output_name
                print(f"    ✅ Matched to parameter: {param_name_candidate}")
                continue

            # Try case-insensitive exact match
            matched = False
            for param_name in param_names:
                if param_name.lower() == param_name_candidate.lower():
                    grad_to_param_map[param_name] = output_name
                    print(f"    ✅ Matched to parameter: {param_name}")
                    matched = True
                    break

            if not matched:
                print(f"    ❌ No matching parameter found!")

print(f"\n📋 Final gradient-to-parameter mapping ({len(grad_to_param_map)}):")
for param, grad in grad_to_param_map.items():
    print(f"  {param} <- {grad}")

print(f"\n⚠️  Expected 5 mappings, got {len(grad_to_param_map)}")
if len(grad_to_param_map) < 5:
    print(f"❌ PROBLEM: Some gradients were not matched to parameters!")
