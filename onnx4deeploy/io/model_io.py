# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
ONNX model I/O and comparison utilities.

This module provides functions for loading, saving, and comparing ONNX models.
"""

from typing import Dict

import onnx


def compare_onnx_models(
    model1_path: str, model2_path: str, name1: str = "Model 1", name2: str = "Model 2"
) -> Dict[str, any]:
    """
    Compare two ONNX models and report differences in nodes, inputs, outputs.

    Args:
        model1_path: Path to the first ONNX model
        model2_path: Path to the second ONNX model
        name1: Display name for the first model
        name2: Display name for the second model

    Returns:
        A dictionary containing:
        - node_diff: Difference in node count
        - missing_outputs: List of outputs present in model1 but not in model2
        - added_outputs: List of outputs present in model2 but not in model1
        - types_diff: Dictionary of node type count differences
    """
    model1 = onnx.load(model1_path)
    model2 = onnx.load(model2_path)

    graph1 = model1.graph
    graph2 = model2.graph

    print(f"\n{'='*60}")
    print(f"Comparing {name1} vs {name2}")
    print(f"{'='*60}")

    # Compare node counts
    print("\n📊 Node Statistics:")
    print(f"  {name1}: {len(graph1.node)} nodes")
    print(f"  {name2}: {len(graph2.node)} nodes")
    print(f"  Difference: {len(graph2.node) - len(graph1.node):+d} nodes")

    # Compare node types
    types1 = {}
    types2 = {}
    for node in graph1.node:
        types1[node.op_type] = types1.get(node.op_type, 0) + 1
    for node in graph2.node:
        types2[node.op_type] = types2.get(node.op_type, 0) + 1

    all_types = set(types1.keys()) | set(types2.keys())

    print("\n📋 Node Type Changes:")
    for op_type in sorted(all_types):
        count1 = types1.get(op_type, 0)
        count2 = types2.get(op_type, 0)
        if count1 != count2:
            print(f"  {op_type}: {count1} → {count2} ({count2 - count1:+d})")

    # Compare graph outputs
    print("\n🎯 Graph Outputs:")
    print(f"  {name1}: {len(graph1.output)} outputs")
    outputs1_names = [out.name for out in graph1.output]
    for i, out in enumerate(outputs1_names[:10]):  # Show first 10
        print(f"    - {out}")
    if len(outputs1_names) > 10:
        print(f"    ... and {len(outputs1_names) - 10} more")

    print(f"  {name2}: {len(graph2.output)} outputs")
    outputs2_names = [out.name for out in graph2.output]
    for i, out in enumerate(outputs2_names[:10]):  # Show first 10
        print(f"    - {out}")
    if len(outputs2_names) > 10:
        print(f"    ... and {len(outputs2_names) - 10} more")

    # Check for missing outputs
    outputs1 = set(outputs1_names)
    outputs2 = set(outputs2_names)

    missing = outputs1 - outputs2
    added = outputs2 - outputs1

    if missing:
        print(f"\n⚠️  Missing outputs in {name2}: {len(missing)} outputs lost")
        for out in list(missing)[:20]:  # Show first 20
            print(f"    - {out}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")

    if added:
        print(f"\n✨ New outputs in {name2}: {len(added)} outputs added")
        for out in list(added)[:20]:  # Show first 20
            print(f"    - {out}")
        if len(added) > 20:
            print(f"    ... and {len(added) - 20} more")

    print(f"{'='*60}\n")

    return {
        "node_diff": len(graph2.node) - len(graph1.node),
        "missing_outputs": list(missing),
        "added_outputs": list(added),
        "types_diff": {
            k: (types2.get(k, 0) - types1.get(k, 0))
            for k in all_types
            if types1.get(k, 0) != types2.get(k, 0)
        },
    }
