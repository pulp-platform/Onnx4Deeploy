"""
Node naming utilities for ONNX models.

This module provides functions for renaming and cleaning node names in ONNX models
to ensure compatibility with C variable naming conventions.
"""

import re
from typing import Dict, Optional

import onnx


def make_c_name(name: str, count: int = 0) -> str:
    """
    Convert a name to a valid C variable name.

    Args:
        name: The original name to convert
        count: A counter to use for generating unique names

    Returns:
        A valid C variable name
    """
    if name.lower() in ["input", "output"]:
        return name  # Keep 'input' and 'output' as is

    name = re.sub(
        r"input|output", "", name, flags=re.IGNORECASE
    )  # Remove 'input' and 'output' from other names
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if name is None or name == "":
        name = f"node_{count}"
    if name[0].isdigit() or name[0] == "_":
        name = f"node_{count}" + name
    return name


def rename_onnx_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Rename all nodes in an ONNX model to valid C variable names.

    Args:
        model: The ONNX model to rename

    Returns:
        The model with renamed nodes
    """
    i_node = 0
    for node in model.graph.node:
        i_node += 1
        node.name = make_c_name(node.name, i_node)
        for i, input_name in enumerate(node.input):
            node.input[i] = make_c_name(input_name)
        for i, output_name in enumerate(node.output):
            node.output[i] = make_c_name(output_name)

    for input in model.graph.input:
        input.name = make_c_name(input.name)
    for output in model.graph.output:
        output.name = make_c_name(output.name)

    for init in model.graph.initializer:
        init.name = make_c_name(init.name)

    return model


def rename_and_save_onnx(input_onnx: str, output_onnx: str) -> None:
    """
    Load an ONNX model, rename its nodes, and save it.

    Args:
        input_onnx: Path to the input ONNX model
        output_onnx: Path to save the renamed model
    """
    model = onnx.load(input_onnx)
    model = rename_onnx_nodes(model)
    onnx.save(model, output_onnx)
    print(f"✅ Renamed ONNX model saved to {output_onnx}")


def rename_nodes(model_path: str, output_path: str) -> Dict[str, str]:
    """
    Rename nodes in an ONNX model by replacing all characters that are invalid
    for C variable names with underscores.

    Args:
        model_path: Path to the input ONNX model
        output_path: Path to save the renamed model

    Returns:
        A dictionary mapping original names to new names
    """
    # Load the model
    model = onnx.load(model_path)

    # Create a map to store original to new name mappings
    name_map = {}

    # Helper function to replace invalid C variable name characters with underscores
    def clean_name(name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        # Replace any character that's not alphanumeric or underscore with underscore
        # Ensure name starts with a letter or underscore (C variable rule)
        cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        if cleaned and cleaned[0].isdigit():
            cleaned = "_" + cleaned
        return cleaned

    # Process graph inputs
    for input in model.graph.input:
        if input.name:
            new_name = clean_name(input.name)
            name_map[input.name] = new_name
            input.name = new_name

    # Process graph outputs
    for output in model.graph.output:
        if output.name:
            new_name = clean_name(output.name)
            name_map[output.name] = new_name
            output.name = new_name

    # Process initializers
    for initializer in model.graph.initializer:
        if initializer.name:
            new_name = clean_name(initializer.name)
            name_map[initializer.name] = new_name
            initializer.name = new_name

    # Process nodes
    for node in model.graph.node:
        # Rename node name if it exists
        if node.name:
            node.name = clean_name(node.name)

        # Rename node inputs
        for i, input_name in enumerate(node.input):
            if input_name in name_map:
                node.input[i] = name_map[input_name]
            else:
                new_name = clean_name(input_name)
                if new_name != input_name:
                    name_map[input_name] = new_name
                    node.input[i] = new_name

        # Rename node outputs
        for i, output_name in enumerate(node.output):
            if output_name in name_map:
                node.output[i] = name_map[output_name]
            else:
                new_name = clean_name(output_name)
                if new_name != output_name:
                    name_map[output_name] = new_name
                    node.output[i] = new_name

        # Rename attribute names if they contain node names
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                # Handle subgraphs if present (recursive call would be needed)
                pass
            elif attribute.type == onnx.AttributeProto.STRINGS:
                # Handle string attributes that might contain node names
                for i, s in enumerate(attribute.strings):
                    s_str = s.decode("utf-8") if isinstance(s, bytes) else s
                    if s_str in name_map:
                        attribute.strings[i] = name_map[s_str].encode("utf-8")

    # Check for any value_info that might need renaming
    for value_info in model.graph.value_info:
        if value_info.name:
            new_name = clean_name(value_info.name)
            name_map[value_info.name] = new_name
            value_info.name = new_name

    # Save the updated model
    onnx.save(model, output_path)

    return name_map
