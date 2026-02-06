"""
ONNX model transformation utilities.

This module provides functions for transforming ONNX model structures,
including node splitting, parameter randomization, and shape inference.
"""

import copy
import random
from typing import List, Optional

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper


def split_convgrad_nodes(input_onnx: str, output_onnx: str) -> None:
    """
    Split ConvGrad nodes with 3 outputs into 3 separate nodes.

    Each node only receives the inputs it actually needs:
    - ConvGradX: [dY, W] - computes input gradient
    - ConvGradW: [dY, X] - computes weight gradient
    - ConvGradB: [dY] - computes bias gradient

    Args:
        input_onnx: Path to the input ONNX model
        output_onnx: Path to save the transformed model
    """
    model = onnx.load(input_onnx)
    graph = model.graph

    # Store original graph outputs to preserve them
    original_graph_outputs = set(out.name for out in graph.output)

    nodes_to_add = []
    nodes_to_remove = []

    # Track all output tensors that should be preserved
    preserved_outputs = {}

    for node in graph.node:
        if node.op_type == "ConvGrad":
            # Get inputs: dY, X, W
            if len(node.input) < 3:
                continue

            dY = node.input[0]  # Output gradient
            X = node.input[1]  # Forward input
            W = node.input[2]  # Weight

            # Get outputs - keep original positions to avoid index misalignment
            # In ONNX, empty strings indicate unused outputs
            # node.output[0] = dX (gradient w.r.t. input)
            # node.output[1] = dW (gradient w.r.t. weight)
            # node.output[2] = dB (gradient w.r.t. bias, optional)

            # Check if we have enough outputs
            if len(node.output) < 2:
                continue

            # Copy attributes from original node
            attrs = {attr.name: attr for attr in node.attribute}

            # Create ConvGradX node if dX output exists (index 0)
            if len(node.output) > 0 and node.output[0] != "":
                dX_output = node.output[0]
                convgrad_x = helper.make_node(
                    "ConvGradX",
                    inputs=[dY, W],  # ConvTranspose-like operation
                    outputs=[dX_output],  # dX
                    name=node.name + "_X",
                )
                # Copy attributes
                for _attr_name, attr in attrs.items():
                    convgrad_x.attribute.append(copy.deepcopy(attr))
                nodes_to_add.append(convgrad_x)

                # Track if this output is a graph output
                if dX_output in original_graph_outputs:
                    preserved_outputs[dX_output] = True

            # Create ConvGradW node if dW output exists (index 1)
            if len(node.output) > 1 and node.output[1] != "":
                dW_output = node.output[1]
                convgrad_w = helper.make_node(
                    "ConvGradW",
                    inputs=[dY, X],  # Correlation operation
                    outputs=[dW_output],  # dW
                    name=node.name + "_W",
                )
                # Copy attributes
                for _attr_name, attr in attrs.items():
                    convgrad_w.attribute.append(copy.deepcopy(attr))
                nodes_to_add.append(convgrad_w)

                # Track if this output is a graph output
                if dW_output in original_graph_outputs:
                    preserved_outputs[dW_output] = True

            # Create ConvGradB node if dB output exists (index 2)
            if len(node.output) > 2 and node.output[2] != "":
                dB_output = node.output[2]
                convgrad_b = helper.make_node(
                    "ConvGradB",
                    inputs=[dY],  # Sum over spatial dimensions
                    outputs=[dB_output],  # dB
                    name=node.name + "_B",
                )
                # Copy attributes (though ConvGradB may not need all of them)
                for attr_name, attr in attrs.items():
                    convgrad_b.attribute.append(copy.deepcopy(attr))
                nodes_to_add.append(convgrad_b)

                # Track if this output is a graph output
                if dB_output in original_graph_outputs:
                    preserved_outputs[dB_output] = True

            # Mark original node for removal
            nodes_to_remove.append(node)

    # Remove original ConvGrad nodes
    for node in nodes_to_remove:
        graph.node.remove(node)

    # Add new split nodes
    graph.node.extend(nodes_to_add)

    # Save the modified model
    onnx.save(model, output_onnx)

    # Verify no graph outputs were lost
    new_model = onnx.load(output_onnx)
    new_graph_outputs = set(out.name for out in new_model.graph.output)
    lost_outputs = original_graph_outputs - new_graph_outputs

    if lost_outputs:
        print(f"⚠️  WARNING: Lost {len(lost_outputs)} graph outputs after ConvGrad split:")
        for out in list(lost_outputs)[:10]:
            print(f"    - {out}")
        if len(lost_outputs) > 10:
            print(f"    ... and {len(lost_outputs) - 10} more")
    else:
        print(f"✅ All {len(original_graph_outputs)} graph outputs preserved after ConvGrad split")

    print(
        f"✅ Split {len(nodes_to_remove)} ConvGrad node(s) into {len(nodes_to_add)} separate gradient nodes"
    )
    if preserved_outputs:
        print(f"   {len(preserved_outputs)} ConvGrad outputs are graph outputs")


def randomize_layernorm_params(model: torch.nn.Module) -> torch.nn.Module:
    """
    Randomize LayerNorm parameters by adding small random noise.

    Args:
        model: PyTorch model containing LayerNorm layers

    Returns:
        The model with randomized LayerNorm parameters
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            with torch.no_grad():
                module.weight.data = (
                    module.weight.data + torch.randn_like(module.weight.data) * 1e-6
                )
                module.bias.data = module.bias.data + torch.randn_like(module.bias.data) * 1e-6

    return model


def fix_shared_initializers_by_node_name(model_path: str, output_path: str) -> bool:
    """
    Fix shared initializers by adding node names as suffixes to ensure uniqueness.

    Add node name suffix to each bias to ensure uniqueness.

    Args:
        model_path: Path to the input ONNX model
        output_path: Path to save the fixed model

    Returns:
        True if shared initializers were found and fixed, False otherwise
    """
    model = onnx.load(model_path)

    print("Fixing shared initializers by adding node names...")

    # Analyze which nodes use each initializer
    initializer_node_mapping = {}  # init_name -> [node_names]

    for init in model.graph.initializer:
        init_name = init.name
        using_nodes = []

        # Find nodes that use this initializer
        for node in model.graph.node:
            if init_name in node.input:
                using_nodes.append(node.name)

        if len(using_nodes) > 1:
            # Record if used by multiple nodes
            initializer_node_mapping[init_name] = using_nodes
            print(f"Initializer '{init_name}' is used by {len(using_nodes)} nodes:")
            for node_name in using_nodes:
                print(f"  - {node_name}")

    if not initializer_node_mapping:
        print("No shared initializers detected. Nothing to fix.")
        return False

    # Create independent copy for each use of shared initializer
    new_initializers = []
    processed_inits = set()

    for init in model.graph.initializer:
        if init.name in initializer_node_mapping:
            # This is an initializer shared by multiple nodes
            print(f"\nProcessing shared initializer: {init.name}")
            processed_inits.add(init.name)

            # Create independent copy for each using node
            for node_name in initializer_node_mapping[init.name]:
                # Create new initializer name: original_name_node_name
                new_name = f"{init.name}_{node_name}"

                # Create copy of initializer
                import copy

                new_init = copy.deepcopy(init)
                new_init.name = new_name
                new_initializers.append(new_init)
                print(f"  Created copy for node '{node_name}': {init.name} -> {new_name}")
        else:
            # Regular initializer, keep as is
            new_initializers.append(init)

    # Update model's initializer list
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_initializers)

    # Update references in nodes
    print("\nUpdating node references...")
    for node in model.graph.node:
        for j, input_name in enumerate(node.input):
            if input_name in processed_inits:
                # This node uses shared initializer, update to dedicated version
                old_name = input_name
                new_name = f"{input_name}_{node.name}"
                node.input[j] = new_name
                print(f"  Node '{node.name}': {old_name} -> {new_name}")

    # Save修复afterofModel
    onnx.save(model, output_path)
    print(f"\nFixed model saved to: {output_path}")

    return True


def randomize_onnx_initializers(
    model: onnx.ModelProto, seed: Optional[int] = None, exclude_patterns: Optional[List[str]] = None
) -> onnx.ModelProto:
    """
    Randomize ONNX model initializers (weights and biases).

    Args:
        model: The ONNX model to randomize
        seed: Random seed for reproducibility
        exclude_patterns: List of patterns to exclude from randomization

    Returns:
        The model with randomized initializers
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if exclude_patterns is None:
        exclude_patterns = ["const", "shape", "Constant"]

    graph = model.graph

    modified_count = 0
    zero_count = 0
    skipped_count = 0

    for initializer in graph.initializer:
        # Skip initializers with specific patterns in their names
        if any(pattern in initializer.name for pattern in exclude_patterns):
            skipped_count += 1
            continue

        # Convert initializer to numpy array
        np_array = numpy_helper.to_array(initializer)

        # Check if array contains only zeros
        if np.all(np_array == 0):
            zero_count += 1

            # Determine appropriate scale based on tensor dimension and type
            if np_array.dtype == np.float32 or np_array.dtype == np.float64:
                # Use Kaiming/He initialization for weights
                if len(np_array.shape) > 1:
                    fan_in = np_array.shape[0]
                    scale = np.sqrt(2.0 / fan_in)
                    np_array = np.random.normal(0, scale, np_array.shape).astype(np_array.dtype)
                else:
                    # For bias terms or 1D tensors
                    np_array = np.random.uniform(-0.1, 0.1, np_array.shape).astype(np_array.dtype)
            elif np_array.dtype == np.int64 or np_array.dtype == np.int32:
                # For integer tensors (e.g., indices)
                max_val = min(100, 2 ** (np_array.itemsize * 8 - 1) - 1)  # Avoid overflow
                np_array = np.random.randint(-max_val, max_val, np_array.shape).astype(
                    np_array.dtype
                )

            # Create new tensor from modified numpy array
            new_tensor = numpy_helper.from_array(np_array, initializer.name)

            # Replace original initializer with new tensor
            initializer.CopyFrom(new_tensor)
            modified_count += 1

    print("Randomization complete:")
    print(f"- Total initializers: {len(graph.initializer)}")
    print(f"- Zero initializers found and randomized: {zero_count}")
    print(f"- Skipped initializers (based on patterns): {skipped_count}")
    print(f"- Modified initializers: {modified_count}")

    return model


def type_inference(input_model_path: str, output_model_path: str) -> None:
    """
    Perform type inference on ONNX model, setting float32 type for variables without explicit types.

    Args:
        input_model_path: Input ONNX model path
        output_model_path: Output ONNX model path
    """
    # Load the ONNX model
    model = onnx.load(input_model_path)
    graph = model.graph

    # Process input tensors
    for input_tensor in graph.input:
        if not input_tensor.type.tensor_type.elem_type:
            print(f"Setting input variable {input_tensor.name} type to FLOAT")
            input_tensor.type.tensor_type.elem_type = TensorProto.FLOAT

    # Process output tensors
    for output_tensor in graph.output:
        if not output_tensor.type.tensor_type.elem_type:
            print(f"Setting output variable {output_tensor.name} type to FLOAT")
            output_tensor.type.tensor_type.elem_type = TensorProto.FLOAT

    # Process intermediate variables
    for value_info in graph.value_info:
        if not value_info.type.tensor_type.elem_type:
            print(f"Setting intermediate variable {value_info.name} type to FLOAT")
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT

    # Check for any tensors mentioned in nodes but missing type info
    processed_tensors = set(
        [tensor.name for tensor in graph.input]
        + [tensor.name for tensor in graph.output]
        + [tensor.name for tensor in graph.value_info]
    )

    missing_tensors = set()
    for node in graph.node:
        for input_name in node.input:
            if input_name not in processed_tensors and input_name:
                missing_tensors.add(input_name)
        for output_name in node.output:
            if output_name not in processed_tensors and output_name:
                missing_tensors.add(output_name)

    # Add missing tensors to value_info with FLOAT type (keeping existing shapes)
    for tensor_name in missing_tensors:
        # Create a basic ValueInfo for the tensor (shape will be inferred by ONNX)
        tensor_value_info = onnx.helper.make_tensor_value_info(
            name=tensor_name, elem_type=TensorProto.FLOAT, shape=None  # Let ONNX infer the shape
        )
        graph.value_info.append(tensor_value_info)
        print(f"Added missing tensor {tensor_name} with FLOAT type")

    # Save the modified model
    onnx.save(model, output_model_path)
    print(f"Successfully saved type-inferred model to {output_model_path}")


def ensure_all_tensor_shapes(model_path: str, output_path: str) -> onnx.ModelProto:
    """
    Ensure all tensors in the ONNX model have shape annotations.
    This function infers shapes for missing tensors, especially gradient intermediates.

    Args:
        model_path: Path to the input ONNX model
        output_path: Path to save the model with complete shapes

    Returns:
        The ONNX model with all tensor shapes inferred
    """
    print("🔍 Checking and fixing tensor shapes...")

    # Load the model
    onnx_model = onnx.load(model_path)
    graph = onnx_model.graph

    # First, run standard shape inference
    try:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    except Exception as e:
        print(f"⚠️  Standard shape inference warning: {e}")

    # Collect all tensor shapes from value_info
    known_shapes = {}

    # Get shapes from inputs
    for input_tensor in graph.input:
        if input_tensor.type.tensor_type.HasField("shape"):
            shape = [
                dim.dim_value if dim.HasField("dim_value") else -1
                for dim in input_tensor.type.tensor_type.shape.dim
            ]
            known_shapes[input_tensor.name] = shape

    # Get shapes from outputs
    for output_tensor in graph.output:
        if output_tensor.type.tensor_type.HasField("shape"):
            shape = [
                dim.dim_value if dim.HasField("dim_value") else -1
                for dim in output_tensor.type.tensor_type.shape.dim
            ]
            known_shapes[output_tensor.name] = shape

    # Get shapes from value_info
    for value_info in graph.value_info:
        if value_info.type.tensor_type.HasField("shape"):
            shape = [
                dim.dim_value if dim.HasField("dim_value") else -1
                for dim in value_info.type.tensor_type.shape.dim
            ]
            known_shapes[value_info.name] = shape

    # Get shapes from initializers
    for init in graph.initializer:
        known_shapes[init.name] = list(init.dims)

    # Find all tensor names used in the graph
    all_tensor_names = set()
    for node in graph.node:
        all_tensor_names.update(node.input)
        all_tensor_names.update(node.output)

    # Find missing tensors
    missing_tensors = all_tensor_names - set(known_shapes.keys())

    if missing_tensors:
        print(f"⚠️  Found {len(missing_tensors)} tensors with missing shapes")
        print(f"Missing tensors: {list(missing_tensors)[:10]}...")  # Show first 10

        # Try to infer shapes from connected nodes
        for tensor_name in missing_tensors:
            # Find nodes that produce this tensor
            producer_nodes = [n for n in graph.node if tensor_name in n.output]
            consumer_nodes = [n for n in graph.node if tensor_name in n.input]

            inferred_shape = None

            # Try to infer from producer
            if producer_nodes:
                producer = producer_nodes[0]

                # For gradient accumulation nodes, use input shape
                if "AccumulateGrad" in producer.op_type or "Gradient" in producer.op_type:
                    for input_name in producer.input:
                        if input_name in known_shapes:
                            inferred_shape = known_shapes[input_name]
                            break

                # For element-wise ops, inherit from input
                elif producer.op_type in ["Add", "Sub", "Mul", "Div", "Relu", "Identity"]:
                    for input_name in producer.input:
                        if input_name in known_shapes:
                            inferred_shape = known_shapes[input_name]
                            break

            # Try to infer from consumer
            if inferred_shape is None and consumer_nodes:
                consumer = consumer_nodes[0]
                for input_name in consumer.input:
                    if input_name != tensor_name and input_name in known_shapes:
                        inferred_shape = known_shapes[input_name]
                        break

            # If we found a shape, add it to value_info
            if inferred_shape is not None:
                # Create value_info with inferred shape
                tensor_type = onnx.TensorProto.FLOAT  # Default to FLOAT
                value_info = helper.make_tensor_value_info(tensor_name, tensor_type, inferred_shape)
                graph.value_info.append(value_info)
                known_shapes[tensor_name] = inferred_shape
                print(f"✅ Inferred shape for {tensor_name}: {inferred_shape}")

    # Final shape inference pass
    try:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        print("✅ Final shape inference completed")
    except Exception as e:
        print(f"⚠️  Final shape inference warning: {e}")

    # Verify all tensors now have shapes
    remaining_missing = []
    for value_info in graph.value_info:
        if not value_info.type.tensor_type.HasField("shape"):
            remaining_missing.append(value_info.name)

    if remaining_missing:
        print(f"⚠️  Warning: {len(remaining_missing)} tensors still missing shapes")
        print(f"Remaining missing: {remaining_missing[:10]}")
    else:
        print("✅ All tensors have shape annotations")

    # Save the model
    onnx.save(onnx_model, output_path)
    print(f"✅ Saved model with complete shapes to {output_path}")

    return onnx_model
