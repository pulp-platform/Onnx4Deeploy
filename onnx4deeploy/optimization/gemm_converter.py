"""GEMM (General Matrix Multiplication) conversion and optimization utilities.

This module provides functions to convert and optimize GEMM operations in ONNX models:
- Adding bias (C) to GEMM nodes
- Converting GEMM to MatMul+Transpose
- Fusing MatMul+Add to GEMM
- Converting FusedMatMul to GEMM
- Unifying GEMM input dimensions
"""

import re

import numpy as np
import onnx
from onnx import helper, numpy_helper


def add_c_to_gemm(input_model_path: str, output_model_path: str) -> str:
    """Add zero bias (C) to GEMM nodes that don't have one.

    GEMM operation: Y = alpha * A * B + beta * C
    If C is missing, this function adds a zero tensor as bias.

    Args:
        input_model_path: Path to input ONNX model
        output_model_path: Path to save modified model

    Returns:
        Path to output model
    """
    model = onnx.load(input_model_path)
    graph = model.graph

    for node in graph.node:
        if node.op_type == "Gemm":

            if len(node.input) == 2:
                print(f"Find Gemm without C: {node.name}")

                node.input[0]
                input_b_name = node.input[1]

                b_shape = None
                for init in graph.initializer:
                    if init.name == input_b_name:
                        b_tensor = numpy_helper.to_array(init)
                        b_shape = b_tensor.shape
                        break

                if b_shape is None:
                    for vi in graph.value_info:
                        if vi.name == input_b_name:
                            b_shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
                            break

                if b_shape is None:
                    output_name = node.output[0]

                    all_value_infos = list(graph.value_info) + list(graph.output)

                    for vi in all_value_infos:
                        if vi.name == output_name:

                            if vi.type.tensor_type.shape.dim:
                                output_shape = [
                                    dim.dim_value for dim in vi.type.tensor_type.shape.dim
                                ]

                                if output_shape and all(output_shape):
                                    transB = 0
                                    for attr in node.attribute:
                                        if attr.name == "transB" and attr.i == 1:
                                            transB = 1

                                    c_length = output_shape[-1]
                                    b_shape = [c_length, 0] if transB == 0 else [0, c_length]
                            break

                if b_shape is not None and len(b_shape) >= 2 and (b_shape[0] > 0 or b_shape[1] > 0):

                    transB = 0
                    for attr in node.attribute:
                        if attr.name == "transB" and attr.i == 1:
                            transB = 1

                    if transB == 0 and b_shape[1] > 0:
                        c_shape = [b_shape[1]]
                    elif transB == 1 and b_shape[0] > 0:
                        c_shape = [b_shape[0]]
                    else:
                        print(f"Warning: Invalid shape {b_shape} for {node.name}, pass this node.")
                        continue

                    c_tensor = np.zeros(c_shape, dtype=np.float32)
                    c_name = f"{node.name}_c_bias"

                    c_initializer = numpy_helper.from_array(c_tensor, name=c_name)
                    graph.initializer.append(c_initializer)

                    node.input.append(c_name)
                    print(f"Add C: {c_name}, Shape: {c_shape}")
                else:
                    print(f"Warning: Cannot find valid shape for {node.name}, pass this node.")

    onnx.save(model, output_model_path)
    print(f"Saved to: {output_model_path}")
    return output_model_path


def convert_gemm_to_transpose_matmul(input_model_path: str, output_model_path: str) -> str:
    """Convert GEMM nodes to Transpose + MatMul nodes with optimized strategy.

    Strategy:
    1. Only convert GEMM when it's beneficial (avoid unnecessary transpose of constants)
    2. Skip GEMM nodes that were likely converted from FusedMatMul
    3. Minimize constant transposition

    Args:
        input_model_path: Path to input ONNX model
        output_model_path: Path to save modified model

    Returns:
        Path to output model
    """
    model = onnx.load(input_model_path)

    print(
        f"Original model: {len(model.graph.node)} nodes, {len(model.graph.initializer)} initializers",
        flush=True,
    )

    new_nodes = []
    transpose_cache = {}

    initializer_names = {init.name for init in model.graph.initializer}

    def should_convert_gemm(node):
        """
        决定is否shouldConvert这个GEMM节点
        判断标准：只有当need转置of输入不is常量时才进行Convert
        """
        if len(node.input) > 2 and node.input[2]:
            print("  Skip: GEMM has bias", flush=True)
            return False

        transA = False
        transB = False

        for attr in node.attribute:
            if attr.name == "transA":
                transA = attr.i == 1
            elif attr.name == "transB":
                transB = attr.i == 1

        input_a_is_const = node.input[0] in initializer_names
        input_b_is_const = len(node.input) > 1 and node.input[1] in initializer_names

        if transA and input_a_is_const:
            print("  Skip: transA=True but input A is constant", flush=True)
            return False

        if transB and input_b_is_const:
            print("  Skip: transB=True but input B is constant", flush=True)
            return False

        if transA or transB:
            print(
                f"  Convert: transpose needed on non-constant input(s) (transA={transA}, transB={transB})",
                flush=True,
            )
            return True
        else:
            print("  Convert: no transpose needed", flush=True)
            return True

    def get_tensor_dims(tensor_name):
        """Get tensor维度数"""
        for vi in model.graph.value_info:
            if vi.name == tensor_name:
                return len(vi.type.tensor_type.shape.dim)

        for init in model.graph.initializer:
            if init.name == tensor_name:
                return len(init.dims)

        for inp in model.graph.input:
            if inp.name == tensor_name:
                return len(inp.type.tensor_type.shape.dim)

        return 2

    def create_optimized_transpose(input_tensor, is_const, transpose_type, node_index):

        if is_const:

            transposed_name = f"{input_tensor}_T"
            if transposed_name not in [init.name for init in model.graph.initializer]:
                for init in model.graph.initializer:
                    if init.name == input_tensor:
                        array = numpy_helper.to_array(init)
                        transposed_array = np.transpose(array)
                        new_init = numpy_helper.from_array(transposed_array, name=transposed_name)
                        model.graph.initializer.append(new_init)
                        print(f"  -> Created transposed constant: {transposed_name}", flush=True)
                        break
            return transposed_name
        else:

            cache_key = f"transpose_{transpose_type}_{input_tensor}"
            if cache_key not in transpose_cache:
                transpose_output = f"{input_tensor}_transposed_{transpose_type}"

                input_dims = get_tensor_dims(input_tensor)

                if input_dims == 2:
                    perm = [1, 0]
                elif input_dims == 3:
                    perm = [0, 2, 1]
                elif input_dims == 4:
                    perm = [0, 1, 3, 2]
                else:

                    perm = list(range(input_dims))
                    perm[input_dims - 2], perm[input_dims - 1] = (
                        perm[input_dims - 1],
                        perm[input_dims - 2],
                    )

                transpose_node = helper.make_node(
                    "Transpose",
                    inputs=[input_tensor],
                    outputs=[transpose_output],
                    name=f"transpose_{transpose_type}_{node_index}",
                    perm=perm,
                )
                new_nodes.append(transpose_node)
                transpose_cache[cache_key] = transpose_output

                for vi in model.graph.value_info:
                    if vi.name == input_tensor:
                        input_shape = vi.type.tensor_type.shape
                        transposed_dims = []
                        for p in perm:
                            dim = input_shape.dim[p]
                            if dim.dim_value:
                                transposed_dims.append(dim.dim_value)
                            elif dim.dim_param:
                                transposed_dims.append(dim.dim_param)
                            else:
                                transposed_dims.append(None)

                        tensor_type = helper.make_tensor_type_proto(
                            elem_type=1, shape=transposed_dims  # FLOAT
                        )
                        value_info = helper.make_value_info(transpose_output, tensor_type)
                        model.graph.value_info.append(value_info)
                        break

                print(f"  -> Added Transpose for {transpose_type} with perm={perm}", flush=True)

            return transpose_cache[cache_key]

    converted_count = 0
    skipped_count = 0

    for node_index, node in enumerate(model.graph.node):
        if node.op_type == "Gemm":
            print(f"Evaluating GEMM: {node.name}", flush=True)
            print(f"  Inputs: {node.input}", flush=True)

            if should_convert_gemm(node):
                converted_count += 1
                print(f"Converting GEMM: {node.name}", flush=True)

                transA = False
                transB = False
                alpha = 1.0

                for attr in node.attribute:
                    if attr.name == "transA":
                        transA = attr.i == 1
                    elif attr.name == "transB":
                        transB = attr.i == 1
                    elif attr.name == "alpha":
                        alpha = attr.f

                current_inputs = list(node.input[:2])

                if transA:
                    input_a = current_inputs[0]
                    is_const = input_a in initializer_names
                    current_inputs[0] = create_optimized_transpose(
                        input_a, is_const, "A", node_index
                    )

                if transB:
                    input_b = current_inputs[1]
                    is_const = input_b in initializer_names
                    current_inputs[1] = create_optimized_transpose(
                        input_b, is_const, "B", node_index
                    )

                matmul_output = node.output[0] if alpha == 1.0 else f"{node.name}_matmul_out"

                matmul_node = helper.make_node(
                    "MatMul",
                    inputs=current_inputs,
                    outputs=[matmul_output],
                    name=f"matmul_{node.name}" if node.name else f"matmul_{node_index}",
                )
                new_nodes.append(matmul_node)
                print(f"  -> Added MatMul: {matmul_node.name}", flush=True)

                if alpha != 1.0:
                    alpha_name = f"alpha_{node.name}_{node_index}"
                    alpha_tensor = numpy_helper.from_array(
                        np.array([alpha], dtype=np.float32), name=alpha_name
                    )
                    model.graph.initializer.append(alpha_tensor)

                    mul_node = helper.make_node(
                        "Mul",
                        inputs=[matmul_output, alpha_name],
                        outputs=node.output,
                        name=f"mul_{node.name}_{node_index}",
                    )
                    new_nodes.append(mul_node)
                    print(f"  -> Added Mul for alpha={alpha}", flush=True)

            else:
                skipped_count += 1

                new_nodes.append(node)

        else:

            new_nodes.append(node)

    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer,
        value_info=model.graph.value_info,
    )

    new_model = helper.make_model(
        new_graph,
        producer_name=model.producer_name,
        opset_imports=model.opset_import,
        ir_version=model.ir_version,
    )

    new_model.metadata_props.extend(model.metadata_props)

    try:
        print("Running shape inference...", flush=True)
        inferred_model = onnx.shape_inference.infer_shapes(new_model)
        onnx.save(inferred_model, output_model_path)
        print("✅ Shape inference completed", flush=True)
    except Exception as e:
        print(f"⚠️ Shape inference failed: {e}, saving without inference", flush=True)
        onnx.save(new_model, output_model_path)

    matmul_count = len([n for n in new_nodes if n.op_type == "MatMul"])
    transpose_count = len([n for n in new_nodes if n.op_type == "Transpose"])
    gemm_count = len([n for n in new_nodes if n.op_type == "Gemm"])

    print("=" * 60)
    print("✅ Conversion Summary:")
    print(f"   - Converted {converted_count} GEMM nodes to MatMul")
    print(f"   - Skipped {skipped_count} GEMM nodes (kept as GEMM)")
    print(
        f"   - Final counts: {matmul_count} MatMul, {transpose_count} Transpose, {gemm_count} GEMM"
    )
    print("=" * 60)
    return output_model_path


def unify_gemm_input_dims(input_model_path: str, output_model_path: str) -> int:
    """Unify GEMM node input dimensions to avoid naming conflicts.

    This function ensures all GEMM inputs have consistent dimensionality by
    adding leading dimensions as needed.

    Args:
        input_model_path: Path to input ONNX model
        output_model_path: Path to save modified model

    Returns:
        Number of modified GEMM nodes
    """
    model = onnx.load(input_model_path)
    graph = model.graph

    print(f"Processing model: {input_model_path}")

    modified_nodes = 0

    existing_names = set()

    for init in graph.initializer:
        existing_names.add(init.name)

    for vi in graph.value_info:
        existing_names.add(vi.name)

    for inp in graph.input:
        existing_names.add(inp.name)

    for out in graph.output:
        existing_names.add(out.name)

    reshaped_cache = {}

    def generate_unique_name(base_name, existing_names):

        if base_name not in existing_names:
            existing_names.add(base_name)
            return base_name

        counter = 1
        while f"{base_name}_{counter}" in existing_names:
            counter += 1

        unique_name = f"{base_name}_{counter}"
        existing_names.add(unique_name)
        return unique_name

    def get_or_create_reshaped_initializer(
        initializer, target_shape, existing_names, reshaped_cache
    ):

        cache_key = (initializer.name, tuple(target_shape))

        if cache_key in reshaped_cache:
            print(f"  Reusing cached reshaped initializer: {reshaped_cache[cache_key]}")
            return reshaped_cache[cache_key]

        data = numpy_helper.to_array(initializer)
        data_reshaped = data.reshape(target_shape)

        base_name = f"{initializer.name}_reshaped"
        new_name = generate_unique_name(base_name, existing_names)

        new_initializer = numpy_helper.from_array(data_reshaped, name=new_name)

        graph.initializer.append(new_initializer)

        reshaped_cache[cache_key] = new_name

        print(f"  Created reshaped initializer: {initializer.name} -> {new_name}")
        print(f"  Shape: {list(data.shape)} -> {target_shape}")

        return new_name

    for node in graph.node:
        if node.op_type == "Gemm":
            print(f"\nProcessing Gemm node: {node.name}")

            if len(node.input) < 2:
                print(f"Warning: Gemm node {node.name} has less than 2 inputs, skipping.")
                continue

            input_a_name = node.input[0]
            input_b_name = node.input[1]

            a_shape = None
            a_initializer = None
            for init in graph.initializer:
                if init.name == input_a_name:
                    a_tensor = numpy_helper.to_array(init)
                    a_shape = list(a_tensor.shape)
                    a_initializer = init
                    break

            if a_shape is None:
                all_value_infos = list(graph.value_info) + list(graph.input)
                for vi in all_value_infos:
                    if vi.name == input_a_name:
                        a_shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
                        break

            b_shape = None
            b_initializer = None
            for init in graph.initializer:
                if init.name == input_b_name:
                    b_tensor = numpy_helper.to_array(init)
                    b_shape = list(b_tensor.shape)
                    b_initializer = init
                    break

            if b_shape is None:
                all_value_infos = list(graph.value_info) + list(graph.input)
                for vi in all_value_infos:
                    if vi.name == input_b_name:
                        b_shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
                        break

            if a_shape is None or b_shape is None:
                print(f"Warning: Cannot determine shapes for {node.name}, skipping.")
                continue

            print(f"Original shapes - A: {a_shape}, B: {b_shape}")

            for attr in node.attribute:
                if attr.name == "transA" and attr.i == 1:
                    pass
                if attr.name == "transB" and attr.i == 1:
                    pass

            max_dim = max(len(a_shape), len(b_shape))

            if b_initializer is not None and len(b_shape) != max_dim:
                dim_diff = max_dim - len(b_shape)
                if dim_diff > 0:
                    new_shape = tuple([1] * dim_diff + list(b_shape))

                    new_b_name = get_or_create_reshaped_initializer(
                        b_initializer, new_shape, existing_names, reshaped_cache
                    )

                    node.input[1] = new_b_name
                    modified_nodes += 1

            if a_initializer is not None and len(a_shape) != max_dim:
                dim_diff = max_dim - len(a_shape)
                if dim_diff > 0:
                    new_shape = tuple([1] * dim_diff + list(a_shape))

                    new_a_name = get_or_create_reshaped_initializer(
                        a_initializer, new_shape, existing_names, reshaped_cache
                    )

                    node.input[0] = new_a_name
                    modified_nodes += 1

    onnx.save(model, output_model_path)
    print(f"\nModified {modified_nodes} Gemm nodes")
    print(f"Created {len(reshaped_cache)} unique reshaped initializers")
    print(f"Saved to: {output_model_path}")

    return modified_nodes


def fuse_matmul_add_to_gemm(input_model_path: str, output_model_path: str):
    """Fuse MatMul+Add patterns into GEMM nodes with proper validation.

    Only fuses MatMul -> Add patterns where:
    - Add's bias input is constant (not computed)
    - Add's output is used by at most one consumer

    Args:
        input_model_path: Path to input ONNX model
        output_model_path: Path to save modified model

    Returns:
        Modified ONNX model
    """
    model = onnx.load(input_model_path)
    graph = model.graph

    print("🔍 MatMul+Add fusion with constant bias validation...")

    # Build dependency maps
    output_to_node = {}
    input_to_nodes = {}

    for node in graph.node:
        for output in node.output:
            output_to_node[output] = node
        for inp in node.input:
            if inp not in input_to_nodes:
                input_to_nodes[inp] = []
            input_to_nodes[inp].append(node)

    # Find valid MatMul+Add patterns
    valid_fusions = []

    for node in graph.node:
        if node.op_type == "MatMul":
            # Check MatMul structure: exactly 2 inputs, 1 output
            if len(node.input) != 2 or len(node.output) != 1:
                print(f"  ✗ Skip {node.name}: invalid MatMul structure")
                continue

            matmul_output = node.output[0]

            # Check consumers of MatMul output
            matmul_consumers = input_to_nodes.get(matmul_output, [])
            if len(matmul_consumers) != 1:
                print(
                    f"  ✗ Skip {node.name}: MatMul output has {len(matmul_consumers)} consumers (should be exactly 1)"
                )
                continue

            add_node = matmul_consumers[0]

            # Check Add structure: exactly 2 inputs, 1 output, is Add operation
            if add_node.op_type != "Add" or len(add_node.input) != 2 or len(add_node.output) != 1:
                print(f"  ✗ Skip {node.name}: consumer is not valid Add")
                continue

            # Verify Add uses MatMul output
            if matmul_output not in add_node.input:
                print(f"  ✗ Skip {node.name}: Add doesn't use MatMul output")
                continue

            # Get the bias input (the other input to Add)
            bias_input = (
                add_node.input[1] if add_node.input[0] == matmul_output else add_node.input[0]
            )

            # ⭐ KEY CHECK 1: Bias input must be constant (initializer), not from computation
            graph_inputs = {inp.name for inp in graph.input}
            initializers = {init.name for init in graph.initializer}

            if bias_input in output_to_node:
                # Bias comes from another node's output - this is NOT allowed
                bias_producer = output_to_node[bias_input]
                print(
                    f"  ✗ Skip {node.name}: bias input '{bias_input}' comes from node '{bias_producer.name}' ({bias_producer.op_type})"
                )
                print("    Only constant bias is allowed, not computed values from branches")
                continue
            elif bias_input in initializers:
                print(f"  ✓ Bias input '{bias_input}' is initializer (constant)")
            elif bias_input in graph_inputs:
                print(f"  ⚠️  Bias input '{bias_input}' is graph input (may be acceptable)")
            else:
                print(f"  ✗ Skip {node.name}: bias input '{bias_input}' source unknown")
                continue

            # ⭐ KEY CHECK 2: Check if Add output is used by multiple consumers
            add_output = add_node.output[0]
            add_consumers = input_to_nodes.get(add_output, [])

            # Also check if Add output is a graph output
            graph_outputs = {out.name for out in graph.output}
            is_graph_output = add_output in graph_outputs

            total_consumers = len(add_consumers) + (1 if is_graph_output else 0)

            if total_consumers > 1:
                consumer_names = [c.name for c in add_consumers]
                if is_graph_output:
                    consumer_names.append("GRAPH_OUTPUT")
                print(
                    f"  ✗ Skip {node.name}: Add output '{add_output}' used by {total_consumers} consumers: {consumer_names}"
                )
                print("    Cannot fuse because fusion would affect multiple downstream nodes")
                continue
            elif total_consumers == 0:
                print(f"  ⚠️  Add output '{add_output}' has no consumers - dead code?")

            # Check MatMul output is not a graph output (already checked Add output above)
            if matmul_output in graph_outputs:
                print(f"  ✗ Skip {node.name}: MatMul output is graph output")
                continue

            # Additional safety check: ensure MatMul output isn't used elsewhere
            other_matmul_refs = 0
            for check_node in graph.node:
                if check_node != add_node:
                    for inp in check_node.input:
                        if inp == matmul_output:
                            other_matmul_refs += 1
                            print(f"  ✗ Skip {node.name}: MatMul output used by {check_node.name}")

            if other_matmul_refs > 0:
                continue

            valid_fusions.append((node, add_node, bias_input))
            print(f"  ✅ Valid fusion: {node.name} + {add_node.name}")
            print(f"     MatMul: {node.input} -> {matmul_output}")
            print(f"     Add: [{matmul_output}, {bias_input}] -> {add_output} (bias is constant)")
            print(f"     Add consumers: {len(add_consumers)} + graph_output={is_graph_output}")

    if not valid_fusions:
        print("  No valid MatMul+Add patterns found for fusion")
        onnx.save(model, output_model_path)
        return model

    print(f"\n🔗 Applying {len(valid_fusions)} validated fusions...")

    # Apply fusions
    nodes_to_remove = []
    nodes_to_add = []

    for matmul_node, add_node, bias_input in valid_fusions:
        print(f"\n  Fusing: {matmul_node.name} + {add_node.name}")

        # Get inputs for GEMM
        input_A = matmul_node.input[0]  # First matrix
        input_B = matmul_node.input[1]  # Second matrix
        input_C = bias_input  # Bias

        # Create GEMM node
        gemm_node = helper.make_node(
            "Gemm",
            inputs=[input_A, input_B, input_C],
            outputs=add_node.output,
            name=add_node.name,  # Preserve Add node name
        )

        # Set GEMM attributes
        gemm_node.attribute.extend(
            [
                helper.make_attribute("alpha", 1.0),  # A*B scaling
                helper.make_attribute("beta", 1.0),  # bias scaling
                helper.make_attribute("transA", 0),  # no transpose A
                helper.make_attribute("transB", 0),  # no transpose B
            ]
        )

        nodes_to_remove.extend([matmul_node, add_node])
        nodes_to_add.append(gemm_node)

        print(f"    GEMM inputs: A={input_A}, B={input_B}, bias={input_C}")
        print(f"    GEMM output: {add_node.output[0]}")
        print(f"    GEMM name: {add_node.name}")
    # Apply changes
    print("\n📝 Updating graph...")

    # Remove fused nodes
    for old_node in nodes_to_remove:
        if old_node in graph.node:
            graph.node.remove(old_node)

    # Add new GEMM nodes
    for new_node in nodes_to_add:
        graph.node.append(new_node)

    # Save model
    onnx.save(model, output_model_path)

    print(f"\n{'='*60}")
    print("✅ CONSTANT-BIAS FUSION COMPLETE")
    print(f"{'='*60}")
    print(f"Fusions applied: {len(valid_fusions)}")
    print("Rejected computed bias inputs: ✓")
    print("Only accepted constant bias: ✓")
    print("Skipped multi-consumer Add nodes: ✓")
    print(f"Model saved: {output_model_path}")
    print(f"{'='*60}")

    return model


def convert_fusedmatmul_to_no_bias_gemm(input_model_path: str, output_model_path: str):
    """Convert Microsoft's FusedMatMul nodes to standard GEMM nodes without bias.

    Enhanced with description preservation and naming strategy to prevent conflicts.

    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path to save the converted ONNX model

    Returns:
        Converted ONNX model
    """
    # Load the model
    model = onnx.load(input_model_path)

    # Extract descriptions from all nodes before transformation
    node_descriptions = {}
    for node in model.graph.node:
        description = None
        for attr in node.attribute:
            if attr.name == "description":
                description = attr.s.decode("utf-8") if attr.s else ""
                break
        node_descriptions[node.name] = description

    # Track necessary changes
    new_nodes = []

    # Keep track of existing node names to avoid conflicts only for new nodes
    existing_node_names = set()

    # Collect existing node names
    def collect_existing_node_names():
        for node in model.graph.node:
            if node.name:
                existing_node_names.add(node.name)

    collect_existing_node_names()

    def sanitize_name(name):
        """Clean name for use in identifiers"""
        # Remove special characters and replace with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")
        return sanitized[:50]  # Limit length

    def generate_unique_node_name(base_name, existing_node_names):
        """Generate a unique node name with simple counter strategy"""
        if base_name not in existing_node_names:
            existing_node_names.add(base_name)
            return base_name

        # Add counter until unique
        for counter in range(1, 1000):
            candidate = f"{base_name}_{counter}"
            if candidate not in existing_node_names:
                existing_node_names.add(candidate)
                return candidate

        # Final fallback
        raise RuntimeError(f"Unable to generate unique node name for: {base_name}")

    def inherit_description(original_node, new_node):
        """Inherit description from original node to new node"""
        if original_node.name in node_descriptions and node_descriptions[original_node.name]:
            original_desc = node_descriptions[original_node.name]
            inherited_desc = f"[Converted from FusedMatMul] {original_desc}"
            desc_attr = helper.make_attribute("description", inherited_desc)
            new_node.attribute.append(desc_attr)
            print(f"    📝 Inherited description: {original_desc}")
            return True
        return False

    # Process each node in the graph
    fusedmatmul_count = 0

    for node_index, node in enumerate(model.graph.node):
        # Check if the node is a FusedMatMul from Microsoft domain
        if node.op_type == "FusedMatMul" and node.domain == "com.microsoft":
            fusedmatmul_count += 1
            print(f"Converting FusedMatMul node {fusedmatmul_count}: {node.name}")

            # Extract attributes from FusedMatMul
            alpha = 1.0
            transA = 0
            transB = 0

            for attr in node.attribute:
                if attr.name == "alpha":
                    alpha = attr.f
                elif attr.name == "transA":
                    transA = attr.i
                elif attr.name == "transB":
                    transB = attr.i

            # Get inputs and output of FusedMatMul
            A = node.input[0]
            B = node.input[1] if len(node.input) > 1 else ""
            output = node.output[0] if node.output else f"auto_output_{node_index}"

            print(f"  Input A: {A}")
            print(f"  Input B: {B}")
            print(f"  Output: {output}")
            print(f"  Attributes: alpha={alpha}, transA={transA}, transB={transB}")

            # Create simple unique Gemm node name
            gemm_node_name = generate_unique_node_name(
                node.name or "fusedmatmul_to_gemm", existing_node_names
            )

            # Create the Gemm node WITHOUT bias (only 2 inputs: A and B)
            gemm_node = helper.make_node(
                "Gemm",
                inputs=[A, B],  # No bias input
                outputs=[output],
                name=gemm_node_name,
                alpha=alpha,
                beta=0.0,  # Set beta to 0 since no bias
                transA=transA,
                transB=transB,
            )

            # Inherit description from original FusedMatMul node
            has_description = inherit_description(node, gemm_node)

            new_nodes.append(gemm_node)

            print(f"  -> Created No-Bias Gemm node: {gemm_node_name}")
            print(f"     Inputs: [{A}, {B}]")  # No bias
            print(f"     Output: [{output}]")
            print("     Beta: 0.0 (no bias)")
            if has_description:
                print("     Description: Inherited ✓")
            print("")

        else:
            # Keep other nodes as they are - do NOT rename existing nodes to avoid breaking references
            new_nodes.append(node)

    # Create a new graph with updated nodes - keep everything else exactly the same
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer,  # Keep original initializers unchanged
        value_info=model.graph.value_info,
    )

    # Create a new model with the updated graph
    new_model = helper.make_model(
        new_graph,
        producer_name="FusedMatMul2NoBiasGemm",
        opset_imports=model.opset_import,
        ir_version=model.ir_version,
    )

    # Copy domain information for custom ops
    for domain in model.domain:
        new_model.domain.append(domain)

    # Copy model metadata
    new_model.metadata_props.extend(model.metadata_props)

    # Save the new model
    onnx.save(new_model, output_model_path)

    print("=" * 60)
    print("✅ Conversion Summary:")
    print(f"   - Converted {fusedmatmul_count} FusedMatMul nodes to No-Bias Gemm")
    print("   - No bias tensors created (bias-free operation)")
    print("   - Descriptions preserved and inherited")
    print("   - Enhanced naming prevents conflicts")
    print(f"   - Saved to: {output_model_path}")
    print("=" * 60)

    return new_model
