import onnx
import os
import re
import subprocess
import yaml
from onnx import helper, numpy_helper, shape_inference
import numpy as np
import copy

def add_c_to_gemm(input_model_path, output_model_path):
    model = onnx.load(input_model_path)
    graph = model.graph
    
    for node in graph.node:
        if node.op_type == 'Gemm':
            
            if len(node.input) == 2:
                print(f"Find Gemm without C: {node.name}")
                
                input_a_name = node.input[0]
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
                                output_shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
                                
                           
                                if output_shape and all(output_shape): 
                                    transB = 0
                                    for attr in node.attribute:
                                        if attr.name == 'transB' and attr.i == 1:
                                            transB = 1
                                    
                                    c_length = output_shape[-1]
                                    b_shape = [c_length, 0] if transB == 0 else [0, c_length]
                            break
                

                if b_shape is not None and len(b_shape) >= 2 and (b_shape[0] > 0 or b_shape[1] > 0):
                    
                    transB = 0
                    for attr in node.attribute:
                        if attr.name == 'transB' and attr.i == 1:
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

def unify_gemm_input_dims(input_model_path, output_model_path):
    """
    统一GEMM节点的输入维度，避免名称冲突
    """
    model = onnx.load(input_model_path)
    graph = model.graph
    
    print(f"Processing model: {input_model_path}")
    
    modified_nodes = 0
    
    # 跟踪所有现有名称，避免冲突
    existing_names = set()
    
    # 收集所有现有名称
    for init in graph.initializer:
        existing_names.add(init.name)
    
    for vi in graph.value_info:
        existing_names.add(vi.name)
        
    for inp in graph.input:
        existing_names.add(inp.name)
        
    for out in graph.output:
        existing_names.add(out.name)
    
    # 跟踪已经创建的reshaped版本，避免重复创建
    reshaped_cache = {}  # original_name -> (target_shape, new_name)
    
    def generate_unique_name(base_name, existing_names):
        """生成唯一名称"""
        if base_name not in existing_names:
            existing_names.add(base_name)
            return base_name
        
        counter = 1
        while f"{base_name}_{counter}" in existing_names:
            counter += 1
        
        unique_name = f"{base_name}_{counter}"
        existing_names.add(unique_name)
        return unique_name
    
    def get_or_create_reshaped_initializer(initializer, target_shape, existing_names, reshaped_cache):
        """获取或创建reshaped的initializer，避免重复"""
        
        cache_key = (initializer.name, tuple(target_shape))
        
        # 检查缓存
        if cache_key in reshaped_cache:
            print(f"  Reusing cached reshaped initializer: {reshaped_cache[cache_key]}")
            return reshaped_cache[cache_key]
        
        # 创建新的reshaped initializer
        data = numpy_helper.to_array(initializer)
        data_reshaped = data.reshape(target_shape)
        
        # 生成唯一名称
        base_name = f"{initializer.name}_reshaped"
        new_name = generate_unique_name(base_name, existing_names)
        
        new_initializer = numpy_helper.from_array(data_reshaped, name=new_name)
        
        # 添加到图中
        graph.initializer.append(new_initializer)
        
        # 缓存结果
        reshaped_cache[cache_key] = new_name
        
        print(f"  Created reshaped initializer: {initializer.name} -> {new_name}")
        print(f"  Shape: {list(data.shape)} -> {target_shape}")
        
        return new_name

    for node in graph.node:
        if node.op_type == 'Gemm':
            print(f"\nProcessing Gemm node: {node.name}")
            
            if len(node.input) < 2:
                print(f"Warning: Gemm node {node.name} has less than 2 inputs, skipping.")
                continue
            
            input_a_name = node.input[0]
            input_b_name = node.input[1]
            
            # 获取输入A的形状
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
            
            # 获取输入B的形状
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
            
            # 获取transpose属性
            transA = 0
            transB = 0
            for attr in node.attribute:
                if attr.name == 'transA' and attr.i == 1:
                    transA = 1
                if attr.name == 'transB' and attr.i == 1:
                    transB = 1
            
            max_dim = max(len(a_shape), len(b_shape))
            
            # 处理输入B的维度统一
            if b_initializer is not None and len(b_shape) != max_dim:
                dim_diff = max_dim - len(b_shape)
                if dim_diff > 0:
                    new_shape = tuple([1] * dim_diff + list(b_shape))
                    
                    # 使用缓存机制获取或创建reshaped版本
                    new_b_name = get_or_create_reshaped_initializer(
                        b_initializer, new_shape, existing_names, reshaped_cache
                    )
                    
                    # 更新节点输入
                    node.input[1] = new_b_name
                    modified_nodes += 1
            
            # 处理输入A的维度统一
            if a_initializer is not None and len(a_shape) != max_dim:
                dim_diff = max_dim - len(a_shape)
                if dim_diff > 0:
                    new_shape = tuple([1] * dim_diff + list(a_shape))
                    
                    # 使用缓存机制获取或创建reshaped版本
                    new_a_name = get_or_create_reshaped_initializer(
                        a_initializer, new_shape, existing_names, reshaped_cache
                    )
                    
                    # 更新节点输入
                    node.input[0] = new_a_name
                    modified_nodes += 1

    onnx.save(model, output_model_path)
    print(f"\nModified {modified_nodes} Gemm nodes")
    print(f"Created {len(reshaped_cache)} unique reshaped initializers")
    print(f"Saved to: {output_model_path}")
    
    return modified_nodes

def optimize_matrix_operations(input_model_path, output_model_path):
    """
    Optimize matrix operations in ONNX model:
    1. Fuse MatMul+Add into GEMM
    2. Add missing C tensor to GEMM operations with only A and B inputs
    
    Args:
        input_model_path: Path to the input model
        output_model_path: Path to save the optimized model
    """
    model = onnx.load(input_model_path)
    graph = model.graph
    
    # Create maps for node traversal
    output_map = {}  # Map output tensor names to their producing nodes
    input_map = {}   # Map input tensor names to nodes consuming them
    
    for node in graph.node:
        for output in node.output:
            output_map[output] = node
        
        for input_name in node.input:
            if input_name not in input_map:
                input_map[input_name] = []
            input_map[input_name].append(node)
    
    # === PASS 1: Fuse MatMul+Add to GEMM ===
    nodes_to_remove = []
    nodes_to_add = []
    
    for node in graph.node:
        if node.op_type == 'MatMul':
            matmul_node = node
            matmul_output = matmul_node.output[0]
            
            # Check if this MatMul's output is used by exactly one Add node
            if matmul_output in input_map and len(input_map[matmul_output]) == 1:
                next_node = input_map[matmul_output][0]
                
                if next_node.op_type == 'Add':
                    add_node = next_node
                    
                    # Get MatMul inputs
                    a_name = matmul_node.input[0]
                    b_name = matmul_node.input[1]
                    
                    # Get Add bias input
                    if add_node.input[0] == matmul_output:
                        c_name = add_node.input[1]
                    else:
                        c_name = add_node.input[0]
                    
                    # Create new GEMM node
                    gemm_node = onnx.helper.make_node(
                        'Gemm',
                        inputs=[a_name, b_name, c_name],
                        outputs=[add_node.output[0]],
                        name=f"{matmul_node.name}_fused_with_{add_node.name}"
                    )
                    
                    # Add GEMM node attributes
                    alpha_attr = onnx.helper.make_attribute('alpha', 1.0)
                    beta_attr = onnx.helper.make_attribute('beta', 1.0)
                    gemm_node.attribute.append(alpha_attr)
                    gemm_node.attribute.append(beta_attr)
                    
                    # Mark nodes for removal and addition
                    nodes_to_remove.append(matmul_node)
                    nodes_to_remove.append(add_node)
                    nodes_to_add.append(gemm_node)
                    
                    print(f"Fusing MatMul {matmul_node.name} and Add {add_node.name} into GEMM {gemm_node.name}")
    
    # Apply the changes from Pass 1
    for node in nodes_to_remove:
        if node in graph.node:  # Check if node is still in graph
            graph.node.remove(node)
    
    for node in nodes_to_add:
        graph.node.append(node)
    
    # === PASS 2: Add missing C tensor to GEMM operations ===
    for node in graph.node:
        if node.op_type == 'Gemm' and len(node.input) == 2:
            print(f"Find Gemm without C: {node.name}")
            
            input_a_name = node.input[0]
            input_b_name = node.input[1]
            
            # Try to get B shape from initializer
            b_shape = None
            for init in graph.initializer:
                if init.name == input_b_name:
                    b_tensor = numpy_helper.to_array(init)
                    b_shape = b_tensor.shape
                    break
            
            # If not found, try to get from value_info
            if b_shape is None:
                for vi in graph.value_info:
                    if vi.name == input_b_name:
                        b_shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
                        break
            
            # If still not found, try to infer from output shape
            if b_shape is None:
                output_name = node.output[0]
                all_value_infos = list(graph.value_info) + list(graph.output)
                
                for vi in all_value_infos:
                    if vi.name == output_name:
                        # Ensure we have dimension info
                        if vi.type.tensor_type.shape.dim:
                            output_shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
                            
                            # Check if output_shape has enough elements
                            if output_shape and all(output_shape):
                                transB = 0
                                for attr in node.attribute:
                                    if attr.name == 'transB' and attr.i == 1:
                                        transB = 1
                                
                                c_length = output_shape[-1]
                                b_shape = [c_length, 0] if transB == 0 else [0, c_length]
                        break
            
            # If we successfully got shape info, add C tensor
            if b_shape is not None and len(b_shape) >= 2 and (b_shape[0] > 0 or b_shape[1] > 0):
                transB = 0
                for attr in node.attribute:
                    if attr.name == 'transB' and attr.i == 1:
                        transB = 1
                
                # Ensure we get valid shapes
                if transB == 0 and b_shape[1] > 0:
                    c_shape = [b_shape[1]]
                elif transB == 1 and b_shape[0] > 0:
                    c_shape = [b_shape[0]]
                else:
                    print(f"Warning: Invalid shape {b_shape} for {node.name}, skip this node.")
                    continue
                
                c_tensor = np.zeros(c_shape, dtype=np.float32)
                c_name = f"{node.name}_c_bias"
                
                c_initializer = numpy_helper.from_array(c_tensor, name=c_name)
                graph.initializer.append(c_initializer)
                
                node.input.append(c_name)
                print(f"Add C: {c_name}, Shape: {c_shape}")
            else:
                print(f"Warning: Cannot find valid shape for {node.name}, skip this node.")
    
    # Save the optimized model
    onnx.save(model, output_model_path)
    print(f"Model optimized and saved to: {output_model_path}")
    return model


def replace_biasgelu_with_gelu_add(input_model_path, output_model_path):
 
    model = onnx.load(input_model_path)
    
    # Collect all value_info entries by name for easy lookup
    value_info_map = {}
    for vi in model.graph.value_info:
        value_info_map[vi.name] = vi
    
    # Add input and output value_info to the map
    for inp in model.graph.input:
        value_info_map[inp.name] = inp
    
    for out in model.graph.output:
        value_info_map[out.name] = out
    
    # Create new node list and value_info list
    new_nodes = []
    new_value_info = []
    biasgelu_count = 0
    
    # Counter for generating unique names
    unique_id = 0
    def get_unique_name(prefix):
        nonlocal unique_id
        name = f"{prefix}_{unique_id}"
        unique_id += 1
        return name
    
    # Process all nodes
    for node in model.graph.node:
        if node.op_type == 'BiasGelu':
            biasgelu_count += 1
            
            # Get BiasGelu inputs and outputs
            input_name = node.input[0]  # X
            bias_name = node.input[1]   # Bias
            output_name = node.output[0]  # Y
            
            # Generate unique name prefix
            prefix = node.name if node.name else f"gelu_add"
            
            # Step 1: First apply Add operation to add bias
            add_output = get_unique_name(f"{prefix}_add_out")
            add_node = helper.make_node(
                'Add',
                inputs=[input_name, bias_name],
                outputs=[add_output],
                name=f"{prefix}_add"
            )
            new_nodes.append(add_node)
            
            # Create value_info for add_output with proper type and shape
            # Use the same type and shape as the input tensor if available
            if input_name in value_info_map:
                input_value_info = value_info_map[input_name]
                add_output_value_info = helper.make_tensor_value_info(
                    add_output,
                    input_value_info.type.tensor_type.elem_type,
                    [d.dim_value if d.dim_value else d.dim_param for d in input_value_info.type.tensor_type.shape.dim]
                )
                new_value_info.append(add_output_value_info)
                value_info_map[add_output] = add_output_value_info
            
            # Step 2: Then apply Gelu activation function
            gelu_node = helper.make_node(
                'Gelu',
                inputs=[add_output],
                outputs=[output_name],
                name=f"{prefix}_gelu"
            )
            new_nodes.append(gelu_node)
            
            # If we have output value_info, make sure it's preserved
            # Otherwise, create it with the same shape and type as the input to Gelu
            if output_name not in value_info_map and add_output in value_info_map:
                output_value_info = helper.make_tensor_value_info(
                    output_name,
                    value_info_map[add_output].type.tensor_type.elem_type,
                    [d.dim_value if d.dim_value else d.dim_param for d in value_info_map[add_output].type.tensor_type.shape.dim]
                )
                new_value_info.append(output_value_info)
                value_info_map[output_name] = output_value_info
        else:
            # Keep other nodes unchanged
            new_nodes.append(node)
    
    print(f"Replaced {biasgelu_count} BiasGelu nodes with Gelu+Add combinations")
    
    # Create new graph with all collected value_info
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer,
        value_info=list(model.graph.value_info) + new_value_info
    )
    
    # Build new model, preserving original model metadata
    new_model = helper.make_model(
        new_graph,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
        doc_string=model.doc_string
    )
    
    # Copy opset imports
    del new_model.opset_import[:]
    new_model.opset_import.extend(model.opset_import)
    
    # Add Microsoft domain if not present (for Gelu)
    has_ms_domain = any(opset.domain == "com.microsoft" for opset in new_model.opset_import)
    if not has_ms_domain:
        ms_opset = helper.make_opsetid("com.microsoft", 1)
        new_model.opset_import.append(ms_opset)
    
    # Copy IR version
    new_model.ir_version = model.ir_version
    
    # Run shape inference to ensure all shapes are properly defined
    try:
        new_model = shape_inference.infer_shapes(new_model)
        print("Shape inference successful")
    except Exception as e:
        print(f"Warning: Shape inference failed: {e}")
    
    # Skip validation and directly save if needed
    try:
        onnx.checker.check_model(new_model)
        print("Model validation successful!")
    except Exception as e:
        print(f"Warning: Model validation failed, but still saving: {e}")
    
    # Save the modified model
    onnx.save(new_model, output_model_path)
    print(f"Saved modified model to {output_model_path}")
    
    return new_model

def fix_layernorm_output(input_model_path: str, output_model_path: str) -> bool:
    """
    Fix output types and shapes for all LayerNorm operators in an ONNX model.
    
    Args:
        input_model_path (str): Path to the input model file
        output_model_path (str): Path to save the output model file
        
    Returns:
        bool: True if the operation succeeded, False otherwise
    """
    try:
        # Load the model
        model = onnx.load(input_model_path)
        graph = model.graph
        
        # Find all LayerNorm nodes
        layernorm_count = 0
        updated_count = 0
        tensor_info = {}
        
        # Collect tensor information
        # Process input tensors
        for input_tensor in graph.input:
            name = input_tensor.name
            shape = [dim.dim_value if dim.dim_value > 0 else None for dim in input_tensor.type.tensor_type.shape.dim]
            elem_type = input_tensor.type.tensor_type.elem_type
            tensor_info[name] = {"shape": shape, "elem_type": elem_type}
        
        # Process intermediate and output tensors
        for value_info in list(graph.value_info) + list(graph.output):
            name = value_info.name
            shape = [dim.dim_value if dim.dim_value > 0 else None for dim in value_info.type.tensor_type.shape.dim]
            elem_type = value_info.type.tensor_type.elem_type
            tensor_info[name] = {"shape": shape, "elem_type": elem_type}
        
        # Fix each LayerNorm node
        for node in graph.node:
            if node.op_type == 'LayerNormalization':
                layernorm_count += 1
                
                if not node.input:
                    continue
                
                # Get input information
                input_name = node.input[0]
                if input_name not in tensor_info:
                    continue
                
                input_info = tensor_info[input_name]
                input_shape = input_info["shape"]
                input_elem_type = input_info["elem_type"]
                
                # Get axis attribute
                axis = -1
                for attr in node.attribute:
                    if attr.name == "axis":
                        axis = attr.i
                        break
                
                # Process all outputs
                for i, output_name in enumerate(node.output):
                    # Determine correct output shape and type
                    output_shape = None
                    output_elem_type = input_elem_type
                    
                    if i == 0:  # Main output - same shape as input
                        output_shape = input_shape
                    else:  # mean and std outputs - shape depends on normalization axis
                        # Handle negative axis index
                        if axis < 0 and input_shape and None not in input_shape:
                            axis = len(input_shape) + axis
                        
                        # Create shape for mean and std (remove normalization axis)
                        if input_shape and None not in input_shape and 0 <= axis < len(input_shape):
                            output_shape = list(input_shape)
                            output_shape.pop(axis)  # Remove normalization axis
                    
                    # Find and remove existing value info
                    for value_info in list(graph.value_info):
                        if value_info.name == output_name:
                            graph.value_info.remove(value_info)
                            break
                    
                    # Create new value info
                    if output_shape and None not in output_shape:
                        new_value_info = onnx.helper.make_tensor_value_info(
                            output_name, 
                            output_elem_type, 
                            output_shape
                        )
                        graph.value_info.append(new_value_info)
                    
                    # Update graph output if needed
                    for j, output in enumerate(list(graph.output)):
                        if output.name == output_name:
                            if output_shape and None not in output_shape:
                                new_output = onnx.helper.make_tensor_value_info(
                                    output_name,
                                    output_elem_type,
                                    output_shape
                                )
                                graph.output.remove(output)
                                graph.output.insert(j, new_output)
                            break
                    
                    # Update tensor info dictionary
                    tensor_info[output_name] = {
                        "shape": output_shape,
                        "elem_type": output_elem_type
                    }
                    
                    print(f"  Output {i}: {output_name}, shape={output_shape}")  # Debug info
                
                updated_count += 1
        
        # Save the model
        onnx.save(model, output_model_path)
        print(f"Updated {updated_count}/{layernorm_count} LayerNorm nodes, model saved to {output_model_path}")
        return True
        
    except Exception as e:
        print(f"Error fixing LayerNorm outputs: {str(e)}")
        return False


def modify_conflict_outputs(input_model_path, output_model_path):
    model = onnx.load(input_model_path)
    graph = model.graph
    
    select_nodes = []
    for node in graph.node:
        if node.op_type == 'LayerNormalization' or node.op_type == 'MaxPool':
        # if node.op_type == 'MaxPool':
            select_nodes.append(node)
    
    print(f"Find {len(select_nodes)} Maxpool")
    
    outputs_to_remove = []
    
    new_nodes = []
    
    for node in graph.node:
        if (node.op_type == 'LayerNormalization' or node.op_type == 'MaxPool') and len(node.output) > 1:
        # if (node.op_type == 'MaxPool') and len(node.output) > 1:
            outputs_to_remove.extend(node.output[1:])
            
            new_node = onnx.NodeProto()
            new_node.CopyFrom(node)
            first_output = node.output[0]
            
            del new_node.output[:]
            new_node.output.append(first_output)
            
            new_nodes.append(new_node)
        else:
            new_nodes.append(node)
    
    del graph.node[:]
    graph.node.extend(new_nodes)
    
    new_outputs = []
    for output in graph.output:
        if output.name not in outputs_to_remove:
            new_outputs.append(output)
    
    del graph.output[:]
    graph.output.extend(new_outputs)
    
    onnx.save(model, output_model_path)
    print(f"Saved to: {output_model_path}")
    
def convert_squeeze_unsqueeze_input_to_attr(input_model_path, output_model_path):
    """
    Convert Squeeze and Unsqueeze nodes with axes as input to axes as attribute.
    This is useful for compatibility with older ONNX versions where axes was only supported as an attribute.
    
    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path to save the converted ONNX model
    """
    model = onnx.load(input_model_path)
    
    modified_nodes = []
    modified_count = 0
    
    initializers = {init.name: init for init in model.graph.initializer}
    
    for node in model.graph.node:
        # Check if the node is Squeeze or Unsqueeze with more than one input
        if (node.op_type in ['Squeeze', 'Unsqueeze']) and len(node.input) > 1:
            modified_count += 1
            
            data_input = node.input[0]
            
            axes_input_name = node.input[1]
            
            if axes_input_name in initializers:
                # Get the axes values from the initializer
                axes_initializer = initializers[axes_input_name]
                axes_np = numpy_helper.to_array(axes_initializer)
                axes_list = axes_np.tolist()
                
                # Make the axes a scalar if it's a single value
                if isinstance(axes_list, list) and len(axes_list) == 1:
                    axes_list = axes_list[0]
                
                # Create a new node with axes as attribute instead of input
                new_node = helper.make_node(
                    op_type=node.op_type,
                    inputs=[data_input],  
                    outputs=list(node.output),
                    name=node.name,
                    axes=axes_list 
                )
                
                # Copy other attributes if they exist
                for attr in node.attribute:
                    if attr.name != 'axes':
                        new_node.attribute.append(attr)
                
                modified_nodes.append(new_node)
            else:
                # If we can't find the axes initializer, keep the original node
                print(f"Warning: Cannot find '{node.name}' axes initializer. Keep the original node.")
                modified_nodes.append(node)
        else:
            # Keep all other nodes as they are
            modified_nodes.append(node)
    
    print(f"Modified {modified_count} Squeeze/Unsqueeze nodes")
    
    # Identify initializers that are no longer referenced
    # This happens when we convert the axes from input to attribute
    used_inputs = set()
    for node in modified_nodes:
        for input_name in node.input:
            used_inputs.add(input_name)
    
    unused_initializers = set()
    for init in model.graph.initializer:
        if init.name not in used_inputs:
            unused_initializers.add(init.name)
    
    # Create a new graph with the modified nodes and without unused initializers
    new_graph = helper.make_graph(
        nodes=modified_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=[init for init in model.graph.initializer if init.name not in unused_initializers]
    )
    
    # Copy over value_info from the original model
    for vi in model.graph.value_info:
        new_graph.value_info.append(vi)
    
    # Create a new model with the updated graph
    new_model = helper.make_model(
        new_graph,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
        doc_string=model.doc_string
    )
    
    # Copy over IR version and opset imports
    new_model.ir_version = model.ir_version
    new_model.opset_import.extend(model.opset_import)
    
    # Save the model
    onnx.save(new_model, output_model_path)
    print(f"Saved to {output_model_path}")
    
    return new_model

def run_optmization_remove_biasgelu(onnx_train_file, onnx_out_file):
    """
    Replace BiasGelu operations with Add+Gelu while maintaining shape consistency.
    
    Args:
        onnx_train_file: Path to input ONNX model file
        onnx_out_file: Path to output ONNX model file
    """
    # Load the model
    model = onnx.load(onnx_train_file)
    graph = model.graph
    
    # Create new nodes list to replace the old ones
    new_nodes = []
    replaced_count = 0
    
    # Keep track of used initializers
    used_initializers = set()
    new_initializers = []
    
    # Process all nodes
    for node in graph.node:
        if node.op_type == "BiasGelu":
            print(f"🔄 Replacing BiasGeluFusion: {node.name}")
            replaced_count += 1
            
            # Get input and output tensors
            X, Bias = node.input
            output = node.output[0]
            
            # Create intermediate tensor name
            intermediate_output = f"{X}_add_bias"
            
            # Create a new unique bias name to avoid sharing
            new_bias_name = f"{Bias}_{node.name}"
            
            # Find the original bias initializer
            bias_initializer = None
            for initializer in graph.initializer:
                if initializer.name == Bias:
                    bias_initializer = initializer
                    break
            
            # Create a copy of the bias initializer with a new name if found
            if bias_initializer is not None:
                new_bias = onnx.helper.make_tensor(
                    name=new_bias_name,
                    data_type=bias_initializer.data_type,
                    dims=bias_initializer.dims,
                    vals=bias_initializer.raw_data if bias_initializer.raw_data else 
                          bias_initializer.float_data or bias_initializer.int32_data or 
                          bias_initializer.int64_data or bias_initializer.uint64_data,
                    raw=bool(bias_initializer.raw_data)
                )
                new_initializers.append(new_bias)
                used_initializers.add(Bias)
            else:
                # If we can't find the initializer, it might be an input or value_info
                # In this case, we should keep the original bias name
                new_bias_name = Bias
            
            # Create Add node with the new bias
            add_node = helper.make_node(
                "Add",
                inputs=[X, new_bias_name],
                outputs=[intermediate_output],
                name=f"{node.name}_Add",
            )
            
            # Create Gelu node
            gelu_node = helper.make_node(
                "Gelu",
                inputs=[intermediate_output],
                outputs=node.output,
                name=f"{node.name}_Gelu",
            )
            
            # Add shape information for the intermediate tensor
            # Try to find X's shape info
            X_shape = None
            X_type = 1  # Default to FLOAT
            
            # Look for X in inputs, outputs, value_info, or initializers
            for info in graph.input:
                if info.name == X:
                    X_shape = [d.dim_value if d.HasField("dim_value") else -1 for d in info.type.tensor_type.shape.dim]
                    X_type = info.type.tensor_type.elem_type
                    break
                    
            if X_shape is None:
                for info in graph.value_info:
                    if info.name == X:
                        X_shape = [d.dim_value if d.HasField("dim_value") else -1 for d in info.type.tensor_type.shape.dim]
                        X_type = info.type.tensor_type.elem_type
                        break
            
            # Add value_info for intermediate tensor
            if X_shape:
                value_info = helper.make_tensor_value_info(
                    intermediate_output,
                    X_type,
                    X_shape
                )
                graph.value_info.append(value_info)
            
            # Add the new nodes to our list
            new_nodes.extend([add_node, gelu_node])
        else:
            # Keep other nodes unchanged
            new_nodes.append(node)
    
    # Add the new initializers to the graph
    for initializer in new_initializers:
        graph.initializer.append(initializer)
    
    # Replace nodes in the graph
    graph.ClearField("node")
    graph.node.extend(new_nodes)
    
    # Rest of the function remains the same...
    # Create a copy of the model for selective shape inference
    safe_model = copy.deepcopy(model)
    
    # Remove Microsoft custom operators that might cause shape inference to fail
    ms_nodes = []
    for node in safe_model.graph.node:
        if node.domain == "com.microsoft":
            ms_nodes.append(node)
    
    if ms_nodes:
        print(f"⚠️ Found {len(ms_nodes)} Microsoft custom operators that might affect shape inference")
        
        # Try to run shape inference on the modified model without MS operators
        try:
            # Create a temporary graph without Microsoft operators
            temp_model = copy.deepcopy(safe_model)
            temp_graph = temp_model.graph
            
            # Remove Microsoft custom operators
            temp_nodes = [node for node in temp_graph.node if node.domain != "com.microsoft"]
            temp_graph.ClearField("node")
            temp_graph.node.extend(temp_nodes)
            
            # Run shape inference on this simplified model
            inferred_model = shape_inference.infer_shapes(temp_model)
            
            # Collect inferred shapes for our new nodes
            inferred_value_infos = {}
            for value_info in inferred_model.graph.value_info:
                inferred_value_infos[value_info.name] = value_info
            
            # Update the original model with any newly inferred shapes
            for name, value_info in inferred_value_infos.items():
                # Skip if already exists
                if any(info.name == name for info in model.graph.value_info):
                    continue
                
                model.graph.value_info.append(value_info)
                
            print("✅ Partial shape inference completed for non-Microsoft operators")
        except Exception as e:
            print(f"⚠️ Partial shape inference failed: {e}")
    else:
        # No Microsoft operators, try regular shape inference
        try:
            model = shape_inference.infer_shapes(model)
            print("✅ Shape inference completed successfully")
        except Exception as e:
            print(f"⚠️ Shape inference failed: {e}")
    
    # Save the modified model
    onnx.save(model, onnx_out_file)
    
    if replaced_count > 0:
        print(f"✅ Successfully replaced {replaced_count} BiasGelu nodes with Add + GELU.")
    else:
        print("⚠️ No BiasGelu nodes were replaced.")
    
    return model

def run_optmization_remove_biasgelugrad(onnx_train_file, onnx_out_file):
    """
    Optimize ONNX model:
    1. Convert BiasGeluGrad_dX nodes to GeluGrad nodes
    2. Remove bias input (input 2)
    3. Delete subsequent ReduceSum nodes
    
    Args:
        onnx_train_file: Input ONNX model path
        onnx_out_file: Output ONNX model path
    """
    # Load ONNX model
    model = onnx.load(onnx_train_file)
    graph = model.graph
    modified_model = copy.deepcopy(model)
    modified_graph = modified_model.graph
    
    # Track nodes to be removed
    nodes_to_remove = set()
    
    # Map original output names to new output names
    output_mapping = {}
    
    # First pass: identify BiasGeluGrad_dX nodes and corresponding ReduceSum nodes
    for i, node in enumerate(graph.node):
        if node.op_type == "BiasGeluGrad_dX" or (
            node.domain == "com.microsoft" and 
            "BiasGeluGrad_dX" in node.name
        ):
            # Create new GeluGrad node
            new_inputs = [node.input[0], node.input[1]]  # Remove bias input (input[2])
            new_node = helper.make_node(
                "GeluGrad",
                inputs=new_inputs,
                outputs=[node.output[0]],
                name=f"GeluGrad_{i}"
            )
            
            # Replace BiasGeluGrad_dX node with new GeluGrad node
            modified_graph.node.remove(modified_graph.node[i])
            modified_graph.node.insert(i, new_node)
            
            # Check for ReduceSum nodes that take this node's output as input
            for j, next_node in enumerate(graph.node):
                if next_node.op_type == "ReduceSum" and node.output[0] in next_node.input:
                    nodes_to_remove.add(j)
                    # Map ReduceSum output to new node output
                    output_mapping[next_node.output[0]] = node.output[0]
    
    # Second pass: remove identified ReduceSum nodes
    reduced_nodes = []
    for i, node in enumerate(modified_graph.node):
        if i not in nodes_to_remove:
            # Update inputs that reference outputs of removed nodes
            for j, input_name in enumerate(node.input):
                if input_name in output_mapping:
                    node.input[j] = output_mapping[input_name]
            reduced_nodes.append(node)
    
    # Replace nodes in the graph
    del modified_graph.node[:]
    modified_graph.node.extend(reduced_nodes)
    
    # Update model outputs if needed
    for output in modified_graph.output:
        if output.name in output_mapping:
            output.name = output_mapping[output.name]
    
    # Save modified model
    onnx.save(modified_model, onnx_out_file)
    print(f"Optimized model saved to {onnx_out_file}")

def optimize_reshape_fusion(input_model_path: str, output_model_path: str) -> None:
    """
    Optimize ONNX model by fusing consecutive Reshape operations.
    
    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path where the optimized ONNX model will be saved
    """
    print(f"Loading model: {input_model_path}")
    model = onnx.load(input_model_path)
    
    # Create mapping from node name to node
    node_map = {}
    for node in model.graph.node:
        node_map[node.name] = node
    
    # Create mapping from input name to producing node
    input_to_node = {}
    for node in model.graph.node:
        for output in node.output:
            input_to_node[output] = node
    
    # Create mapping from output name to consuming nodes
    output_to_nodes = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name not in output_to_nodes:
                output_to_nodes[input_name] = []
            output_to_nodes[input_name].append(node)
    
    # Find all Reshape nodes
    reshape_nodes = [node for node in model.graph.node if node.op_type == "Reshape"]
    
    # Track nodes to be removed by index rather than node objects
    # This avoids the "unhashable type: 'NodeProto'" error
    nodes_to_remove_indices = []
    
    # Track value info to keep
    value_info_to_keep = set(vi.name for vi in model.graph.value_info)
    
    # For each Reshape node, check if its input is also from a Reshape node
    for reshape_node in reshape_nodes:
        # Get the input of the current Reshape node
        input_name = reshape_node.input[0]
        
        # Check if the input comes from another Reshape operation
        if input_name in input_to_node and input_to_node[input_name].op_type == "Reshape":
            previous_reshape = input_to_node[input_name]
            
            # Check if the previous Reshape is only used by the current Reshape
            if input_name in output_to_nodes and len(output_to_nodes[input_name]) == 1:
                print(f"Found fusible Reshape pair: {previous_reshape.name} -> {reshape_node.name}")
                
                # Get the shape tensors for both Reshape nodes
                prev_shape_tensor_name = previous_reshape.input[1]
                current_shape_tensor_name = reshape_node.input[1]
                
                # Modify the current Reshape node to connect directly to the input of the previous Reshape
                reshape_node.input[0] = previous_reshape.input[0]
                
                # Mark the previous Reshape node for removal by its index
                for i, node in enumerate(model.graph.node):
                    if (node.name == previous_reshape.name and 
                        node.op_type == previous_reshape.op_type and
                        node.input == previous_reshape.input and 
                        node.output == previous_reshape.output):
                        nodes_to_remove_indices.append(i)
                        break
                
                # Intermediate value info doesn't need to be kept
                if input_name in value_info_to_keep:
                    value_info_to_keep.remove(input_name)
    
    # Handle custom nodes from Microsoft
    # Since Microsoft nodes might have a different structure or behavior
    # We need to be careful when dealing with them
    custom_nodes = [node for node in model.graph.node if node.domain.startswith('com.microsoft')]
    print(f"Found {len(custom_nodes)} Microsoft custom nodes. These will be preserved.")
    
    # Create a new graph excluding the nodes to be removed
    new_nodes = []
    for i, node in enumerate(model.graph.node):
        if i not in nodes_to_remove_indices:
            new_nodes.append(node)
    
    # Create a new value info list, keeping only the needed value info
    new_value_info = [vi for vi in model.graph.value_info if vi.name in value_info_to_keep]
    
    # Create a new graph
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer,
        value_info=new_value_info
    )
    
    # Create a new model
    new_model = helper.make_model(
        new_graph, 
        producer_name="ONNX Reshape Fusion Optimizer",
        ir_version=model.ir_version,
        opset_imports=model.opset_import
    )
    
    # Preserve custom opsets from the original model
    new_model.opset_import.extend([opset for opset in model.opset_import if opset.domain.startswith('com.microsoft')])
    
    # Save the optimized model
    onnx.save(new_model, output_model_path)
    
    # Print statistics
    print(f"Original model node count: {len(model.graph.node)}")
    print(f"Optimized model node count: {len(new_model.graph.node)}")
    print(f"Removed Reshape nodes: {len(nodes_to_remove_indices)}")
    print(f"Optimized model saved to: {output_model_path}")


def remove_identity_reducesum(input_model_path, output_model_path):
    """
    Fixed version of remove_identity_reducesum to handle multiple levels of Identity
    and transform appropriate ReduceSum nodes to Reshape operations
    """
    import onnx
    import numpy as np
    from onnx import shape_inference, helper, TensorProto
    
    # Load the model and infer shapes
    model = onnx.load(input_model_path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: Shape inference failed: {e}. Continuing without shape information.")
    
    graph = model.graph
    
    # Build node mapping and input-output relationships
    node_map = {node.name: node for node in graph.node}
    output_to_node = {}  # maps output name to producing node
    input_to_nodes = {}  # maps input name to consuming nodes
    
    for node in graph.node:
        for output in node.output:
            output_to_node[output] = node
        
        for input_name in node.input:
            if input_name not in input_to_nodes:
                input_to_nodes[input_name] = []
            input_to_nodes[input_name].append(node)
    
    # Store tensor shapes
    shape_info = {}
    for info in list(graph.value_info) + list(graph.input) + list(graph.output):
        if hasattr(info.type.tensor_type.shape, 'dim'):
            dims = []
            for dim in info.type.tensor_type.shape.dim:
                if dim.dim_value:
                    dims.append(dim.dim_value)
                else:
                    dims.append(-1)
            shape_info[info.name] = dims
    
    # Get initializer shapes
    for initializer in graph.initializer:
        shape_info[initializer.name] = list(initializer.dims)
    
    # Identity resolution: build a complete mapping directly to the source
    replacement_map = {}
    identity_nodes = []
    
    # First pass: collect all Identity nodes
    for node in graph.node:
        if node.op_type == "Identity":
            identity_nodes.append(node)
    
    # Function to recursively resolve Identity chains
    def resolve_identity_source(tensor_name):
        """Recursively resolve the source of a tensor through Identity nodes"""
        if tensor_name in output_to_node and output_to_node[tensor_name].op_type == "Identity":
            identity_node = output_to_node[tensor_name]
            source_name = identity_node.input[0]
            # Recursive call to handle chained identities
            return resolve_identity_source(source_name)
        return tensor_name
    
    # Build replacement map by resolving full Identity chains at once
    for node in identity_nodes:
        output_name = node.output[0]
        source_name = resolve_identity_source(node.input[0])  # Get the ultimate source
        replacement_map[output_name] = source_name
    
    # Process ReduceSum nodes
    reducesum_nodes = []
    reshape_nodes_to_add = []
    
    for node in graph.node:
        if node.op_type == "ReduceSum":
            input_name = node.input[0]
            output_name = node.output[0]
            
            # Resolve input if it's from an Identity node
            if input_name in replacement_map:
                input_name = replacement_map[input_name]
            
            # Check for dimension 1 reduction with keepdims=0
            keepdims = 1  # Default value
            for attr in node.attribute:
                if attr.name == "keepdims":
                    keepdims = attr.i
                    break
            
            # Get reduction axes
            axes = []
            for attr in node.attribute:
                if attr.name == "axes":
                    axes = list(attr.ints)
                    break
            
            # If opset >= 13, axes might be an input
            if len(node.input) > 1 and not axes:
                axes_name = node.input[1]
                for initializer in graph.initializer:
                    if initializer.name == axes_name:
                        axes = onnx.numpy_helper.to_array(initializer).tolist()
                        if not isinstance(axes, list):
                            axes = [axes]
                        break
            
            # Get input shape
            if input_name in shape_info:
                input_shape = shape_info[input_name]
                
                # Check if all reduction axes have dimension 1
                all_dim_one = True
                for axis in axes:
                    # Handle negative axis
                    if axis < 0:
                        axis = len(input_shape) + axis
                    
                    if 0 <= axis < len(input_shape) and input_shape[axis] == 1:
                        continue
                    else:
                        all_dim_one = False
                        break
                
                if all_dim_one and axes:
                    if keepdims == 1:
                        # Simple replacement case
                        replacement_map[output_name] = input_name
                        reducesum_nodes.append(node)
                    elif keepdims == 0:
                        # Need to add a Reshape node
                        # Calculate output shape by removing dimensions with size 1
                        output_shape = []
                        for i, dim in enumerate(input_shape):
                            if i not in axes and (i - len(input_shape)) not in axes:
                                output_shape.append(dim)
                        
                        # Create shape tensor for Reshape
                        shape_tensor_name = f"{node.name}_shape"
                        shape_tensor = helper.make_tensor(
                            name=shape_tensor_name,
                            data_type=TensorProto.INT64,
                            dims=[len(output_shape)],
                            vals=output_shape
                        )
                        
                        # Create Reshape node
                        reshape_node = helper.make_node(
                            "Reshape",
                            inputs=[input_name, shape_tensor_name],
                            outputs=[output_name],
                            name=f"{node.name}_reshape"
                        )
                        
                        # Store for later addition
                        reshape_nodes_to_add.append((reshape_node, shape_tensor))
                        reducesum_nodes.append(node)
    
    # Update all node inputs using the complete replacement mapping
    for node in graph.node:
        if node not in identity_nodes and node not in reducesum_nodes:
            modified = False
            for i, input_name in enumerate(node.input):
                # Apply replacement if needed, with special attention to Reshape nodes
                if input_name in replacement_map:
                    node.input[i] = replacement_map[input_name]
                    modified = True
                    
                    # Special handling for Reshape nodes to ensure shape info is preserved
                    if node.op_type == "Reshape" and i == 0:  # If this is the data input to Reshape
                        # If a Reshape node's input was replaced, ensure shape remains correct
                        if len(node.input) > 1:  # Has shape input
                            shape_input = node.input[1]
                            # Verify shape input exists
                            shape_exists = False
                            for init in graph.initializer:
                                if init.name == shape_input:
                                    shape_exists = True
                                    break
                            
                            if not shape_exists:
                                print(f"Warning: Reshape node {node.name} has missing shape input after Identity removal")
                                # Attempt to create a shape input if missing
                                if node.output[0] in shape_info:
                                    output_shape = shape_info[node.output[0]]
                                    shape_tensor_name = f"{node.name}_shape_fixed"
                                    shape_tensor = helper.make_tensor(
                                        name=shape_tensor_name,
                                        data_type=TensorProto.INT64,
                                        dims=[len(output_shape)],
                                        vals=output_shape
                                    )
                                    graph.initializer.append(shape_tensor)
                                    node.input[1] = shape_tensor_name
                                    print(f"  Fixed: Added shape tensor {shape_tensor_name} with shape {output_shape}")
            
            if modified and node.op_type == "Reshape":
                print(f"Updated inputs for Reshape node: {node.name}")
                print(f"  New inputs: {node.input}")
    
    # Update graph outputs
    for output in graph.output:
        if output.name in replacement_map:
            # Find the value_info for the replacement tensor
            replacement_info = None
            for info in graph.value_info:
                if info.name == replacement_map[output.name]:
                    replacement_info = info
                    break
            
            if replacement_info:
                # Copy type information from the replacement
                output.type.CopyFrom(replacement_info.type)
            
            # Update name
            old_name = output.name
            output.name = replacement_map[old_name]
            print(f"Updated graph output: {old_name} -> {output.name}")
    
    # Remove nodes and add new reshape nodes
    new_nodes = [node for node in graph.node if node not in identity_nodes and node not in reducesum_nodes]
    
    # Add shape tensors to initializers and reshape nodes
    for reshape_node, shape_tensor in reshape_nodes_to_add:
        graph.initializer.append(shape_tensor)
        new_nodes.append(reshape_node)
    
    # Clear and re-add nodes
    graph.ClearField("node")
    graph.node.extend(new_nodes)
    
    # Save model
    onnx.save(model, output_model_path)
    
    print(f"Saved to {output_model_path}")
    print(f"Removed {len(identity_nodes)} Identity nodes")
    print(f"Removed {len(reducesum_nodes)} ReduceSum nodes")
    print(f"Added {len(reshape_nodes_to_add)} Reshape nodes")
    
    return model

def convert_reducesum_axes_to_attr(input_file: str, output_file: str):
    model = onnx.load(input_file)
    graph = model.graph
    
    new_nodes = []
    
    initializers = {init.name: init for init in graph.initializer}
    
    for node in graph.node:
        if node.op_type == "ReduceSum":
            if len(node.input) >= 2:
                data_input = node.input[0]
                axes_input = node.input[1]
                
                if axes_input in initializers:
                    axes_tensor = initializers[axes_input]
                    axes_np = numpy_helper.to_array(axes_tensor)
                    axes_list = axes_np.tolist()
                    
                    new_node = helper.make_node(
                        op_type="ReduceSum",
                        inputs=[data_input],
                        outputs=node.output,
                        name=node.name,
                        axes=axes_list
                    )
                    
                    for attr in node.attribute:
                        if attr.name != "axes":
                            new_node.attribute.append(attr)
                    
                    new_nodes.append(new_node)
                else:
                    new_nodes.append(node)
            else:
                new_nodes.append(node)
        else:
            new_nodes.append(node)
    
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=graph.input,
        outputs=graph.output,
        initializer=graph.initializer,
        value_info=graph.value_info
    )
    
    new_model = helper.make_model(
        new_graph,
        producer_name="ReduceSumAxesConverter",
        ir_version=model.ir_version,
        opset_imports=model.opset_import
    )
    
    new_model.metadata_props.extend(model.metadata_props)
    
    for domain in model.domain:
        new_model.domain.append(domain)
    
    onnx.save(new_model, output_file)
    print(f"Model converted and saved to: {output_file}")


import onnx
import numpy as np
import hashlib
import re
from onnx import helper, numpy_helper

def convert_fusedmatmul_to_gemm(input_model_path, output_model_path):
    """
    Convert Microsoft's FusedMatMul nodes to standard Gemm nodes in an ONNX model.
    Enhanced with complex naming strategy to prevent any naming conflicts.
    Names are encoded with input/output information and unique identifiers.
    
    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path to save the converted ONNX model
    """
    # Load the model
    model = onnx.load(input_model_path)
    
    # Track necessary changes
    new_nodes = []
    new_initializers = []
    
    # Keep track of all existing names to avoid conflicts
    existing_names = set()
    
    # Collect all existing names from the model
    def collect_existing_names():
        for node in model.graph.node:
            if node.name:
                existing_names.add(node.name)
            for output in node.output:
                existing_names.add(output)
        
        for init in model.graph.initializer:
            existing_names.add(init.name)
        
        for vi in model.graph.value_info:
            existing_names.add(vi.name)
        
        for inp in model.graph.input:
            existing_names.add(inp.name)
        
        for out in model.graph.output:
            existing_names.add(out.name)
    
    collect_existing_names()
    
    def sanitize_name(name):
        """Clean name for use in identifiers"""
        # Remove special characters and replace with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized[:50]  # Limit length
    
    def encode_inputs_outputs(inputs, outputs):
        """Create a unique hash from inputs and outputs"""
        # Combine all input and output names
        combined = '|'.join(inputs + outputs)
        # Create a short hash
        hash_obj = hashlib.md5(combined.encode())
        return hash_obj.hexdigest()[:8]
    
    def create_enhanced_name(base_name, inputs, outputs, node_index, name_type="node"):
        """Create an enhanced name with input/output encoding"""
        
        # 1. Sanitize base name
        clean_base = sanitize_name(base_name) if base_name else f"auto_{name_type}"
        
        # 2. Create input signature
        input_sig = "_".join([sanitize_name(inp)[:15] for inp in inputs[:2]])  # First 2 inputs
        
        # 3. Create output signature  
        output_sig = "_".join([sanitize_name(out)[:15] for out in outputs[:2]])  # First 2 outputs
        
        # 4. Create hash of all inputs/outputs
        io_hash = encode_inputs_outputs(inputs, outputs)
        
        # 5. Combine everything
        enhanced_name = f"{clean_base}_i{input_sig}_o{output_sig}_h{io_hash}_n{node_index}"
        
        # 6. Ensure uniqueness
        return generate_absolutely_unique_name(enhanced_name, existing_names)
    
    def generate_absolutely_unique_name(base_name, existing_names):
        """Generate a globally unique name with fallback strategies"""
        if base_name not in existing_names:
            existing_names.add(base_name)
            return base_name
        
        # Strategy 1: Add counter
        for counter in range(1, 1000):
            candidate = f"{base_name}_v{counter}"
            if candidate not in existing_names:
                existing_names.add(candidate)
                return candidate
        
        # Strategy 2: Add timestamp-like suffix
        import time
        timestamp_suffix = str(int(time.time() * 1000000))[-8:]  # Last 8 digits
        candidate = f"{base_name}_t{timestamp_suffix}"
        if candidate not in existing_names:
            existing_names.add(candidate)
            return candidate
        
        # Strategy 3: Add random suffix (fallback)
        import random
        for _ in range(100):
            random_suffix = ''.join(random.choices('0123456789abcdef', k=8))
            candidate = f"{base_name}_r{random_suffix}"
            if candidate not in existing_names:
                existing_names.add(candidate)
                return candidate
        
        # Final fallback
        raise RuntimeError(f"Unable to generate unique name for: {base_name}")
    
    def infer_tensor_shape(tensor_name):
        """Infer shape of a tensor from model info"""
        # Check value_info
        for vi in model.graph.value_info:
            if vi.name == tensor_name:
                return [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
        
        # Check initializers
        for init in model.graph.initializer:
            if init.name == tensor_name:
                return list(init.dims)
        
        # Check graph inputs
        for inp in model.graph.input:
            if inp.name == tensor_name:
                return [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        
        return None
    
    def create_enhanced_bias_tensor(original_node, inputs, outputs, node_index, target_shape):
        """Create a bias tensor with enhanced naming"""
        
        # Create enhanced name for bias
        bias_base_name = f"{original_node.name}_zero_bias" if original_node.name else "auto_bias"
        
        bias_name = create_enhanced_name(
            bias_base_name,
            inputs,
            outputs + [f"bias_for_{outputs[0]}"],  # Include bias info in outputs
            node_index,
            "bias"
        )
        
        # Create zero tensor
        zero_tensor = numpy_helper.from_array(
            np.zeros(target_shape, dtype=np.float32),
            name=bias_name
        )
        
        return zero_tensor, bias_name
    
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
            
            # Infer shapes for bias creation
            a_shape = infer_tensor_shape(A)
            b_shape = infer_tensor_shape(B)
            
            print(f"  Inferred shapes: A={a_shape}, B={b_shape}")
            
            # Calculate bias shape
            if a_shape and b_shape:
                # Apply transpose if needed
                effective_a_shape = a_shape[::-1] if transA else a_shape
                effective_b_shape = b_shape[::-1] if transB else b_shape
                
                # For matmul: [M,K] * [K,N] = [M,N], bias needs shape [N]
                if len(effective_b_shape) >= 2:
                    c_shape = [effective_b_shape[-1]]
                elif len(effective_b_shape) == 1:
                    c_shape = [effective_b_shape[0]]
                else:
                    c_shape = [1]  # Fallback
            else:
                c_shape = [1]  # Safe fallback
            
            print(f"  Calculated bias shape: {c_shape}")
            
            # Create enhanced bias tensor
            bias_tensor, bias_name = create_enhanced_bias_tensor(
                node, 
                [A, B], 
                [output], 
                node_index, 
                c_shape
            )
            new_initializers.append(bias_tensor)
            
            print(f"  Created bias tensor: {bias_name}")
            
            # Create enhanced Gemm node name
            gemm_node_name = create_enhanced_name(
                node.name or "fusedmatmul_to_gemm",
                [A, B, bias_name],
                [output],
                node_index,
                "gemm"
            )
            
            # Create the Gemm node
            gemm_node = helper.make_node(
                "Gemm",
                inputs=[A, B, bias_name],
                outputs=[output],
                name=gemm_node_name,
                alpha=alpha,
                beta=1.0,
                transA=transA,
                transB=transB
            )
            
            new_nodes.append(gemm_node)
            
            print(f"  -> Created Gemm node: {gemm_node_name}")
            print(f"     Inputs: [{A}, {B}, {bias_name}]")
            print(f"     Output: [{output}]")
            print("")
            
        else:
            # Keep other nodes as they are, but ensure their names are unique
            if node.name:
                # Ensure node name is unique
                unique_node_name = generate_absolutely_unique_name(node.name, existing_names)
                if unique_node_name != node.name:
                    print(f"Renamed node: {node.name} -> {unique_node_name}")
                    node.name = unique_node_name
            
            new_nodes.append(node)
    
    # Create a new graph with updated nodes and initializers
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=list(model.graph.initializer) + new_initializers,
        value_info=model.graph.value_info
    )
    
    # Create a new model with the updated graph
    new_model = helper.make_model(
        new_graph,
        producer_name="EnhancedFusedMatMul2Gemm",
        opset_imports=model.opset_import,
        ir_version=model.ir_version
    )
    
    # Copy domain information for custom ops
    for domain in model.domain:
        new_model.domain.append(domain)
    
    # Copy model metadata
    new_model.metadata_props.extend(model.metadata_props)

    # Save the new model
    onnx.save(new_model, output_model_path)
    
    print(f"="*60)
    print(f"✅ Conversion Summary:")
    print(f"   - Converted {fusedmatmul_count} FusedMatMul nodes to Gemm")
    print(f"   - Added {len(new_initializers)} new bias tensors")
    print(f"   - Enhanced naming prevents conflicts")
    print(f"   - Saved to: {output_model_path}")
    print(f"="*60)
    
    return new_model



def convert_sum_to_add(input_model_path, output_model_path):
    """
    Convert Sum operators to Add operators in an ONNX model.
    Sum operator can take multiple inputs, while Add takes exactly two inputs.
    This function breaks down Sum operators with >2 inputs into a series of Add operators.
    
    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path to save the converted ONNX model
    """
    # Load the model
    model = onnx.load(input_model_path)
    
    # Track necessary changes
    new_nodes = []
    processed_nodes = set()
    
    # Process each node in the graph
    for i, node in enumerate(model.graph.node):
        # Skip already processed nodes
        if i in processed_nodes:
            continue
            
        # Check if the node is a Sum operator
        if node.op_type == "Sum":
            input_count = len(node.input)
            
            if input_count == 1:
                # Sum with one input is just an Identity
                identity_node = helper.make_node(
                    "Identity",
                    inputs=[node.input[0]],
                    outputs=node.output,
                    name=f"{node.name}_identity"
                )
                new_nodes.append(identity_node)
            
            elif input_count == 2:
                # Sum with two inputs can be directly converted to Add
                add_node = helper.make_node(
                    "Add",
                    inputs=[node.input[0], node.input[1]],
                    outputs=node.output,
                    name=f"{node.name}_add"
                )
                new_nodes.append(add_node)
                
            else:
                # Sum with more than two inputs needs to be broken down into a series of Add operations
                # We'll create intermediate outputs for all but the last Add
                intermediate_outputs = []
                
                for j in range(input_count - 1):
                    if j == 0:
                        # First Add takes the first two inputs of Sum
                        input1 = node.input[0]
                        input2 = node.input[1]
                    else:
                        # Subsequent Adds take the output of the previous Add and the next input
                        input1 = intermediate_outputs[-1]
                        input2 = node.input[j + 1]
                    
                    # For the last Add, use the original output, otherwise create an intermediate output
                    if j == input_count - 2:
                        output = node.output[0]
                    else:
                        output = f"{node.name}_intermediate_{j}"
                        intermediate_outputs.append(output)
                    
                    # Create the Add node
                    add_node = helper.make_node(
                        "Add",
                        inputs=[input1, input2],
                        outputs=[output],
                        name=f"{node.name}_add_{j}"
                    )
                    new_nodes.append(add_node)
            
            # Mark this node as processed
            processed_nodes.add(i)
        else:
            # Keep other nodes as they are
            new_nodes.append(node)
    
    # Create a new graph with updated nodes
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer,
        value_info=model.graph.value_info
    )
    
    # Create a new model with the updated graph
    # Preserve opset imports and other model metadata
    new_model = helper.make_model(
        new_graph,
        producer_name="SumToAddConverter",
        opset_imports=model.opset_import,
        ir_version=model.ir_version
    )
    
    # Copy domain information for custom ops
    for domain in model.domain:
        new_model.domain.append(domain)
    
    # Copy model metadata
    new_model.metadata_props.extend(model.metadata_props)
    
    # Save the new model
    onnx.save(new_model, output_model_path)
    print(f"Converted model saved to {output_model_path}")
    
    return new_model

def rename_softmaxgrad_op(input_model_path: str, output_model_path: str, 
                          old_op_name: str = "SoftmaxGrad_13", 
                          new_op_name: str = "SoftmaxGrad"):
    """
    Rename Microsoft's custom operator SoftmaxGrad_13 to SoftmaxGrad.
    
    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path to save the converted ONNX model
        old_op_name: Original operator name (default: "SoftmaxGrad_13")
        new_op_name: New operator name (default: "SoftmaxGrad")
    """
    model = onnx.load(input_model_path)
    
    modified_nodes = []
    modified_count = 0
    
    # Process each node in the graph
    for node in model.graph.node:
        # Check if the node is the target Microsoft domain operator
        if node.op_type == old_op_name and node.domain == "com.microsoft":
            modified_count += 1
            
            # Create a new node with the updated op_type
            new_node = helper.make_node(
                op_type=new_op_name,
                inputs=list(node.input),
                outputs=list(node.output),
                name=node.name,
                domain=node.domain  # Keep the original domain
            )
            
            # Copy all attributes from the original node
            for attr in node.attribute:
                new_node.attribute.append(attr)
            
            modified_nodes.append(new_node)
        else:
            # Keep all other nodes as they are
            modified_nodes.append(node)
    
    print(f"Modified {modified_count} {old_op_name} nodes to {new_op_name}")
    
    # Create a new graph with the modified nodes
    new_graph = helper.make_graph(
        nodes=modified_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer
    )
    
    # Copy over value_info from the original model
    for vi in model.graph.value_info:
        new_graph.value_info.append(vi)
    
    # Create a new model with the updated graph
    new_model = helper.make_model(
        new_graph,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
        doc_string=model.doc_string
    )
    
    # Copy over IR version and opset imports
    new_model.ir_version = model.ir_version
    new_model.opset_import.extend(model.opset_import)
    
    # Save the model
    onnx.save(new_model, output_model_path)
    print(f"Saved to {output_model_path}")
    
    return new_model

def remove_softmax_loss_outputs(input_model_path, output_model_path):
    """
    Remove loss outputs from SoftmaxCrossEntropyLoss nodes, keeping only the log probability output.
    
    Args:
        input_model_path (str): Path to the input ONNX model
        output_model_path (str): Path to save the modified ONNX model
    """
    import onnx
    
    # Load the model
    model = onnx.load(input_model_path)
    graph = model.graph
    
    # Find SoftmaxCrossEntropyLoss nodes
    target_nodes = []
    for node in graph.node:
        if node.op_type == 'SoftmaxCrossEntropyLoss':
            target_nodes.append(node)
    
    print(f"Found {len(target_nodes)} SoftmaxCrossEntropyLoss nodes")
    
    # Outputs to remove (first output - loss)
    outputs_to_remove = []
    
    # Create new nodes with modified outputs
    new_nodes = []
    for node in graph.node:
        if node.op_type == 'SoftmaxCrossEntropyLoss' and len(node.output) > 1:
            # Keep only the second output (log probabilities) and remove the first (loss)
            outputs_to_remove.append(node.output[0])
            
            # Create a new node with only the second output
            new_node = onnx.NodeProto()
            new_node.CopyFrom(node)
            log_prob_output = node.output[1]
            
            # Clear outputs and set only the log probability output
            del new_node.output[:]
            new_node.output.append(log_prob_output)
            
            new_nodes.append(new_node)
        else:
            # Keep other nodes unchanged
            new_nodes.append(node)
    
    # Replace all nodes with the new set
    del graph.node[:]
    graph.node.extend(new_nodes)
    
    # Filter graph outputs to remove loss outputs
    new_outputs = []
    for output in graph.output:
        if output.name not in outputs_to_remove:
            new_outputs.append(output)
    
    # Replace graph outputs with filtered list
    del graph.output[:]
    graph.output.extend(new_outputs)
    
    # Save the modified model
    onnx.save(model, output_model_path)
    print(f"Saved model with loss outputs removed to: {output_model_path}")

def remove_softmax_grad_loss_inputs(input_model_path, output_model_path):
    
    # Load the model
    model = onnx.load(input_model_path)
    graph = model.graph
    
    # Find SoftmaxCrossEntropyLossGrad nodes
    target_nodes = []
    for node in graph.node:
        if node.op_type == 'SoftmaxCrossEntropyLossGrad':
            target_nodes.append(node)
    
    print(f"Found {len(target_nodes)} SoftmaxCrossEntropyLossGrad nodes")
    
    # Inputs to remove (first input)
    inputs_to_remove = []
    
    # Create new nodes with modified inputs
    new_nodes = []
    for node in graph.node:
        if node.op_type == 'SoftmaxCrossEntropyLossGrad' and len(node.input) > 2:
            # Remove the first input and keep the rest
            first_input = node.input[0]
            inputs_to_remove.append(first_input)
            
            # Create a new node without the first input
            new_node = onnx.NodeProto()
            new_node.CopyFrom(node)
            
            # Keep only the second and third inputs
            remaining_inputs = list(node.input[1:])
            del new_node.input[:]
            new_node.input.extend(remaining_inputs)
            
            new_nodes.append(new_node)
        else:
            # Keep other nodes unchanged
            new_nodes.append(node)
    
    # Replace all nodes with the new set
    del graph.node[:]
    graph.node.extend(new_nodes)
    
    # Save the modified model
    onnx.save(model, output_model_path)
    print(f"Saved model with SoftmaxCrossEntropyLossGrad first input removed to: {output_model_path}")

def optimize_softmax_axis(input_model_path, output_model_path):
   
    model = onnx.load(input_model_path)
    
    # Track if we made any changes
    optimized = False
    
    # Create a map of value_info by name for easy access
    value_info_map = {vi.name: vi for vi in model.graph.value_info}
    value_info_map.update({vi.name: vi for vi in model.graph.input})
    value_info_map.update({vi.name: vi for vi in model.graph.output})
    
    # Function to get shape from value_info
    def get_shape(tensor_name):
        if tensor_name in value_info_map:
            shape = []
            for dim in value_info_map[tensor_name].type.tensor_type.shape.dim:
                if dim.dim_param:
                    # Handle symbolic dimensions (set to -1 for dynamic dimension)
                    shape.append(-1)
                else:
                    shape.append(dim.dim_value)
            return shape
        return None
    
    # Track the names of nodes to be removed
    nodes_to_remove = []
    
    # Track new nodes and value_infos to be added
    new_nodes = []
    new_value_infos = []
    
    # For each node in the graph
    for i, node in enumerate(model.graph.node):
        if node.op_type == "Softmax":
            # Get the input and output names
            input_name = node.input[0]
            output_name = node.output[0]
            
            # Get the axis attribute
            axis = None
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = attr.i
                    break
            
            # If axis is not set, it defaults to 1 in ONNX
            if axis is None:
                axis = 1
            
            # Get the input shape
            input_shape = get_shape(input_name)
            if input_shape is None:
                print(f"Warning: Could not determine shape for {input_name}, skipping optimization")
                continue
            
            # Check if all dimensions after axis are 1
            all_ones_after_axis = all(dim == 1 for dim in input_shape[axis+1:]) if axis+1 < len(input_shape) else True
            
            # Only optimize if the axis is not the last dimension and all subsequent dimensions are 1
            if axis != len(input_shape) - 1 and all_ones_after_axis and axis >= 0:
                print(f"Optimizing Softmax node with input shape {input_shape} and axis={axis}")
                
                # Create unique names for intermediate tensors
                reshape_before_output = f"{input_name}_reshaped_before_softmax"
                softmax_output = f"{output_name}_after_softmax"
                
                # Calculate new shapes
                # Move the axis dimension to the end and flatten all the 1s
                new_shape_before = []
                for i in range(len(input_shape)):
                    if i < axis:
                        new_shape_before.append(input_shape[i])
                    elif i == axis:
                        continue
                    elif i > axis:
                        continue
                new_shape_before.append(input_shape[axis])
                
                # Create reshape node before softmax
                reshape_before_node = helper.make_node(
                    "Reshape",
                    inputs=[input_name, f"{input_name}_shape_before"],
                    outputs=[reshape_before_output],
                    name=f"Reshape_before_softmax_{output_name}"
                )
                
                # Create initializer for the shape tensor
                shape_tensor_before = numpy_helper.from_array(
                    np.array(new_shape_before, dtype=np.int64),
                    name=f"{input_name}_shape_before"
                )
                
                # Create new softmax node with axis set to -1 (last dimension)
                new_softmax_node = helper.make_node(
                    "Softmax",
                    inputs=[reshape_before_output],
                    outputs=[softmax_output],
                    name=f"Softmax_optimized_{output_name}",
                    axis=-1  # Use -1 to always target the last dimension
                )
                
                # Create reshape node after softmax to restore original shape
                reshape_after_node = helper.make_node(
                    "Reshape",
                    inputs=[softmax_output, f"{output_name}_shape_after"],
                    outputs=[output_name],
                    name=f"Reshape_after_softmax_{output_name}"
                )
                
                # Create initializer for the shape tensor
                shape_tensor_after = numpy_helper.from_array(
                    np.array(input_shape, dtype=np.int64),
                    name=f"{output_name}_shape_after"
                )
                
                # Create value info for reshape_before_output
                reshape_before_vi = helper.make_tensor_value_info(
                    reshape_before_output,
                    value_info_map[input_name].type.tensor_type.elem_type,
                    new_shape_before
                )
                
                # Create value info for softmax_output
                softmax_output_vi = helper.make_tensor_value_info(
                    softmax_output,
                    value_info_map[output_name].type.tensor_type.elem_type,
                    new_shape_before  # Shape doesn't change after softmax
                )
                
                # Add all new nodes and value infos
                new_nodes.extend([reshape_before_node, new_softmax_node, reshape_after_node])
                new_value_infos.extend([reshape_before_vi, softmax_output_vi])
                model.graph.initializer.extend([shape_tensor_before, shape_tensor_after])
                
                # Mark the original node for removal
                nodes_to_remove.append(node)
                optimized = True
    
    # Remove the original nodes that were optimized
    for node in nodes_to_remove:
        model.graph.node.remove(node)
    
    # Add the new nodes and value infos
    model.graph.node.extend(new_nodes)
    model.graph.value_info.extend(new_value_infos)
    
    # Save the optimized model
    print(f"Saving optimized model to {output_model_path}")
    onnx.save(model, output_model_path)
    
    print(f"Optimization complete. Modified {len(nodes_to_remove)} Softmax nodes.")
    return optimized

def process_layernormgrad_nodes(model_path, output_path):
    """
    Process LayerNormalizationGrad nodes to:
    1. Keep only the data gradient as output
    2. Modify inputs to keep only: upstream gradient, input, weight, and bias
    
    Args:
        model_path: Path to the input ONNX model
        output_path: Path where the modified model will be saved
    """
    # Load the model
    model = onnx.load(model_path)
    
    # Process each node
    for i, node in enumerate(model.graph.node):
        if node.op_type == "LayerNormalizationGrad":
            # Get the original inputs and outputs
            inputs = list(node.input)
            outputs = list(node.output)
            
            if len(outputs) >= 1:
                # Keep only the first output (data gradient)
                data_grad = outputs[0]
                del node.output[:]
                node.output.append(data_grad)
            
            # Now modify the inputs
            # We need to find the bias input or create a connection for it
            # Assuming the bias is available somewhere in the graph
            
            # Find where the bias might be
            bias_name = None
            
            # Option 1: Try to find if bias exists as an initializer
            # Naming convention might be similar to weight name but with "bias" instead of "weight"
            if len(inputs) >= 3:
                weight_name = inputs[2]  # Assuming weight is the third input
                possible_bias_name = weight_name.replace("weight", "bias")
                
                # Check if this name exists in initializers
                for initializer in model.graph.initializer:
                    if initializer.name == possible_bias_name:
                        bias_name = possible_bias_name
                        break
            
            # Option 2: If bias name not found, look in other nodes' inputs
            if bias_name is None:
                for n in model.graph.node:
                    if n.op_type == "LayerNormalization":
                        # LayerNorm forward node typically has bias as input
                        if len(n.input) >= 3:
                            # Usually the third input to LayerNorm is bias
                            bias_name = n.input[2]
                            break
            
            # If bias was found, create new input list with only what we need
            if bias_name is not None:
                new_inputs = []
                
                # Keep upstream gradient and input
                if len(inputs) >= 2:
                    new_inputs.extend([inputs[0], inputs[1]])
                
                # Keep weight
                if len(inputs) >= 3:
                    new_inputs.append(inputs[2])
                
                # Add bias
                new_inputs.append(bias_name)
                
                # Replace inputs
                del node.input[:]
                node.input.extend(new_inputs)
            else:
                print(f"Warning: Bias input could not be found for node {node.name}")
    
    # Save the modified model
    onnx.save(model, output_path)
    print(f"Modified model saved to {output_path}")