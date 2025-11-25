import onnx
import onnxruntime as ort
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper, numpy_helper

def generate_gemm_onnx_and_data(save_path=None):
    """ Generate ONNX model for GEMM operator based on config, with optional save path """
    
    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")
    
    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config["gemm"]
        
    input_a_shape = tuple(model_config["input_a_shape"])
    input_b_shape = tuple(model_config["input_b_shape"])
    
    # Get GEMM attributes from config
    transA = model_config.get("transA", 0)
    transB = model_config.get("transB", 0)
    alpha = model_config.get("alpha", 1.0)
    beta = model_config.get("beta", 1.0)
    
    # Get constants configuration (default: all are variables)
    a_is_const = model_config.get("a_is_const", False)
    b_is_const = model_config.get("b_is_const", False)
    c_is_const = model_config.get("c_is_const", False)
    
    # NEW: Check if bias should be included
    use_bias = model_config.get("use_bias", True)  # Default to True for backward compatibility
    
    # Validate matrix multiplication compatibility for GEMM
    if len(input_a_shape) < 2 or len(input_b_shape) < 2:
        raise ValueError("Input shapes must have at least 2 dimensions")
    
    # Extract the last two dimensions for matrix multiplication
    a_last_two_dims = input_a_shape[-2:]
    b_last_two_dims = input_b_shape[-2:]
    
    # Calculate effective shapes after applying transpose
    # For transA=1: We swap the last two dimensions of A
    # For transB=1: We swap the last two dimensions of B
    a_effective_shape = (a_last_two_dims[1], a_last_two_dims[0]) if transA else a_last_two_dims
    b_effective_shape = (b_last_two_dims[1], b_last_two_dims[0]) if transB else b_last_two_dims
    
    # For matrix multiplication to work:
    # - A's cols (effective_shape[1]) must equal B's rows (effective_shape[0])
    if a_effective_shape[1] != b_effective_shape[0]:
        print(f"A effective shape: {a_effective_shape}, B effective shape: {b_effective_shape}")
        raise ValueError(f"Incompatible matrix dimensions for GEMM: {input_a_shape} (transA={transA}) and {input_b_shape} (transB={transB})")
    
    # Output dimensions:
    # - Rows come from A's effective rows
    # - Columns come from B's effective columns
    output_rows = a_effective_shape[0]
    output_cols = b_effective_shape[1]
    
    # Handle bias (C) input only if use_bias is True
    input_c_shape = None
    input_c = None
    
    if use_bias:
        # For GEMM, we also need a bias (C) input
        # Check if bias shape is provided in config
        if "input_c_shape" in model_config:
            input_c_shape = tuple(model_config["input_c_shape"])
            
            # Verify if C shape is compatible for broadcasting
            if len(input_c_shape) > len(input_a_shape):
                raise ValueError(f"C shape {input_c_shape} has too many dimensions compared to output shape")
            
            # If C has fewer dimensions than the output, verify compatibility for broadcasting
            if len(input_c_shape) < len(input_a_shape):
                # For broadcasting rules check
                # Last dimensions must match or be 1 for broadcast
                c_rev = input_c_shape[::-1]  # Reverse C shape for checking from right to left
                out_rev = (output_rows, output_cols)[::-1]  # Reverse output shape
                
                for i in range(min(len(c_rev), len(out_rev))):
                    if c_rev[i] != out_rev[i] and c_rev[i] != 1:
                        raise ValueError(f"C shape {input_c_shape} incompatible for broadcasting to output shape")
        else:
            # Default bias shape is [output_rows, output_cols]
            input_c_shape = (output_rows, output_cols)
        
        # Generate random bias data
        input_c = np.random.randn(*input_c_shape).astype(np.float32) * 0.1
    
    # Calculate final output shape, preserving batch dimensions if any
    batch_dims = input_a_shape[:-2]
    output_shape = batch_dims + (output_rows, output_cols)
    
    # Set default save path if not provided
    base_path = save_path if save_path else os.path.join(script_dir, "onnx")
    
    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")
    
    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)
    
    # Generate random input data with appropriate distribution
    # Using smaller standard deviation to avoid numerical issues
    input_a = np.random.randn(*input_a_shape).astype(np.float32) * 0.1
    input_b = np.random.randn(*input_b_shape).astype(np.float32) * 0.1
    
    # Save input data (conditionally include input_c)
    if use_bias:
        np.savez(input_file, input_a=input_a, input_b=input_b, input_c=input_c)
    else:
        np.savez(input_file, input_a=input_a, input_b=input_b)
    
    # Create the graph
    graph_inputs = []
    graph_initializers = []
    node_inputs = []
    
    # Handle input A (variable or constant)
    if a_is_const:
        # Create as initializer
        tensor_a = numpy_helper.from_array(input_a, name="input_a")
        graph_initializers.append(tensor_a)
        node_inputs.append("input_a")
        print(f"Input A configured as constant tensor")
    else:
        # Create as graph input
        input_tensor_a = helper.make_tensor_value_info("input_a", TensorProto.FLOAT, input_a_shape)
        graph_inputs.append(input_tensor_a)
        node_inputs.append("input_a")
        print(f"Input A configured as variable tensor")
    
    # Handle input B (variable or constant)
    if b_is_const:
        # Create as initializer
        tensor_b = numpy_helper.from_array(input_b, name="input_b")
        graph_initializers.append(tensor_b)
        node_inputs.append("input_b")
        print(f"Input B configured as constant tensor")
    else:
        # Create as graph input
        input_tensor_b = helper.make_tensor_value_info("input_b", TensorProto.FLOAT, input_b_shape)
        graph_inputs.append(input_tensor_b)
        node_inputs.append("input_b")
        print(f"Input B configured as variable tensor")
    
    # Handle input C (variable or constant) only if bias is used
    if use_bias:
        if c_is_const:
            # Create as initializer
            tensor_c = numpy_helper.from_array(input_c, name="input_c")
            graph_initializers.append(tensor_c)
            node_inputs.append("input_c")
            print(f"Input C configured as constant tensor")
        else:
            # Create as graph input
            input_tensor_c = helper.make_tensor_value_info("input_c", TensorProto.FLOAT, input_c_shape)
            graph_inputs.append(input_tensor_c)
            node_inputs.append("input_c")
            print(f"Input C configured as variable tensor")
    else:
        print(f"Bias (input C) disabled - generating GEMM without bias")
    
    # Create output tensor
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
    
    # Create GEMM node with attributes
    gemm_node = helper.make_node(
        "Gemm", 
        inputs=node_inputs, 
        outputs=["output"], 
        name="gemm_node",
        transA=transA,
        transB=transB,
        alpha=alpha,
        beta=beta
    )
    
    # Create ONNX graph and model
    graph_def = helper.make_graph(
        [gemm_node], 
        "gemm_graph", 
        graph_inputs,  # Only variable inputs are graph inputs
        [output_tensor],
        initializer=graph_initializers  # Add constant tensors as initializers
    )
    
    model_def = helper.make_model(
        graph_def, 
        producer_name="gemm_model", 
        opset_imports=[helper.make_opsetid("", 13)]
    )
    
    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")
    
    # Prepare inputs for inference
    feed_dict = {}
    if not a_is_const:
        feed_dict["input_a"] = input_a
    if not b_is_const:
        feed_dict["input_b"] = input_b
    if use_bias and not c_is_const:
        feed_dict["input_c"] = input_c
    
    # Run inference using ONNX Runtime to validate the model
    try:
        ort_session = ort.InferenceSession(onnx_file)
        output_data = ort_session.run(None, feed_dict)[0]
        
        # Save output data
        np.savez(output_file, output=output_data)
        print(f"✅ Output data saved to {output_file}")
        
        # Verify result with numpy to double-check
        a_matrix = input_a.transpose(-2, -1) if transA else input_a
        b_matrix = input_b.transpose(-2, -1) if transB else input_b
        
        # Calculate expected output
        matmul_result = alpha * np.matmul(a_matrix, b_matrix)
        
        if use_bias:
            # For C, we need to handle broadcasting correctly
            if len(input_c_shape) < len(output_shape):
                # Determine broadcast dimensions
                pad_dims = len(output_shape) - len(input_c_shape)
                broadcast_shape = tuple([1] * pad_dims) + input_c_shape
                reshaped_c = input_c.reshape(broadcast_shape)
                
                # Create broadcasted C with same shape as output
                broadcast_c = np.broadcast_to(reshaped_c, output_shape)
                expected_output = matmul_result + beta * broadcast_c
            else:
                expected_output = matmul_result + beta * input_c
        else:
            # No bias case
            expected_output = matmul_result
        
        # Check if results match
        max_diff = np.max(np.abs(output_data - expected_output))
        print(f"Maximum difference between ONNX and NumPy computation: {max_diff}")
        if max_diff > 1e-5:
            print("⚠️ Warning: Significant difference between ONNX and NumPy results!")
        
    except Exception as e:
        print(f"⚠️ Error during inference: {e}")
        # Even if inference fails, we still created the model
    
    # Print model details
    print(f"Model details:")
    print(f"  Input A shape: {input_a_shape} ({'constant' if a_is_const else 'variable'})")
    print(f"  Input B shape: {input_b_shape} ({'constant' if b_is_const else 'variable'})")
    if use_bias:
        print(f"  Input C shape: {input_c_shape} ({'constant' if c_is_const else 'variable'})")
    else:
        print(f"  Input C: None (bias disabled)")
    print(f"  Output shape: {output_shape}")
    print(f"  Effective A shape (after transpose): {a_effective_shape}")
    print(f"  Effective B shape (after transpose): {b_effective_shape}")
    print(f"  transA: {transA}")
    print(f"  transB: {transB}")
    print(f"  alpha: {alpha}")
    print(f"  beta: {beta}")
    print(f"  use_bias: {use_bias}")
    
    return onnx_file, input_file, output_file

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_gemm_onnx_and_data(save_path)