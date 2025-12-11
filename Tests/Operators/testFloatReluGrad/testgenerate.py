import onnx
import onnxruntime as ort
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper

def generate_relugrad_onnx_and_data(save_path=None):
    """ Generate ONNX model for ReluGrad operator based on config, with optional save path """
    
    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")
    
    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config["relugrad"]
        
    input_shape = tuple(model_config["input_shape"])
    
    # Set default save path if not provided
    base_path = save_path if save_path else os.path.join(script_dir, "onnx")
    
    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")
    
    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)
    
    # Generate random input data
    # 1. Gradient from next layer (dY)
    grad_output = np.random.randn(*input_shape).astype(np.float32)
    
    # 2. Original input to ReLU (X) - with some negative and positive values
    original_input = np.random.uniform(-1, 1, input_shape).astype(np.float32)
    
    # Save input data
    np.savez(input_file, 
             grad_output=grad_output, 
             original_input=original_input)
    
    # Define ONNX tensors
    input_tensor_grad = helper.make_tensor_value_info("grad_output", TensorProto.FLOAT, input_shape)
    input_tensor_orig = helper.make_tensor_value_info("original_input", TensorProto.FLOAT, input_shape)
    
    output_tensor_grad = helper.make_tensor_value_info("grad_input", TensorProto.FLOAT, input_shape)
    
    # Create ONNX computation graph with ReluGrad node
    # Note: ONNX doesn't have a built-in ReluGrad operator, using custom domain
    relu_grad_node = helper.make_node(
        "ReluGrad", 
        inputs=["grad_output", "original_input"], 
        outputs=["grad_input"], 
        name="relu_grad_node",
        domain="com.microsoft"  # Custom domain for gradient operators
    )
    
    graph_def = helper.make_graph(
        [relu_grad_node], 
        "relugrad_graph", 
        [input_tensor_grad, input_tensor_orig], 
        [output_tensor_grad]
    )
    
    # Create custom opset for Microsoft domain
    opsets = [
        helper.make_opsetid("", 13),  # Standard ONNX opset
        helper.make_opsetid("com.microsoft", 1)  # Custom domain opset
    ]
    
    model_def = helper.make_model(graph_def, producer_name="relugrad_model", opset_imports=opsets)
    
    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")
    
    # Compute expected output using numpy
    # ReluGrad logic: dX = dY if X > 0 else 0
    expected_output = np.where(original_input > 0, grad_output, 0).astype(np.float32)
    
    # Save expected output
    np.savez(output_file, grad_input=expected_output)
    print(f"✅ Expected output data saved to {output_file}")
    
    print("⚠️ Note: Running this model requires a runtime with custom op support for ReluGrad.")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_relugrad_onnx_and_data(save_path)
