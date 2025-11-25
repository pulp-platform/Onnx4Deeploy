import onnx
import onnxruntime as ort
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper

def generate_layernormgrad_onnx_and_data(save_path=None):
    """ Generate ONNX model for LayerNormalizationGrad operator based on config, with optional save path """
    
    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")
    
    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config["layernormgrad"]
        
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
    # 1. Gradient from next layer
    grad_in = np.random.randn(*input_shape).astype(np.float32)
    
    # 2. Original input to LayerNorm
    original_input = np.random.randn(*input_shape).astype(np.float32)
    
    # 3. LayerNorm weight and bias parameters
    # Assuming last dimension is normalized based on axis=-1
    last_dim = input_shape[-1]
    weight = np.random.randn(last_dim).astype(np.float32)  # gamma
    bias = np.random.randn(last_dim).astype(np.float32)    # beta
    
    # Save input data
    np.savez(input_file, 
             grad_in=grad_in, 
             original_input=original_input, 
             weight=weight, 
             bias=bias)
    
    # Define ONNX tensors
    input_tensor_grad = helper.make_tensor_value_info("grad_in", TensorProto.FLOAT, input_shape)
    input_tensor_orig = helper.make_tensor_value_info("original_input", TensorProto.FLOAT, input_shape)
    input_tensor_weight = helper.make_tensor_value_info("weight", TensorProto.FLOAT, [last_dim])
    input_tensor_bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [last_dim])
    
    output_tensor_grad = helper.make_tensor_value_info("output_grad", TensorProto.FLOAT, input_shape)
    
    # Create ONNX computation graph with LayerNormalizationGrad node
    # Note: Since ONNX doesn't have a built-in LayerNormGrad operator, we'll use a custom domain
    layernorm_grad_node = helper.make_node(
        "LayerNormalizationGrad", 
        inputs=["grad_in", "original_input", "weight", "bias"], 
        outputs=["output_grad"], 
        name="layernorm_grad_node",
        axis=-1,
        epsilon=0.00000999999974737852,
        stash_type=1,
        domain="com.microsoft"  # This matches what's in your image
    )
    
    graph_def = helper.make_graph(
        [layernorm_grad_node], 
        "layernormgrad_graph", 
        [input_tensor_grad, input_tensor_orig, input_tensor_weight, input_tensor_bias], 
        [output_tensor_grad]
    )
    
    # Create custom opset for Microsoft domain
    opsets = [
        helper.make_opsetid("", 13),  # Standard ONNX opset
        helper.make_opsetid("com.microsoft", 1)  # Custom domain opset
    ]
    
    model_def = helper.make_model(graph_def, producer_name="layernormgrad_model", opset_imports=opsets)
    
    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")
    
    # Compute expected output using numpy (simplified implementation)
    # Note: This is a simplified implementation for testing purposes only
    def layernorm_grad_numpy(grad_in, original_input, weight, bias, axis=-1, epsilon=1e-5):
        # Calculate mean and variance used in the forward pass
        mean = np.mean(original_input, axis=axis, keepdims=True)
        var = np.var(original_input, axis=axis, keepdims=True)
        std = np.sqrt(var + epsilon)
        
        # Normalized input from forward pass
        normalized = (original_input - mean) / std
        
        # Calculate gradients for weight and bias (not returned but needed for computation)
        dweight = np.sum(grad_in * normalized, axis=tuple(range(grad_in.ndim-1)))
        dbias = np.sum(grad_in, axis=tuple(range(grad_in.ndim-1)))
        
        # Gradient for normalized input
        dnormalized = grad_in * weight
        
        # Gradient for original input - this is the complex part
        N = original_input.shape[axis]
        dx_norm = dnormalized
        
        # First component: direct gradient through normalization
        dx = dx_norm / std
        
        # Second component: gradient through the mean
        dx_mean = -np.sum(dx_norm, axis=axis, keepdims=True) / std
        dx = dx + dx_mean / N
        
        # Third component: gradient through the variance
        dx_var = -0.5 * np.sum(dx_norm * (original_input - mean), axis=axis, keepdims=True) * (var + epsilon)**(-1.5)
        dx = dx + dx_var * 2 * (original_input - mean) / N
        
        return dx
    
    # Compute expected output
    expected_output = layernorm_grad_numpy(
        grad_in, 
        original_input, 
        weight, 
        bias, 
        axis=-1, 
        epsilon=0.00000999999974737852
    )
    
    # Save expected output
    np.savez(output_file, output_grad=expected_output)
    print(f"✅ Expected output data saved to {output_file}")
    
    # Note: We can't actually run this with normal ONNX Runtime as it doesn't support LayerNormGrad
    # You would need a custom runtime or modify this code to use your specific implementation
    print("⚠️ Note: Running this model requires a runtime with custom op support.")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_layernormgrad_onnx_and_data(save_path)