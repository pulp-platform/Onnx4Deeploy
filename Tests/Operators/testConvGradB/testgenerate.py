import onnx
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper

def generate_convgradb_onnx_and_data(save_path=None):
    """ Generate ONNX model for ConvGradB (bias gradient) operator with test data """

    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")

    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_config = config.get("convgradb", {})
    output_grad_shape = tuple(model_config.get("output_grad_shape", [1, 2, 5, 5]))
    bias_shape = tuple(model_config.get("bias_shape", [2]))  # Bias shape is [C_out]
    opset_version = model_config.get("opset_version", 13)

    print(f"Generating ConvGradB model with:")
    print(f"  Output gradient shape (NCHW): {output_grad_shape}")
    print(f"  Bias gradient shape: {bias_shape}")

    # Set default save path if not provided
    base_path = save_path if save_path else os.path.join(script_dir, "onnx")

    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")

    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)

    # Generate random data
    output_grad = np.random.randn(*output_grad_shape).astype(np.float32)

    # Save input (only output_grad for ConvGradB)
    np.savez(input_file, output_grad=output_grad)
    print(f"✅ Input data saved to {input_file}")

    # Define ONNX tensors
    output_grad_tensor = helper.make_tensor_value_info("output_grad", TensorProto.FLOAT, output_grad_shape)
    bias_grad_tensor = helper.make_tensor_value_info("bias_grad", TensorProto.FLOAT, bias_shape)

    # Create ConvGradB node
    convgradb_node = helper.make_node(
        "ConvGradB",
        inputs=["output_grad"],
        outputs=["bias_grad"],
        name="convgradb_node"
    )

    # Create graph with ONE input
    graph_def = helper.make_graph(
        [convgradb_node],
        "convgradb_graph",
        [output_grad_tensor],
        [bias_grad_tensor]
    )

    model_def = helper.make_model(
        graph_def,
        producer_name="convgradb_model",
        opset_imports=[helper.make_opsetid("", opset_version)]
    )

    # Manually set output shape
    output_tensor = model_def.graph.output[0]
    del output_tensor.type.tensor_type.shape.dim[:]
    for dim in bias_shape:
        output_tensor.type.tensor_type.shape.dim.add().dim_value = dim
    print("✅ Manually set output shape for ConvGradB")

    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")

    # Compute expected bias gradient
    # For bias gradient: sum over batch, height, and width dimensions
    # bias_grad[c] = sum over (batch, h, w) of output_grad[batch, c, h, w]
    batch, c_out, out_h, out_w = output_grad_shape
    
    bias_grad = np.zeros(bias_shape, dtype=np.float32)
    
    for c in range(c_out):
        grad_sum = 0.0
        for b in range(batch):
            for h in range(out_h):
                for w in range(out_w):
                    grad_sum += output_grad[b, c, h, w]
        bias_grad[c] = grad_sum

    np.savez(output_file, bias_grad=bias_grad)
    print(f"✅ Expected output data saved to {output_file}")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_convgradb_onnx_and_data(save_path)
