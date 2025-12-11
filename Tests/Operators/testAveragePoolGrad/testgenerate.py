import onnx
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper, shape_inference

def generate_averagepoolgrad_onnx_and_data(save_path=None):
    """ Generate ONNX model for AveragePoolGrad operator with test data """

    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")

    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_config = config.get("averagepoolgrad", {})
    input_grad_shape = tuple(model_config.get("input_grad_shape", [1, 16, 16, 16]))
    original_input_shape = tuple(model_config.get("original_input_shape", [1, 16, 32, 32]))
    kernel_shape = model_config.get("kernel_shape", [2, 2])
    strides = model_config.get("strides", [2, 2])
    pads = model_config.get("pads", [0, 0, 0, 0])
    count_include_pad = model_config.get("count_include_pad", 0)
    ceil_mode = model_config.get("ceil_mode", 0)
    opset_version = model_config.get("opset_version", 13)

    print(f"Generating AveragePoolGrad model with:")
    print(f"  Input gradient shape: {input_grad_shape}")
    print(f"  Original input shape: {original_input_shape}")
    print(f"  Kernel shape: {kernel_shape}")
    print(f"  Strides: {strides}")
    print(f"  Pads: {pads}")
    print(f"  Count include pad: {count_include_pad}")
    print(f"  Ceil mode: {ceil_mode}")

    # Set default save path if not provided
    base_path = save_path if save_path else os.path.join(script_dir, "onnx")

    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")

    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)

    # Generate random gradient data (gradient from downstream layer)
    input_grad = np.random.randn(*input_grad_shape).astype(np.float32)

    # Save input data
    np.savez(input_file, input_grad=input_grad)
    print(f"✅ Input data saved to {input_file}")

    # Output shape is the original input shape
    output_shape = original_input_shape

    print(f"  Output gradient shape: {output_shape}")

    # Define ONNX tensors for the model
    input_grad_tensor = helper.make_tensor_value_info("input_grad", TensorProto.FLOAT, input_grad_shape)
    output_grad_tensor = helper.make_tensor_value_info("output_grad", TensorProto.FLOAT, output_shape)

    # Create AveragePoolGrad node with original_input_shape as second input
    averagepoolgrad_node = helper.make_node(
        "AveragePoolGrad",
        inputs=["input_grad"],
        outputs=["output_grad"],
        name="averagepoolgrad_node",
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,
        count_include_pad=count_include_pad,
        ceil_mode=ceil_mode
    )

    # Create graph with the nodes
    graph_def = helper.make_graph(
        [averagepoolgrad_node],
        "averagepoolgrad_graph",
        [input_grad_tensor],
        [output_grad_tensor]
    )

    # Create model with custom domain for backward operators
    model_def = helper.make_model(
        graph_def,
        producer_name="averagepoolgrad_model",
        opset_imports=[helper.make_opsetid("", opset_version)]
    )

    # Manually set output shape (skip standard shape inference for custom op)
    output_tensor = model_def.graph.output[0]
    # Clear existing dimensions first
    del output_tensor.type.tensor_type.shape.dim[:]
    # Add dimensions with correct values
    for dim in output_shape:
        output_tensor.type.tensor_type.shape.dim.add().dim_value = dim
    print("✅ Manually set output shape for AveragePoolGrad")

    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")

    # Compute the expected output gradient manually using numpy
    batch, channels, out_h, out_w = input_grad_shape
    _, _, in_h, in_w = original_input_shape
    kernel_h, kernel_w = kernel_shape
    stride_h, stride_w = strides
    pad_top, pad_left, pad_bottom, pad_right = pads

    # Initialize output gradient with zeros
    output_grad = np.zeros(original_input_shape, dtype=np.float32)

    # Distribute the gradient back to the input
    for b in range(batch):
        for c in range(channels):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride_h
                    w_start = w * stride_w
                    h_end = min(h_start + kernel_h, in_h + pad_top + pad_bottom)
                    w_end = min(w_start + kernel_w, in_w + pad_left + pad_right)

                    # Calculate actual pool size
                    if count_include_pad:
                        pool_size = kernel_h * kernel_w
                    else:
                        # Calculate actual valid region size
                        actual_h_start = max(0, h_start - pad_top)
                        actual_w_start = max(0, w_start - pad_left)
                        actual_h_end = min(in_h, h_end - pad_top)
                        actual_w_end = min(in_w, w_end - pad_left)
                        pool_size = max(1, (actual_h_end - actual_h_start) * (actual_w_end - actual_w_start))

                    # Distribute gradient equally to all positions in the pooling window
                    grad_per_position = input_grad[b, c, h, w] / pool_size

                    # Add to output gradient (accounting for padding)
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            in_h_idx = h_start + kh - pad_top
                            in_w_idx = w_start + kw - pad_left

                            # Only update if within valid input bounds
                            if 0 <= in_h_idx < in_h and 0 <= in_w_idx < in_w:
                                output_grad[b, c, in_h_idx, in_w_idx] += grad_per_position

    # Save output data
    np.savez(output_file, output_grad=output_grad)

    print(f"✅ Expected output data saved to {output_file}")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_averagepoolgrad_onnx_and_data(save_path)
