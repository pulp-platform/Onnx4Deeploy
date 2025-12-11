import onnx
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper, shape_inference

def generate_averagepool_onnx_and_data(save_path=None):
    """ Generate ONNX model for AveragePool operator with test data """

    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")

    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_config = config.get("averagepool", {})
    input_shape = tuple(model_config.get("input_shape", [1, 16, 32, 32]))
    kernel_shape = model_config.get("kernel_shape", [2, 2])
    strides = model_config.get("strides", [2, 2])
    pads = model_config.get("pads", [0, 0, 0, 0])
    count_include_pad = model_config.get("count_include_pad", 0)
    ceil_mode = model_config.get("ceil_mode", 0)
    opset_version = model_config.get("opset_version", 13)

    print(f"Generating AveragePool model with:")
    print(f"  Input shape: {input_shape}")
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

    # Generate random input data
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # Save input data
    np.savez(input_file, input=input_data)
    print(f"✅ Input data saved to {input_file}")

    # Calculate output shape
    batch, channels, height, width = input_shape
    kernel_h, kernel_w = kernel_shape
    stride_h, stride_w = strides
    pad_top, pad_left, pad_bottom, pad_right = pads

    if ceil_mode:
        out_h = int(np.ceil((height + pad_top + pad_bottom - kernel_h) / stride_h)) + 1
        out_w = int(np.ceil((width + pad_left + pad_right - kernel_w) / stride_w)) + 1
    else:
        out_h = int(np.floor((height + pad_top + pad_bottom - kernel_h) / stride_h)) + 1
        out_w = int(np.floor((width + pad_left + pad_right - kernel_w) / stride_w)) + 1

    output_shape = (batch, channels, out_h, out_w)

    print(f"  Output shape: {output_shape}")

    # Define ONNX tensors for the model
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

    # Create AveragePool node
    averagepool_node = helper.make_node(
        "AveragePool",
        inputs=["input"],
        outputs=["output"],
        name="averagepool_node",
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,
        count_include_pad=count_include_pad,
        ceil_mode=ceil_mode
    )

    # Create graph with the AveragePool node
    graph_def = helper.make_graph(
        [averagepool_node],
        "averagepool_graph",
        [input_tensor],
        [output_tensor]
    )

    # Create model
    model_def = helper.make_model(
        graph_def,
        producer_name="averagepool_model",
        opset_imports=[helper.make_opsetid("", opset_version)]
    )

    # Apply shape inference
    model_def = shape_inference.infer_shapes(model_def)

    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")

    # Compute the expected output manually using numpy
    # Pad the input if needed
    if any(p > 0 for p in pads):
        padded_input = np.pad(
            input_data,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0
        )
    else:
        padded_input = input_data

    # Compute average pooling
    output_data = np.zeros(output_shape, dtype=np.float32)

    for b in range(batch):
        for c in range(channels):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride_h
                    w_start = w * stride_w
                    h_end = min(h_start + kernel_h, padded_input.shape[2])
                    w_end = min(w_start + kernel_w, padded_input.shape[3])

                    # Extract the pooling window
                    pool_region = padded_input[b, c, h_start:h_end, w_start:w_end]

                    if count_include_pad:
                        # Count all positions including padding
                        pool_size = kernel_h * kernel_w
                    else:
                        # Count only valid (non-padded) positions
                        pool_size = pool_region.size

                    # Average pooling
                    output_data[b, c, h, w] = np.sum(pool_region) / pool_size

    # Save output data
    np.savez(output_file, output=output_data)

    print(f"✅ Expected output data saved to {output_file}")
    print("Note: This script computes expected output manually.")


if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_averagepool_onnx_and_data(save_path)
