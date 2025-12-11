import onnx
import onnxruntime as ort
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper


def generate_split_onnx_and_data(save_path=None):
    """Generate ONNX model for Split operator based on config, with optional save path"""

    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")

    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_config = config["split"]

    input_shape = tuple(model_config["input_shape"])
    axis = model_config["axis"]
    num_outputs = model_config["num_outputs"]

    # Set default save path if not provided
    base_path = save_path if save_path else os.path.join(script_dir, "onnx")

    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")

    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)

    # Generate random input data and save it directly
    input_data = np.random.randn(*input_shape).astype(np.float32)
    np.savez(input_file, input=input_data)

    # Calculate output shapes
    # For even split, each output has input_shape[axis] / num_outputs along the split axis
    split_size = input_shape[axis] // num_outputs
    output_shapes = []
    for i in range(num_outputs):
        output_shape = list(input_shape)
        output_shape[axis] = split_size
        output_shapes.append(tuple(output_shape))

    # Define ONNX tensors
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)

    # Create output tensor infos
    output_tensors = []
    output_names = []
    for i in range(num_outputs):
        output_name = f"output_{i}"
        output_names.append(output_name)
        output_tensors.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shapes[i]))

    # Create Split node
    # For ONNX opset 13+, Split requires 'split' attribute or input
    # We'll use equal splits for simplicity
    split_node = helper.make_node(
        "Split",
        inputs=["input"],
        outputs=output_names,
        axis=axis,
        name="split_node"
    )

    # Create ONNX computation graph
    graph_def = helper.make_graph([split_node], "split_graph", [input_tensor], output_tensors)
    model_def = helper.make_model(graph_def, producer_name="split_model", opset_imports=[helper.make_opsetid("", 13)])

    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"✅ ONNX model saved to {onnx_file}")
    print(f"   Input shape: {input_shape}")
    print(f"   Split axis: {axis}")
    print(f"   Number of outputs: {num_outputs}")
    for i, shape in enumerate(output_shapes):
        print(f"   Output {i} shape: {shape}")

    # Run inference using ONNX Runtime
    ort_session = ort.InferenceSession(onnx_file)
    outputs = ort_session.run(None, {"input": input_data})

    # Save output data
    output_dict = {}
    for i, output in enumerate(outputs):
        output_dict[f"output_{i}"] = output
        print(f"   Output {i} actual shape: {output.shape}")

    np.savez(output_file, **output_dict)
    print(f"✅ Output data saved to {output_file}")


if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_split_onnx_and_data(save_path)
