import onnx
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper


def generate_groupnormgradb_onnx_and_data(save_path=None):
    """Generate ONNX model for GroupNormGradB operator with test data"""

    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")

    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_config = config.get("groupnormgradb", {})
    input_shape = tuple(model_config.get("input_shape", [1, 8, 1, 64]))
    num_groups = model_config.get("num_groups", 1)
    epsilon = model_config.get("epsilon", 0.001)
    opset_version = model_config.get("opset_version", 13)

    N, C, H, W = input_shape

    print(f"Generating GroupNormGradB model with:")
    print(f"  Input shape (dY): {input_shape}")
    print(f"  Num groups: {num_groups}")
    print(f"  Output shape (dBeta): [{C}]")
    print(f"  Epsilon: {epsilon}")

    # Set default save path if not provided
    base_path = save_path if save_path else os.path.join(script_dir, "onnx")

    # Define standard filenames
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")

    # Ensure the save directory exists
    os.makedirs(base_path, exist_ok=True)

    # Generate random data
    # dY: upstream gradient (same shape as input)
    # GroupNormGradB only needs dY to compute dBeta
    dY = np.random.randn(*input_shape).astype(np.float32)

    # Save input data
    np.savez(input_file, dY=dY)
    print(f"Input data saved to {input_file}")

    # Define ONNX tensors
    dY_tensor = helper.make_tensor_value_info("dY", TensorProto.FLOAT, input_shape)
    dBeta_tensor = helper.make_tensor_value_info("dBeta", TensorProto.FLOAT, [C])

    # Create GroupNormGradB node
    groupnormgradb_node = helper.make_node(
        "GroupNormGradB",
        inputs=["dY"],
        outputs=["dBeta"],
        name="groupnormgradb_node",
        epsilon=epsilon,
        num_groups=num_groups,
    )

    # Create graph
    graph_def = helper.make_graph(
        [groupnormgradb_node],
        "groupnormgradb_graph",
        [dY_tensor],
        [dBeta_tensor],
    )

    # Create model
    model_def = helper.make_model(
        graph_def,
        producer_name="groupnormgradb_model",
        opset_imports=[helper.make_opsetid("", opset_version)],
    )

    # Manually set output shape
    output_tensor = model_def.graph.output[0]
    del output_tensor.type.tensor_type.shape.dim[:]
    output_tensor.type.tensor_type.shape.dim.add().dim_value = C

    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"ONNX model saved to {onnx_file}")

    # Compute expected dBeta using numpy
    dBeta = compute_groupnormgradb_numpy(dY)

    # Save expected output
    np.savez(output_file, dBeta=dBeta)
    print(f"Expected output data saved to {output_file}")


def compute_groupnormgradb_numpy(dY):
    """
    Compute GroupNormGradB: gradient w.r.t. beta (bias parameter)

    GroupNorm: Y = gamma * X_norm + beta

    dBeta = sum over (N, H, W) of dY
    For each channel c: dBeta[c] = sum_{n,h,w} dY[n,c,h,w]

    This is simply a reduction sum over all dimensions except the channel dimension.
    """
    N, C, H, W = dY.shape

    # Sum over batch, height, and width dimensions
    # dBeta[c] = sum_{n,h,w} dY[n,c,h,w]
    dBeta = dY.sum(axis=(0, 2, 3)).astype(np.float32)

    return dBeta


if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_groupnormgradb_onnx_and_data(save_path)
