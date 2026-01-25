import onnx
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper


def generate_groupnorm_h2_onnx_and_data(save_path=None):
    """Generate ONNX model for GroupNormalization with H=2 (like EPIDENET ln1)"""

    # Config for H=2 test case
    input_shape = (1, 4, 2, 997)   # [N, C, H, W] - H=2 like EPIDENET ln1
    num_groups = 1                  # num_groups=1 (LayerNorm behavior)
    epsilon = 0.001
    opset_version = 13

    N, C, H, W = input_shape
    channels_per_group = C // num_groups

    print(f"Generating GroupNormalization model with H={H}:")
    print(f"  Input shape: {input_shape}")
    print(f"  Num groups: {num_groups}")
    print(f"  Channels per group: {channels_per_group}")
    print(f"  Output shape (Y): {input_shape}")
    print(f"  Stat shape: [{N}, {num_groups}, 2]")
    print(f"  Epsilon: {epsilon}")

    # Set default save path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = save_path if save_path else os.path.join(script_dir, "onnx")

    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")

    os.makedirs(base_path, exist_ok=True)

    # Generate random data with fixed seed for reproducibility
    np.random.seed(42)
    X = np.random.randn(*input_shape).astype(np.float32)
    gamma = np.random.randn(C).astype(np.float32)
    beta = np.random.randn(C).astype(np.float32)

    # Save input data
    np.savez(input_file, X=X, gamma=gamma, beta=beta)
    print(f"Input data saved to {input_file}")

    # Define ONNX input tensors
    X_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    gamma_tensor = helper.make_tensor_value_info("gamma", TensorProto.FLOAT, [C])
    beta_tensor = helper.make_tensor_value_info("beta", TensorProto.FLOAT, [C])

    # Define ONNX output tensors
    Y_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, input_shape)
    stat_tensor = helper.make_tensor_value_info("stat", TensorProto.FLOAT, [N, num_groups, 2])

    # Create GroupNormStats node (computes mean and inv_std)
    groupnorm_stat_node = helper.make_node(
        "GroupNormStats",
        inputs=["X"],
        outputs=["stat"],
        name="groupnorm_stat_node",
        epsilon=epsilon,
        num_groups=num_groups,
    )

    # Create GroupNormForward node (applies normalization)
    groupnorm_node = helper.make_node(
        "GroupNormForward",
        inputs=["X", "gamma", "beta", "stat"],
        outputs=["Y"],
        name="groupnorm_fwd_node",
        num_groups=num_groups,
    )

    # Create graph
    graph_def = helper.make_graph(
        [groupnorm_stat_node, groupnorm_node],
        "groupnorm_h2_graph",
        [X_tensor, gamma_tensor, beta_tensor],
        [Y_tensor, stat_tensor],
    )

    # Create model
    model_def = helper.make_model(
        graph_def,
        producer_name="groupnorm_h2_model",
        opset_imports=[helper.make_opsetid("", opset_version)],
    )

    # Manually set output shapes
    Y_output = model_def.graph.output[0]
    del Y_output.type.tensor_type.shape.dim[:]
    for dim in input_shape:
        Y_output.type.tensor_type.shape.dim.add().dim_value = dim

    stat_output = model_def.graph.output[1]
    del stat_output.type.tensor_type.shape.dim[:]
    stat_output.type.tensor_type.shape.dim.add().dim_value = N
    stat_output.type.tensor_type.shape.dim.add().dim_value = num_groups
    stat_output.type.tensor_type.shape.dim.add().dim_value = 2

    # Save ONNX model
    onnx.save(model_def, onnx_file)
    print(f"ONNX model saved to {onnx_file}")

    # Compute expected outputs using numpy
    Y, stat = compute_groupnorm_numpy(X, gamma, beta, num_groups, epsilon)

    # Save expected output
    np.savez(output_file, Y=Y, stat=stat)
    print(f"Expected output data saved to {output_file}")
    print(f"Y shape: {Y.shape}, stat shape: {stat.shape}")
    print(f"Y sample values: {Y.flat[:5]}")
    print(f"Stat values: mean={stat[0,0,0]:.6f}, inv_std={stat[0,0,1]:.6f}")


def compute_groupnorm_numpy(X, gamma, beta, num_groups, epsilon):
    """
    Compute GroupNormalization forward pass
    """
    N, C, H, W = X.shape
    channels_per_group = C // num_groups

    # Reshape for group computation
    X_reshaped = X.reshape(N, num_groups, channels_per_group, H, W)

    # Compute statistics per group
    mean = X_reshaped.mean(axis=(2, 3, 4), keepdims=True)
    var = X_reshaped.var(axis=(2, 3, 4), keepdims=True)
    std = np.sqrt(var + epsilon)
    inv_std = 1.0 / std

    # Normalize
    X_norm = (X_reshaped - mean) / std

    # Reshape back to [N, C, H, W]
    X_norm = X_norm.reshape(N, C, H, W)

    # Apply scale and shift
    gamma_reshaped = gamma.reshape(1, C, 1, 1)
    beta_reshaped = beta.reshape(1, C, 1, 1)

    Y = gamma_reshaped * X_norm + beta_reshaped

    # Create stat array [N, G, 2]
    mean_flat = mean.reshape(N, num_groups)
    inv_std_flat = inv_std.reshape(N, num_groups)
    stat = np.stack([mean_flat, inv_std_flat], axis=-1).astype(np.float32)

    return Y.astype(np.float32), stat


if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_groupnorm_h2_onnx_and_data(save_path)
