import onnx
import numpy as np
import yaml
import sys
import os
from onnx import TensorProto, helper


def generate_groupnormgradw_h2_epidenet_onnx_and_data(save_path=None):
    """Generate ONNX model for GroupNormGradW using EXACT EPIDENET layer_norm3 data"""

    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config.yaml")

    # Load configuration
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    model_config = config.get("groupnormgradw", {})
    input_shape = tuple(model_config.get("input_shape", [1, 16, 2, 20]))
    num_groups = model_config.get("num_groups", 1)
    epsilon = model_config.get("epsilon", 0.001)
    opset_version = model_config.get("opset_version", 13)

    N, C, H, W = input_shape
    print(f"Generating GroupNormGradW H2 EPIDENET model:")
    print(f"  Input shape: {input_shape}")
    print(f"  H = {H}")
    print(f"  Stat shape: [{N}, {num_groups}, 2]")

    # Set default save path
    base_path = save_path if save_path else os.path.join(script_dir, "onnx")
    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")
    os.makedirs(base_path, exist_ok=True)

    # ================================================================
    # PASTE THE DUMPED DATA FROM EPIDENET HERE
    # Run EPIDENET test and copy X_DATA, dY_DATA from the output
    # ================================================================

    # Placeholder - replace with actual dumped data
    # X_DATA will be dumped from EPIDENET Network.c
    # dY_DATA will be dumped from EPIDENET Network.c
    # stat values will be dumped from EPIDENET Network.c

    X_DATA = None  # Will be filled after running EPIDENET
    dY_DATA = None  # Will be filled after running EPIDENET
    stat_mean = -0.06101860  # From EPIDENET debug
    stat_inv_std = 2.40103400  # From EPIDENET debug

    if X_DATA is None or dY_DATA is None:
        print("ERROR: Please paste X_DATA and dY_DATA from EPIDENET debug output!")
        print("Run the EPIDENET test first and copy the dumped arrays here.")
        return

    # Create numpy arrays from the data
    X = np.array(X_DATA, dtype=np.float32).reshape(input_shape)
    dY = np.array(dY_DATA, dtype=np.float32).reshape(input_shape)

    # Combined stat: [N, G, 2]
    stat = np.array([[[stat_mean, stat_inv_std]]], dtype=np.float32)

    # Save inputs
    np.savez(input_file, dY=dY, X=X, stat=stat)
    print(f"Input data saved to {input_file}")

    # Define ONNX tensors
    dY_vi = helper.make_tensor_value_info("dY", TensorProto.FLOAT, input_shape)
    X_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    stat_vi = helper.make_tensor_value_info("stat", TensorProto.FLOAT, [N, num_groups, 2])
    dGamma_vi = helper.make_tensor_value_info("dGamma", TensorProto.FLOAT, [C])

    # Create GroupNormGradW node
    groupnormgradw_node = helper.make_node(
        "GroupNormGradW",
        inputs=["dY", "X", "stat"],
        outputs=["dGamma"],
        name="groupnormgradw_h2_epidenet_node",
        num_groups=num_groups,
    )

    # Create graph
    graph_def = helper.make_graph(
        [groupnormgradw_node],
        "groupnormgradw_h2_epidenet_graph",
        [dY_vi, X_vi, stat_vi],
        [dGamma_vi],
        value_info=[stat_vi]
    )

    # Create model
    model_def = helper.make_model(
        graph_def,
        producer_name="groupnormgradw_h2_epidenet_model",
        opset_imports=[helper.make_opsetid("", opset_version)],
    )

    # Save ONNX
    onnx.save(model_def, onnx_file)
    print(f"ONNX model saved to {onnx_file}")

    # Compute expected dGamma using numpy (same as C code should compute)
    dGamma = compute_groupnormgradw_numpy_combined(dY, X, stat, num_groups)

    # Save expected output
    np.savez(output_file, dGamma=dGamma)
    print(f"Expected output data saved to {output_file}")
    print(f"dGamma shape: {dGamma.shape}")
    print(f"dGamma values: {dGamma}")


def compute_groupnormgradw_numpy_combined(dY, X, stat, num_groups):
    """
    Numpy implementation for GradW using [N, G, 2] stat
    dGamma = sum over (N, H, W) of (dY * X_norm)
    """
    N, C, H, W = X.shape
    G = num_groups
    C_p_G = C // G

    # Parse stat
    mean = stat[:, :, 0]     # [N, G]
    inv_std = stat[:, :, 1]  # [N, G]

    # Reshape
    X_reshaped = X.reshape(N, G, C_p_G, H, W)
    dY_reshaped = dY.reshape(N, G, C_p_G, H, W)

    # Broadcast
    mean_b = mean.reshape(N, G, 1, 1, 1)
    inv_std_b = inv_std.reshape(N, G, 1, 1, 1)

    # X_norm = (X - mean) * inv_std
    X_norm = (X_reshaped - mean_b) * inv_std_b

    # dGamma = sum(dY * X_norm) over N, H, W
    # Result shape: [G, C_p_G], then flatten to [C]
    dGamma_reshaped = (dY_reshaped * X_norm).sum(axis=(0, 3, 4))
    dGamma = dGamma_reshaped.reshape(C).astype(np.float32)

    return dGamma


if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_groupnormgradw_h2_epidenet_onnx_and_data(save_path)
