import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
import torch
import yaml
from epidenet_model.epidenet import *

# Add CCT utils to path
cct_utils_path = os.path.join(os.path.dirname(__file__), '..', 'CCT')
sys.path.insert(0, cct_utils_path)
from utils.utils import randomize_onnx_initializers, rename_and_save_onnx, unify_gemm_input_dims


def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    epidenet_config = config["epidenet"]
    training_config = config["training"]

    return (
        epidenet_config["pretrained"],
        epidenet_config["C"],
        epidenet_config["T"],
        epidenet_config["N"],
        epidenet_config["p_dropout"],
        epidenet_config["batch_size"],
        epidenet_config["opset_version"],
    )


def generate_epidenet_onnx_and_data(save_path=None):
    """Generate ONNX model for EpiDeNet EOG classification based on config"""

    pretrained, C, T, N, p_dropout, batch_size, opset_version = load_config()
    print(f"✅ Loaded config: C={C}, T={T}, N={N}, opset_version={opset_version}")

    # For EOG task, input shape is (batch_size, 1, C, T)
    # batch_size=1, 1 channel wrapper, C EOG channels, T time samples
    input_shape = (1, 1, C, T)

    folder_name = f"EPIDENET_EOG_C{C}_T{T}_N{N}"

    base_path = save_path if save_path else os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx", folder_name)

    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")

    os.makedirs(base_path, exist_ok=True)

    # Create deployment model (BatchNorm fused, no dropout)
    # This model is optimized for on-chip deployment
    from epidenet_model.epidenet import EpiDeNetDeploy, EpiDeNet

    # Option 1: Load pretrained model and fuse BatchNorm
    if pretrained and pretrained is not False:
        print(f"📦 Loading pretrained model from {pretrained}...")
        try:
            # Load training model with BatchNorm
            training_model = EpiDeNet(
                C=C,
                T=T,
                output_classes=N,
                p_dropout=p_dropout,
            )

            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, dict) and 'net' in checkpoint:
                training_model.load_state_dict(checkpoint['net'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                training_model.load_state_dict(checkpoint['state_dict'])
            else:
                training_model.load_state_dict(checkpoint)

            training_model.eval()

            # Create deployment model and fuse BatchNorm
            model = EpiDeNetDeploy(
                C=C,
                T=T,
                output_classes=N,
            )
            model.load_and_fuse_from_training_model(training_model)
            print("✅ BatchNorm fused into Conv layers for deployment")

        except Exception as e:
            print(f"⚠️  Could not load pretrained model: {e}")
            print("   Using random weights instead...")
            model = EpiDeNetDeploy(
                C=C,
                T=T,
                output_classes=N,
            )
    else:
        # Option 2: Create model with random weights
        print("📦 Creating deployment model with random weights...")
        model = EpiDeNetDeploy(
            C=C,
            T=T,
            output_classes=N,
        )

    model.eval()

    # Generate random input data for EOG signals
    input_data = np.random.randn(*input_shape).astype(np.float32)
    np.savez(input_file, input=input_data)

    input_tensor = torch.tensor(input_data)

    # Export to ONNX (fixed batch size for cleaner graph)
    torch.onnx.export(
        model,
        input_tensor,
        onnx_file,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        # Remove dynamic_axes for cleaner ONNX graph
    )

    onnx_model = onnx.load(onnx_file)
    onnx_model = randomize_onnx_initializers(onnx_model)
    print(f"✅ ONNX model saved to {onnx_file}")
    rename_and_save_onnx(onnx_file, onnx_file)

    # Simplify ONNX model to remove auxiliary nodes
    print(f"🔧 Simplifying ONNX model...")
    import onnxsim
    onnx_model_simplified, check = onnxsim.simplify(onnx_file)
    onnx.save(onnx_model_simplified, onnx_file)
    print(f"✅ ONNX model simplified and saved to {onnx_file}")

    # Run inference to generate output data
    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(None, {"input": input_data})[0]

    np.savez(output_file, output=output_data)
    print(f"✅ Output data saved to {output_file}")

    # Unify GEMM input dimensions
    unify_gemm_input_dims(onnx_file, onnx_file)
    print(f"✅ Successfully unified GEMM input dimensions. Saved as {onnx_file}")


if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_epidenet_onnx_and_data(save_path)
