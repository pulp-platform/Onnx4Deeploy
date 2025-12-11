import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
import torch
import yaml
from mi_bminet_model.mi_bminet import *
from utils.utils import *


def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mi_bminet_config = config["mi_bminet"]
    training_config = config["training"]

    return (
        mi_bminet_config["pretrained"],
        mi_bminet_config["F1"],
        mi_bminet_config["D"],
        mi_bminet_config["F2"],
        mi_bminet_config["C"],
        mi_bminet_config["T"],
        mi_bminet_config["N"],
        mi_bminet_config["Nf"],
        mi_bminet_config["Nf2"],
        mi_bminet_config["p_dropout"],
        mi_bminet_config["activation"],
        mi_bminet_config["seed"],
        mi_bminet_config["dropout_type"],
        mi_bminet_config["fold_id"],
        mi_bminet_config["batch_size"],
        mi_bminet_config["opset_version"],
    )


def generate_mi_bminet_onnx_and_data(save_path=None):
    """Generate ONNX model for MI-BMInet BCI based on config, with optional save path"""

    pretrained, F1, D, F2, C, T, N, Nf, Nf2, p_dropout, activation, seed, dropout_type, fold_id, batch_size, opset_version = load_config()
    print(f"✅ Loaded config: F1={F1}, D={D}, C={C}, T={T}, N={N}, Nf={Nf}, Nf2={Nf2}, opset_version={opset_version}")

    # For BCI task, input shape is (batch_size, 1, C, T)
    # batch_size=1, 1 channel (EEG), C channels, T time samples
    input_shape = (1, 1, C, T)

    folder_name = f"MI_BMInet_BCI_F1{F1}_D{D}_C{C}_T{T}_N{N}"

    base_path = save_path if save_path else os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx", folder_name)

    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")

    os.makedirs(base_path, exist_ok=True)

    # Create deployment model (no BatchNorm, no dropout, no padding)
    # This model is optimized for on-chip deployment with fused BatchNorm
    from mi_bminet_model.mi_bminet import MIBMINetDeploy, MIBMINet

    # Option 1: Load pretrained model and fuse BatchNorm
    if pretrained and pretrained is not False:
        print(f"📦 Loading pretrained model from {pretrained}...")
        try:
            # Load training model with BatchNorm
            training_model = MIBMINet(
                F1=F1,
                D=D,
                F2=F2,
                C=C,
                T=T,
                N=N,
                Nf=Nf,
                Nf2=Nf2,
                p_dropout=p_dropout,
                activation=activation,
                pretrained=pretrained,
                fold_id=fold_id
            )
            training_model.eval()

            # Create deployment model and fuse BatchNorm
            model = MIBMINetDeploy(
                F1=F1,
                D=D,
                F2=F2,
                C=C,
                T=T,
                N=N,
                Nf=Nf,
                Nf2=Nf2,
                activation=activation,
            )
            model.load_and_fuse_from_training_model(training_model)
            print("✅ BatchNorm fused into Conv layers for deployment")

        except Exception as e:
            print(f"⚠️  Could not load pretrained model: {e}")
            print("   Using random weights instead...")
            model = MIBMINetDeploy(
                F1=F1,
                D=D,
                F2=F2,
                C=C,
                T=T,
                N=N,
                Nf=Nf,
                Nf2=Nf2,
                activation=activation,
            )
    else:
        # Option 2: Create model with random weights
        print("📦 Creating deployment model with random weights...")
        model = MIBMINetDeploy(
            F1=F1,
            D=D,
            F2=F2,
            C=C,
            T=T,
            N=N,
            Nf=Nf,
            Nf2=Nf2,
            activation=activation,
        )

    model.eval()

    # Generate random input data for EEG signals
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
    generate_mi_bminet_onnx_and_data(save_path)
