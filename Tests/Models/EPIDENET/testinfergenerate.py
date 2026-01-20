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


def count_parameters(model):
    """
    Count the number of trainable and total parameters in a PyTorch model
    
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
        param_dict: Dictionary with detailed parameter breakdown by layer
    """
    total_params = 0
    trainable_params = 0
    param_dict = {}
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        param_dict[name] = {
            'shape': list(param.shape),
            'num_params': num_params,
            'trainable': param.requires_grad
        }
    
    return total_params, trainable_params, param_dict


def print_model_summary(model, model_name="EpiDeNet"):
    """
    Print comprehensive model summary including architecture and parameters
    """
    print("\n" + "="*80)
    print(f"  {model_name} Model Summary")
    print("="*80)
    
    # Count parameters
    total_params, trainable_params, param_dict = count_parameters(model)
    
    # Print architecture
    print(f"\n📐 Model Architecture:")
    print("-"*80)
    print(model)
    
    # Print detailed parameter breakdown
    print(f"\n📊 Parameter Breakdown:")
    print("-"*80)
    print(f"{'Layer Name':<40} {'Shape':<25} {'# Params':>12}")
    print("-"*80)
    
    for name, info in param_dict.items():
        shape_str = str(info['shape'])
        trainable_marker = "" if info['trainable'] else " (frozen)"
        print(f"{name:<40} {shape_str:<25} {info['num_params']:>12,}{trainable_marker}")
    
    print("-"*80)
    print(f"{'Total Parameters':<40} {'':<25} {total_params:>12,}")
    print(f"{'Trainable Parameters':<40} {'':<25} {trainable_params:>12,}")
    print(f"{'Non-trainable Parameters':<40} {'':<25} {(total_params - trainable_params):>12,}")
    print("="*80)
    
    # Print model size estimate
    param_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per float32
    print(f"\n💾 Estimated Model Size:")
    print(f"   Float32: {param_size_mb:.2f} MB")
    print(f"   Float16: {param_size_mb/2:.2f} MB")
    print(f"   Int8:    {param_size_mb/4:.2f} MB")
    print("="*80 + "\n")
    
    return total_params, trainable_params


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
    print(f"\n✅ Loaded config: C={C}, T={T}, N={N}, opset_version={opset_version}")

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
        print(f"\n📦 Loading pretrained model from {pretrained}...")
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
            
            # Print training model summary
            print_model_summary(training_model, "EpiDeNet (Training Version - with BatchNorm)")

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
        print("\n📦 Creating deployment model with random weights...")
        model = EpiDeNetDeploy(
            C=C,
            T=T,
            output_classes=N,
        )

    model.eval()
    
    # Print deployment model summary
    total_params, trainable_params = print_model_summary(model, "EpiDeNet (Deployment Version - BatchNorm Fused)")

    # Generate random input data for EOG signals
    print(f"📊 Generating test input data with shape: {input_shape}")
    input_data = np.random.randn(*input_shape).astype(np.float32)
    np.savez(input_file, input=input_data)

    input_tensor = torch.tensor(input_data)
    
    # Test forward pass
    print(f"\n🧪 Testing forward pass...")
    with torch.no_grad():
        output = model(input_tensor)
        print(f"   Input shape:  {input_tensor.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output classes: {N}")

    # Export to ONNX (fixed batch size for cleaner graph)
    print(f"\n🔧 Exporting to ONNX (opset {opset_version})...")
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
    print(f"\n🔧 Simplifying ONNX model...")
    import onnxsim
    onnx_model_simplified, check = onnxsim.simplify(onnx_file)
    onnx.save(onnx_model_simplified, onnx_file)
    print(f"✅ ONNX model simplified and saved to {onnx_file}")

    # Run inference to generate output data
    print(f"\n🚀 Running ONNX inference...")
    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(None, {"input": input_data})[0]

    np.savez(output_file, output=output_data)
    print(f"✅ Output data saved to {output_file}")

    # Unify GEMM input dimensions
    print(f"\n🔧 Unifying GEMM input dimensions...")
    unify_gemm_input_dims(onnx_file, onnx_file)
    print(f"✅ Successfully unified GEMM input dimensions. Saved as {onnx_file}")
    
    # Final summary
    print("\n" + "="*80)
    print("  Final Export Summary")
    print("="*80)
    print(f"  Model:           EpiDeNet EOG Classifier")
    print(f"  Input Shape:     {input_shape}")
    print(f"  Output Classes:  {N}")
    print(f"  Total Params:    {total_params:,}")
    print(f"  ONNX File:       {onnx_file}")
    print(f"  Input Data:      {input_file}")
    print(f"  Output Data:     {output_file}")
    print("="*80 + "\n")


if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_epidenet_onnx_and_data(save_path)