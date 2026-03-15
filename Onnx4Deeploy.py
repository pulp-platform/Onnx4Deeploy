#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Onnx4Deeploy - Unified command-line entry point

Usage:
    # Generate operator test
    python Onnx4Deeploy.py -operator Relu -o ./output

    # Generate model inference graph
    python Onnx4Deeploy.py -model CCT -mode infer -o ./output

    # Generate model training graph
    python Onnx4Deeploy.py -model CCT -mode train -o ./output
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def list_available_models():
    """List available model exporters"""
    from onnx4deeploy.models import (
        CCTExporter,
        EpiDeNetExporter,
        LightweightCnnExporter,
        QLiteCnnExporter,
        MambaExporter,
        MIBMInetExporter,
        MobileNetV2Exporter,
        MobileViTExporter,
        ResNetExporter,
        SimpleMlpExporter,
        SleepConViTExporter,
    )

    models = {
        # Computer Vision - Classic CNNs
        "ResNet18": {
            "class": ResNetExporter,
            "description": "ResNet-18 (Residual Network, 11.7M params)",
            "input_shape": "(B, 3, 224, 224)",
            "classes": 1000,
            "config": {"variant": "resnet18"},
        },
        "ResNet34": {
            "class": ResNetExporter,
            "description": "ResNet-34 (Residual Network, 21.8M params)",
            "input_shape": "(B, 3, 224, 224)",
            "classes": 1000,
            "config": {"variant": "resnet34"},
        },
        "ResNet50": {
            "class": ResNetExporter,
            "description": "ResNet-50 (MLPerf Benchmark, 25.6M params)",
            "input_shape": "(B, 3, 224, 224)",
            "classes": 1000,
            "config": {"variant": "resnet50"},
        },
        "MobileNetV2": {
            "class": MobileNetV2Exporter,
            "description": "MobileNetV2 (MLPerf Mobile, 3.5M params)",
            "input_shape": "(B, 3, 224, 224)",
            "classes": 1000,
            "config": {"width_mult": 1.0},
        },
        "MobileViT-XXS": {
            "class": MobileViTExporter,
            "description": "MobileViT-XXS (Hybrid CNN-Transformer, ~1.3M params)",
            "input_shape": "(B, 3, 256, 256)",
            "classes": 1000,
            "config": {"variant": "mobile_vit_xxs"},
        },
        "MobileViT-XS": {
            "class": MobileViTExporter,
            "description": "MobileViT-XS (Hybrid CNN-Transformer, ~2.3M params)",
            "input_shape": "(B, 3, 256, 256)",
            "classes": 1000,
            "config": {"variant": "mobile_vit_xs"},
        },
        "MobileViT-S": {
            "class": MobileViTExporter,
            "description": "MobileViT-S (Hybrid CNN-Transformer, ~5.6M params)",
            "input_shape": "(B, 3, 256, 256)",
            "classes": 1000,
            "config": {"variant": "mobile_vit_s"},
        },
        # Transformer Models
        "CCT": {
            "class": CCTExporter,
            "description": "Compact Convolutional Transformer (Vision)",
            "input_shape": "(B, 3, 32, 32)",
            "classes": 10,
        },
        "Mamba": {
            "class": MambaExporter,
            "description": "Mamba (Selective State Space Model for Sequences)",
            "input_shape": "(B, 512, 256)",
            "classes": 10,
            "config": {"max_seq_len": 512, "d_model": 256, "n_layers": 4},
        },
        # EEG/BCI Models
        "EpiDeNet": {
            "class": EpiDeNetExporter,
            "description": "EpiDeNet (EEG Epilepsy Detection)",
            "input_shape": "(B, 1, 8, 2000)",
            "classes": 11,
        },
        "MIBMInet": {
            "class": MIBMInetExporter,
            "description": "MI-BMInet (Motor Imagery BMI)",
            "input_shape": "(B, 1, 8, 2000)",
            "classes": 2,
        },
        "SleepConViT": {
            "class": SleepConViTExporter,
            "description": "SleepConViT (Vision Transformer for Sleep Stage Classification)",
            "input_shape": "(B, 1, 3000)",
            "classes": 5,
        },
        # Simple Models
        "SimpleMLP": {
            "class": SimpleMlpExporter,
            "description": "Simple Multi-Layer Perceptron (Demo)",
            "input_shape": "(B, 1, 28, 28)",
            "classes": 10,
        },
        "LightweightCNN": {
            "class": LightweightCnnExporter,
            "description": "Lightweight CNN (Compact CNN for image classification)",
            "input_shape": "(B, 1, 28, 28)",
            "classes": 10,
        },
        "QLiteCNN": {
            "class": QLiteCnnExporter,
            "description": "QLite CNN (Compact CNN for image classification)",
            "input_shape": "(B, 1, 28, 28)",
            "classes": 10,
        },
    }
    return models


def list_available_operators():
    """List available operators"""
    operators = {
        # Basic operators
        "Add": "Addition operator",
        "Relu": "ReLU activation function",
        "Transpose": "Tensor transpose",
        "Split": "Tensor split",
        # Matrix operations
        "Gemm": "General matrix multiplication",
        "MatMul": "Matrix multiplication",
        "Conv2d": "2D convolution",
        # Pooling
        "MaxPool": "Max pooling",
        "AveragePool": "Average pooling",
        "AveragePoolGrad": "Average pooling gradient",
        # Normalization
        "LayerNorm": "Layer normalization",
        "LayerNormGrad": "Layer normalization gradient",
        "GroupNorm": "Group normalization",
        "GroupNormGradX": "Group normalization input gradient",
        "GroupNormGradW": "Group normalization weight gradient",
        # Convolution
        "ConvGradX": "Convolution input gradient",
        "ConvGradW": "Convolution weight gradient",
        "ConvGradB": "Convolution bias gradient",
        # ZO
        "PerturbNormal": "Perturb input with gaussian random noise",
        "PerturbUniform": "Perturb input with uniform random noise",
        "PerturbTriangle": "Perturb input with triangle random noise",
        "PerturbRademacher": "Perturb input with Rademacher random noise",
        "PerturbEggroll": "Perturb input with Eggroll random noise",
        "RQSPerturbrademacher": "Perturb input with quantized Rademacher random noise",
        "RQSPerturbUniform": "Perturb input with quantized Uniform random noise",
        
        # Others
        "ReduceSum": "Sum reduction",
        "SoftmaxCrossEntropy": "Softmax cross entropy",
        "ReluGrad": "ReLU gradient",
        
    }
    return operators


def generate_operator(operator_name: str, output_path: Optional[str] = None):
    """Generate operator test"""
    print(f"\n{'='*70}")
    print(f"🔧 Generating operator: {operator_name}")
    print(f"{'='*70}\n")

    # Set default output path if not specified
    if output_path is None:
        output_path = str(project_root / "onnx" / "operator" / operator_name.lower())
        print(f"📁 Using default output path: {output_path}\n")

    # Dynamically import operator class
    try:
        # Try multiple class name patterns
        possible_class_names = [
            f"{operator_name}OperatorTest",  # Standard pattern: ReluOperatorTest
            f"{operator_name}Operator",  # Alternative: ReluOperator
            f"{operator_name}Test",  # Short pattern: ReluTest
        ]

        # Try multiple module name patterns
        possible_module_names = [
            f"{operator_name.lower()}",  # relu
            f"{operator_name.lower()}_operator",  # relu_operator
            f"{operator_name.lower()}_exporter",  # relu_exporter
        ]

        operator_class = None
        for module_suffix in possible_module_names:
            if operator_class:
                break
            module_name = f"onnx4deeploy.operators.{module_suffix}"
            try:
                module = __import__(module_name, fromlist=["*"])
                for class_name in possible_class_names:
                    try:
                        operator_class = getattr(module, class_name)
                        break
                    except AttributeError:
                        continue
            except ImportError:
                continue

        if not operator_class:
            raise ImportError(f"Operator not found: {operator_name}")

        # Create operator instance
        operator = operator_class(save_path=output_path)

        # Generate
        onnx_file, input_file, output_file = operator.generate()

        print(f"\n{'='*70}")
        print("✅ Operator generation completed!")
        print(f"{'='*70}")
        print(f"\n📁 Generated files:")
        print(f"  ✓ ONNX model:  {onnx_file}")
        print(f"  ✓ Test input:   {input_file}")
        print(f"  ✓ Test output:   {output_file}")
        print(f"\n💡 Output source: ONNX Runtime")

        return onnx_file

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def generate_model(model_name: str, mode: str, output_path: Optional[str] = None, noise_type: str = "gaussian"):
    """Generate model ONNX"""
    print(f"\n{'='*70}")
    print(f"🚀 Generating model: {model_name} ({mode.upper()} mode)")
    print(f"{'='*70}\n")

    # Set default output path if not specified
    if output_path is None:
        output_path = str(project_root / "onnx" / "model" / f"{model_name.lower()}_{mode}")
        print(f"📁 Using default output path: {output_path}\n")

    # Get model class
    models = list_available_models()

    # Case-insensitive model name lookup
    model_name_lower = model_name.lower()
    model_key = None
    for key in models.keys():
        if key.lower() == model_name_lower:
            model_key = key
            break

    if model_key is None:
        print(f"❌ Unknown model: {model_name}")
        print(f"\nAvailable models:")
        for name, info in models.items():
            print(f"  - {name}: {info['description']}")
        sys.exit(1)

    model_class = models[model_key]["class"]

    try:
        # Create exporter
        exporter = model_class(save_path=output_path)

        # Apply model-specific configuration if available
        if "config" in models[model_key]:
            exporter.config = exporter.load_config()
            exporter.config.update(models[model_key]["config"])

        # Export according to mode
        if mode == "infer":
            onnx_file = exporter.export_inference()
            mode_desc = "Inference mode"
        elif mode == "q-infer":
            onnx_file = exporter.export_inference(quant=True)
            mode_desc = "Quantized Inference mode"
        elif mode == "train":
            onnx_file = exporter.export_training()
            mode_desc = "Training mode"
        elif mode == "zo-train":
            onnx_file = exporter.export_zo_training(noise_type=noise_type)
            mode_desc = "Zeroth-order Training mode"
        elif mode == "q-zo-train":
            onnx_file = exporter.export_zo_training(noise_type=noise_type, quant=True)
            mode_desc = "Quantized Zeroth-order Training mode"
        else:
            print(f"❌ Unknown mode: {mode}")
            print("   Available modes: infer, train, zo-train, q-infer, q-zo-train")
            sys.exit(1)

        print(f"\n{'='*70}")
        print(f"✅ {mode_desc} model generation completed!")
        print(f"{'='*70}")

        # Display generated files
        output_dir = Path(output_path) if output_path else Path(onnx_file).parent
        print(f"\n📁 Generated files:")

        files_to_check = ["network.onnx", "inputs.npz", "outputs.npz"]
        if mode == "train":
            files_to_check.extend(["network_train.onnx", "optimizer_model.onnx"])
        elif mode in ["zo-train", "q-zo-train"]:
            files_to_check.append("network_zo.onnx")
        
        for file in files_to_check:
            file_path = output_dir / file
            if file_path.exists():
                size = file_path.stat().st_size / 1024
                size_str = f"{size:.1f} KB" if size < 1024 else f"{size/1024:.1f} MB"
                print(f"  ✓ {file:<25} ({size_str})")

        print(f"\n💡 Output source: PyTorch model (for verifying ONNX correctness)")

        return onnx_file

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def print_usage_examples():
    """Print usage examples"""
    print("\n" + "=" * 70)
    print("📖 Usage Examples")
    print("=" * 70)

    print("\n🔧 Operator level:")
    print("  python Onnx4Deeploy.py -operator Relu -o ./output/relu")
    print("  python Onnx4Deeploy.py -operator Add -o ./output/add")

    print("\n🚀 Model level:")
    print("  # MLPerf benchmark models")
    print("  python Onnx4Deeploy.py -model ResNet50 -mode infer -o ./output/resnet50")
    print("  python Onnx4Deeploy.py -model MobileNetV2 -mode infer -o ./output/mobilenetv2")
    print("")
    print("  # Hybrid and Transformer models")
    print("  python Onnx4Deeploy.py -model MobileViT-XS -mode infer -o ./output/mobilevit")
    print("  python Onnx4Deeploy.py -model CCT -mode infer -o ./output/cct_infer")
    print("  python Onnx4Deeploy.py -model CCT -mode train -o ./output/cct_train")
    print("  python Onnx4Deeploy.py -model Mamba -mode infer -o ./output/mamba")
    print("")
    print("  # Other models")
    print("  python Onnx4Deeploy.py -model ResNet18 -mode infer -o ./output/resnet18")
    print("  python Onnx4Deeploy.py -model MIBMInet -mode infer -o ./output/mibminet")

    print("\n📋 List available options:")
    print("  python Onnx4Deeploy.py --list-models")
    print("  python Onnx4Deeploy.py --list-operators")
    print()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Onnx4Deeploy - ONNX model and operator generation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate operator test
  python Onnx4Deeploy.py -operator Relu -o ./output

  # Generate model inference graph
  python Onnx4Deeploy.py -model CCT -mode infer -o ./output

  # Generate model training graph
  python Onnx4Deeploy.py -model CCT -mode train -o ./output

  # List available options
  python Onnx4Deeploy.py --list-models
  python Onnx4Deeploy.py --list-operators
        """,
    )

    # Main parameter group
    main_group = parser.add_mutually_exclusive_group(required=False)
    main_group.add_argument(
        "-operator",
        "--operator",
        type=str,
        metavar="NAME",
        help="Generate operator test (e.g.: Relu, Add, Gemm)",
    )
    main_group.add_argument(
        "-model",
        "--model",
        type=str,
        metavar="NAME",
        help="Generate model ONNX (e.g.: ResNet18, ResNet50, MobileNetV2, MobileViT-XS, CCT, Mamba, MIBMInet)",
    )

    # Model mode parameters
    parser.add_argument(
        "-mode",
        "--mode",
        type=str,
        choices=["infer", "train", "zo-train", "q-infer", "q-zo-train"],
        default="infer",
        help="Model export mode: infer (inference), train (BP training), zo-train (zeroth-order training), q-infer (quantized inference), or q-zo-train (quantized zeroth-order training) [default: infer]",
    )

    # Output path
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        metavar="PATH",
        help="Output directory path [default: ./onnx/operator/<name> or ./onnx/model/<name>_<mode>]",
    )

    # List options
    list_group = parser.add_mutually_exclusive_group()
    list_group.add_argument("--list-models", action="store_true", help="List all available models")
    list_group.add_argument(
        "--list-operators", action="store_true", help="List all available operators"
    )

    # Other options
    parser.add_argument("--examples", action="store_true", help="Show usage examples")
    parser.add_argument("--noise-type", type=str, choices=["gaussian", "uniform", "triangle", "rademacher", "eggroll"], 
                        default="gaussian", help="Noise type for perturbation operators [default: gaussian]")
    # Parse arguments
    args = parser.parse_args()

    # Handle list options
    if args.list_models:
        print("\n" + "=" * 70)
        print("📋 Available Models")
        print("=" * 70 + "\n")
        models = list_available_models()
        for name, info in models.items():
            print(f"  {name:<15} {info['description']}")
            print(f"  {'':15} Input: {info['input_shape']}, Classes: {info['classes']}")
            print()
        return

    if args.list_operators:
        print("\n" + "=" * 70)
        print("📋 Available Operators")
        print("=" * 70 + "\n")
        operators = list_available_operators()
        for name, desc in sorted(operators.items()):
            print(f"  {name:<25} {desc}")
        print()
        return

    if args.examples:
        print_usage_examples()
        return

    # Check if an operation was specified
    if not args.operator and not args.model:
        parser.print_help()
        print("\n❌ Error: Must specify -operator or -model")
        print("\n💡 Tip: Use --examples to see usage examples")
        print("         Use --list-models to see available models")
        print("         Use --list-operators to see available operators")
        sys.exit(1)

    # Execute operation
    if args.operator:
        generate_operator(args.operator, args.output)
    elif args.model:
        generate_model(args.model, args.mode, args.output, args.noise_type)


if __name__ == "__main__":
    main()
