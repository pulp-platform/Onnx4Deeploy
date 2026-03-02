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
        MambaExporter,
        MIBMInetExporter,
        MobileNetV2Exporter,
        MobileViTExporter,
        ResNetExporter,
        SimpleMlpExporter,
        SleepConViTExporter,
        TinyViTExporter,
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
        "TinyViT-5M": {
            "class": TinyViTExporter,
            "description": "TinyViT-5M (Compact Vision Transformer, ~5M params)",
            "input_shape": "(B, 3, 64, 64)",
            "classes": 10,
            "config": {"variant": "tiny_vit_5m", "img_size": 64, "num_classes": 10},
        },
        "TinyViT-11M": {
            "class": TinyViTExporter,
            "description": "TinyViT-11M (Compact Vision Transformer, ~11M params)",
            "input_shape": "(B, 3, 64, 64)",
            "classes": 10,
            "config": {"variant": "tiny_vit_11m", "img_size": 64, "num_classes": 10},
        },
        "TinyViT-21M": {
            "class": TinyViTExporter,
            "description": "TinyViT-21M (Compact Vision Transformer, ~21M params)",
            "input_shape": "(B, 3, 64, 64)",
            "classes": 10,
            "config": {"variant": "tiny_vit_21m", "img_size": 64, "num_classes": 10},
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
    }
    return models


def list_available_operators():
    """List available operators"""
    operators = {
        # Basic operators
        "Add": "Addition operator",
        "Relu": "ReLU activation function",
        "Transpose": "Tensor transpose",
        "Concat": "Tensor concatenation (supports 3 inputs)",
        "Split": "Tensor split",
        # Matrix operations
        "Gemm": "General matrix multiplication",
        "MatMul": "Matrix multiplication",
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
        "Conv2D": "2D convolution (supports asymmetric padding)",
        "ConvGradX": "Convolution input gradient",
        "ConvGradW": "Convolution weight gradient",
        "ConvGradB": "Convolution bias gradient",
        # Others
        "ReduceSum": "Sum reduction",
        "SoftmaxCrossEntropy": "Softmax cross entropy",
        "ReluGrad": "ReLU gradient",
        # Training operators (custom domain: com.microsoft)
        "InPlaceAccumulatorV2": "Gradient accumulation with lazy reset (com.microsoft)",
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


def generate_model(
    model_name: str,
    mode: str,
    output_path: Optional[str] = None,
    n_batches: Optional[int] = None,
    n_steps: Optional[int] = None,
    n_epochs: Optional[float] = None,
    n_accum: int = 1,
    batch_size: int = 1,
    dataset: str = "random",
    data_path: Optional[str] = None,
    data_size: Optional[int] = None,
):
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

        # Resolve n_batches from whichever training-length parameter was given.
        # Priority: --n-batches > --n-steps > --n-epochs > default(4)
        if mode == "train":
            import math

            if n_batches is not None:
                # explicit --n-batches: use as-is (auto-rounded in create_training_test_data)
                pass
            elif n_steps is not None:
                # --n-steps S  →  n_batches = S × n_accum
                n_batches = n_steps * n_accum
                print(f"📐 --n-steps {n_steps}  × --n-accum {n_accum}  → n_batches={n_batches}")
            elif n_epochs is not None:
                # --n-epochs E  →  n_batches = ceil(E × data_size / n_accum) × n_accum
                if data_size is None:
                    print("❌ --n-epochs requires --data-size to be set")
                    sys.exit(1)
                total_samples = n_epochs * data_size
                n_batches = math.ceil(total_samples / n_accum) * n_accum
                actual_epochs = n_batches / data_size
                print(
                    f"📐 --n-epochs {n_epochs}  × data_size {data_size}  ÷ --n-accum {n_accum}"
                    f"  → n_batches={n_batches}  (≈{actual_epochs:.1f} epochs)"
                )
            else:
                n_batches = 4  # default

        # Store CLI overrides that must survive the internal load_config() call
        # inside export_training().  Exporters that support _config_overrides
        # will apply these at the end of their load_config() implementation.
        exporter._config_overrides = {}
        if mode == "train":
            exporter._config_overrides["n_batches"] = n_batches
            exporter._config_overrides["n_accum"] = n_accum
            exporter._config_overrides["batch_size"] = batch_size
        # Dataset selection applies to both infer and train data generation
        exporter._config_overrides["dataset"] = dataset
        if data_path is not None:
            exporter._config_overrides["data_path"] = data_path
        if data_size is not None:
            exporter._config_overrides["data_size"] = data_size

        # Apply model-specific configuration if available
        if "config" in models[model_key]:
            exporter.config = exporter.load_config()
            exporter.config.update(models[model_key]["config"])

        # Export according to mode
        if mode == "infer":
            onnx_file = exporter.export_inference()
            mode_desc = "Inference mode"
        elif mode == "train":
            onnx_file = exporter.export_training()
            mode_desc = "Training mode"
        else:
            print(f"❌ Unknown mode: {mode}")
            print("   Available modes: infer, train")
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
    print("  python Onnx4Deeploy.py -model TinyViT-5M -mode infer -o ./output/tinyvit_5m")
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
        help="Generate model ONNX (e.g.: ResNet18, ResNet50, MobileNetV2, MobileViT-XS, TinyViT-5M, CCT, Mamba, MIBMInet)",
    )

    # Model mode parameters
    parser.add_argument(
        "-mode",
        "--mode",
        type=str,
        choices=["infer", "train"],
        default="infer",
        help="Model export mode: infer (inference) or train (training) [default: infer]",
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

    # Training-specific options
    train_len_group = parser.add_mutually_exclusive_group()
    train_len_group.add_argument(
        "--n-epochs",
        type=float,
        default=None,
        dest="n_epochs",
        metavar="E",
        help="(train mode) Number of full passes over the data pool. "
        "Requires --data-size. n_batches = ceil(E × data_size / n_accum) × n_accum. "
        "Example: --data-size 20 --n-accum 8 --n-epochs 25 → n_batches=504.",
    )
    train_len_group.add_argument(
        "--n-steps",
        type=int,
        default=None,
        dest="n_steps",
        metavar="S",
        help="(train mode) Number of SGD weight-update steps. "
        "n_batches = S × n_accum. "
        "Example: --n-steps 63 --n-accum 8 → n_batches=504.",
    )
    train_len_group.add_argument(
        "--n-batches",
        type=int,
        default=None,
        dest="n_batches",
        metavar="N",
        help="(train mode) Total forward-pass count (low-level, backward-compat). "
        "Auto-rounded down to nearest multiple of n_accum if not divisible. "
        "Prefer --n-epochs or --n-steps for clearer semantics.",
    )
    parser.add_argument(
        "--n-accum",
        type=int,
        default=1,
        dest="n_accum",
        metavar="N",
        help="(train mode) Effective batch size: number of samples accumulated "
        "per SGD update (gradient accumulation). Default: 1.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        dest="batch_size",
        metavar="N",
        help="(train mode) Number of samples per mini-batch (batch size). Default: 1.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="random",
        choices=["random", "mnist"],
        dest="dataset",
        help="Data source for training test data. "
        "'random' (default): random Gaussian inputs. "
        "'mnist': real MNIST images (downloaded automatically if needed).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        dest="data_path",
        metavar="PATH",
        help="Root directory for dataset files (used with --dataset mnist). "
        "Default: /tmp/mnist.",
    )
    parser.add_argument(
        "--data-size",
        type=int,
        default=None,
        dest="data_size",
        metavar="N",
        help="(mnist) Fixed pool size for epoch-cycling mode. "
        "None (default): each batch draws fresh random images — no loss descent visible. "
        "N: fix a pool of N images and cycle through them with per-epoch shuffle — "
        "loss descends as the network repeatedly sees the same images. "
        "Rule of thumb: set --n-batches to at least 4×N for visible convergence.",
    )

    # Other options
    parser.add_argument("--examples", action="store_true", help="Show usage examples")

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
        generate_model(
            args.model,
            args.mode,
            args.output,
            n_batches=args.n_batches,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            n_accum=args.n_accum,
            batch_size=args.batch_size,
            dataset=args.dataset,
            data_path=args.data_path,
            data_size=args.data_size,
        )


if __name__ == "__main__":
    main()
