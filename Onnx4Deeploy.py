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
from typing import List, Optional

from onnx4deeploy.core.optimizer_onnx import derive_optimizer_dir

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def list_available_models():
    """List available model exporters"""
    from onnx4deeploy.models import (
        AutoencoderExporter,
        CCTExporter,
        DSCNNExporter,
        EpiDeNetExporter,
        LightweightCnnExporter,
        MambaExporter,
        MIBMInetExporter,
        MobileNetV2Exporter,
        MobileViTExporter,
        ResNetExporter,
        SimpleCnnExporter,
        SimpleMlpExporter,
        SleepConViTExporter,
        SpeechNetExporter,
        TinyTransformerExporter,
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
            # Small config for on-chip testing (Siracusa L2=2MB).
            # Default T=2000 needs ~49MB L2; T=128 needs ~1.4MB.
            "config": {"time_steps": 128, "F1": 4, "Nf": 16, "Nf2": 4},
        },
        "SleepConViT": {
            "class": SleepConViTExporter,
            "description": "SleepConViT (Vision Transformer for Sleep Stage Classification)",
            "input_shape": "(B, 1, 3000)",
            "classes": 5,
        },
        # ── MLperf Tiny Benchmarks ──────────────────────────────────────────
        # IC  — Image Classification (CIFAR-10, ResNet-8)
        "ResNet8": {
            "class": ResNetExporter,
            "description": "ResNet-8 (MLperf Tiny IC, CIFAR-10, ~78K params)",
            "input_shape": "(B, 3, 32, 32)",
            "classes": 10,
            "config": {"variant": "resnet8", "img_size": 32, "num_classes": 10},
        },
        # VWW — Visual Wake Words (MobileNetV2-0.35, 96×96, 2 classes)
        "MobileNetV2-VWW": {
            "class": MobileNetV2Exporter,
            "description": "MobileNetV2-0.35 (MLperf Tiny VWW, 96×96, person/not-person)",
            "input_shape": "(B, 3, 96, 96)",
            "classes": 2,
            "config": {"width_mult": 0.35, "img_size": 96, "num_classes": 2},
        },
        # KWS — Keyword Spotting (DS-CNN-XS, MFCC 25×10, 12 classes)
        "DSCNN": {
            "class": DSCNNExporter,
            "description": "DS-CNN-XS (MLperf Tiny KWS, MFCC 25×10, 12 classes, ~10K params)",
            "input_shape": "(B, 1, 25, 10)",
            "classes": 12,
            "config": {"variant": "xs", "n_time": 25, "n_freq": 10},
        },
        # KWS full-size reference (DS-CNN-S, 49×10)
        "DSCNN-S": {
            "class": DSCNNExporter,
            "description": "DS-CNN-S (MLperf Tiny KWS reference, MFCC 49×10, ~270K params)",
            "input_shape": "(B, 1, 49, 10)",
            "classes": 12,
            "config": {"variant": "s", "n_time": 49, "n_freq": 10},
        },
        # AD  — Anomaly Detection (FC Autoencoder, 128-dim, MSE loss)
        "Autoencoder": {
            "class": AutoencoderExporter,
            "description": "FC Autoencoder-tiny (MLperf Tiny AD, 128-dim, MSE loss, ~26K params)",
            "input_shape": "(B, 128)",
            "classes": None,
            "config": {"variant": "tiny", "input_dim": 128},
        },
        # AD  — full MLperf Tiny reference autoencoder
        "Autoencoder-MLPerf": {
            "class": AutoencoderExporter,
            "description": "FC Autoencoder (MLperf Tiny AD reference, 128→[128,128,128]→128)",
            "input_shape": "(B, 128)",
            "classes": None,
            "config": {"variant": "mlperf", "input_dim": 128},
        },
        # Simple Models
        "SimpleMLP": {
            "class": SimpleMlpExporter,
            "description": "Simple Multi-Layer Perceptron (Demo)",
            "input_shape": "(B, 1, 28, 28)",
            "classes": 10,
        },
        "SimpleCNN": {
            "class": SimpleCnnExporter,
            "description": "Simple CNN (2 strided-conv + FC, no MaxPool, training-ready)",
            "input_shape": "(B, 1, 16, 16)",
            "classes": 10,
        },
        "TinyTransformer": {
            "class": TinyTransformerExporter,
            "description": "Tiny Patch Transformer for MNIST (~10K params, fast compile)",
            "input_shape": "(B, 16, 49)",
            "classes": 10,
        },
        # EMG / Bio-Signal Models
        "SpeechNet": {
            "class": SpeechNetExporter,
            "description": "SpeechNet (SilentWear EMG silent speech, ~15K params)",
            "input_shape": "(B, 1, 14, 700)",
            "classes": 9,
            "config": {"num_channels": 14, "time_steps": 700, "num_classes": 9},
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
        "ConvGrad": "Combined convolution input + weight gradient (dX + dW [+ dB])",
        "ConvGradB": "Convolution bias gradient",
        # Others
        "ReduceSum": "Sum reduction",
        "SoftmaxCrossEntropy": "Softmax cross entropy",
        "SoftmaxCrossEntropyDualOutput": "Softmax cross entropy (loss + log_prob outputs)",
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
        import re as _re

        _snake = _re.sub(
            r"(?<=[a-z0-9])(?=[A-Z])", "_", operator_name
        ).lower()  # ConvGradXW → conv_grad_xw
        possible_module_names = [
            f"{operator_name.lower()}",  # relu
            _snake,  # conv_grad_xw
            f"{_snake}_xw",  # conv_grad → conv_grad_xw (legacy filename)
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
    learning_rate: Optional[float] = None,
    classes: Optional[List[int]] = None,
    use_lora: bool = False,
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    training_strategy: Optional[str] = None,
):
    """Generate model ONNX"""
    print(f"\n{'='*70}")
    print(f"🚀 Generating model: {model_name} ({mode.upper()} mode)")
    print(f"{'='*70}\n")

    # Set default output path if not specified.
    # LoRA exports get a "_lora" tag in the directory name so they don't collide
    # with the corresponding non-LoRA training export (e.g. cct_lora_train vs cct_train).
    if output_path is None:
        lora_tag = "_lora" if use_lora else ""
        output_path = str(
            project_root / "onnx" / "model" / f"{model_name.lower()}{lora_tag}_{mode}"
        )
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
        if mode in ("train", "train_single_step"):
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
        if mode in ("train", "train_single_step"):
            exporter._config_overrides["n_batches"] = n_batches
            exporter._config_overrides["n_accum"] = n_accum
            exporter._config_overrides["batch_size"] = batch_size
            if learning_rate is not None:
                exporter._config_overrides["learning_rate"] = learning_rate
        # Dataset selection applies to both infer and train data generation
        exporter._config_overrides["dataset"] = dataset
        if data_path is not None:
            exporter._config_overrides["data_path"] = data_path
        if data_size is not None:
            exporter._config_overrides["data_size"] = data_size
        if classes is not None:
            exporter._config_overrides["classes"] = classes
        # LoRA fine-tuning toggles (only models with LoRA support read these:
        # currently TinyTransformer and CCT). Unrelated exporters silently ignore them.
        # --use-lora alone implies training_strategy="lora" (the common case);
        # explicitly passing --training-strategy still wins.
        if use_lora:
            exporter._config_overrides["use_lora"] = True
            if training_strategy is None:
                training_strategy = "lora"
        if lora_r is not None:
            exporter._config_overrides["lora_r"] = lora_r
        if lora_alpha is not None:
            exporter._config_overrides["lora_alpha"] = lora_alpha
        if training_strategy is not None:
            exporter._config_overrides["training_strategy"] = training_strategy

        # Apply model-specific configuration via _config_overrides so it survives
        # the internal load_config() call inside export_inference/export_training().
        if "config" in models[model_key]:
            exporter._config_overrides.update(models[model_key]["config"])

        # Export according to mode
        if mode == "infer":
            onnx_file = exporter.export_inference()
            mode_desc = "Inference mode"
        elif mode == "train":
            onnx_file = exporter.export_training()
            mode_desc = "Training mode"
        elif mode == "train_single_step":
            onnx_file = exporter.export_training_single_step()
            mode_desc = "Single-step (training-as-inference) mode"
        elif mode == "quant":
            onnx_file = exporter.export_quantized()
            mode_desc = "Quantized (QCDQ) mode"
        else:
            print(f"❌ Unknown mode: {mode}")
            print("   Available modes: infer, train, train_single_step, quant")
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
            # Also show the optimizer ONNX written to the sibling _optimizer directory.
            opt_dir = derive_optimizer_dir(output_path)
            if opt_dir is not None:
                opt_onnx = Path(opt_dir) / "network.onnx"
                if opt_onnx.exists():
                    size = opt_onnx.stat().st_size / 1024
                    size_str = f"{size:.1f} KB" if size < 1024 else f"{size/1024:.1f} MB"
                    rel = str(
                        opt_onnx.relative_to(output_dir.parent)
                        if output_dir.parent in opt_onnx.parents
                        else opt_onnx
                    )
                    print(f"  ✓ {'optimizer/network.onnx':<25} ({size_str})  [{rel}]")

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
        choices=["infer", "train", "train_single_step", "quant"],
        default="infer",
        help="Model export mode: infer (FP32 inference), train (training), "
        "train_single_step (training graph wired up for inference-runner-style "
        "per-tensor gradient verification: lazy_reset_grad pinned True, "
        "outputs.npz holds raw ORT grads), or quant (Brevitas QCDQ ONNX via "
        "DeepQuant — see docs/Quantization_Integration.md). [default: infer]",
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
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        dest="learning_rate",
        metavar="LR",
        help="(train mode) SGD learning rate. Overrides the exporter's default "
        "(typically 0.001). Example: --lr 0.05.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        dest="classes",
        metavar="C",
        help="(mnist) Restrict training to specific digit classes. "
        "Example: --classes 0 8  trains a binary 0-vs-8 classifier.",
    )
    parser.add_argument(
        "--training-strategy",
        type=str,
        default=None,
        dest="training_strategy",
        metavar="STRATEGY",
        help="(train mode) Override the exporter's default training strategy "
        "(controls which parameters are trainable). "
        "TinyTransformer: full | head | custom | lora.  "
        "CCT: no_tokenizer | last_block | linear | full | custom | lora. "
        "Use 'lora' together with --use-lora to fine-tune only the LoRA adapters.",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        dest="use_lora",
        help="(train mode, TinyTransformer & CCT) Attach LoRA A/B adapters to the "
        "attention block (q/k/v[/out|proj]) AND default training_strategy to 'lora' "
        "so only the LoRA adapters get gradients. Pass --training-strategy explicitly "
        "to override (e.g. '--use-lora --training-strategy full' for LoRA-augmented "
        "full fine-tuning).",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        dest="lora_r",
        metavar="R",
        help="(train mode, --use-lora) LoRA rank. Default: 4.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        dest="lora_alpha",
        metavar="A",
        help="(train mode, --use-lora) LoRA scaling numerator; effective scale = alpha / r. "
        "Default: 16.",
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
            learning_rate=args.learning_rate,
            classes=args.classes,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            training_strategy=args.training_strategy,
        )


if __name__ == "__main__":
    main()
