# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Base ONNX Exporter - Unified abstraction for exporting PyTorch models to ONNX.

This module provides the core abstraction layer for Onnx4Deeploy, eliminating duplicate
code across CCT, EpiDeNet, MI-BMInet and other models.

Supports both training and inference mode exports with different optimization passes.
"""

from __future__ import annotations

import io
import os
import shutil
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import onnx
import torch
from onnx import helper
from onnxruntime.training import artifacts

from DeepQuant.DeepQuant.Export4Deeploy import exportBrevitas

from .onnx_utils import print_model_info, randomize_onnx_initializers
from onnx4deeploy.transform.zo_transform import generate_zo_graph


class ExportMode(Enum):
    """Export mode: training or inference."""

    TRAINING = "train"
    INFERENCE = "infer"
    ZO_TRAINING = "zo-train"


class BaseONNXExporter(ABC):
    """
    Base class for ONNX model exporters.

    This class provides a unified interface for exporting PyTorch models to ONNX,
    handling the common workflow for both training and inference modes.

    Training Mode Workflow:
    1. Load configuration
    2. Create PyTorch model
    3. Export to ONNX
    4. Run inference optimizations
    5. Generate training artifacts
    6. Add optimizer nodes (SGD/Adam)
    7. Run training optimizations
    8. Perform shape and type inference

    Inference Mode Workflow:
    1. Load configuration
    2. Create PyTorch model
    3. Export to ONNX
    4. Run inference optimizations
    5. Perform shape inference

    Subclasses must implement:
    - load_config(): Load model-specific configuration
    - create_model(): Create the PyTorch model
    - get_input_shape(): Return the input tensor shape
    - get_trainable_params(): Return list of trainable parameter names (for training mode)
    """

    def __init__(self, save_path: Optional[str] = None, config_file: str = "config.yaml"):
        """
        Initialize the exporter.

        Args:
            save_path: Optional custom path to save ONNX files
            config_file: Path to configuration YAML file
        """
        self.save_path = save_path
        self.config_file = config_file
        self.config = None
        self.paths = None

    @abstractmethod
    def load_config(self) -> Dict[str, Any]:
        """
        Load model-specific configuration.

        Returns:
            Dictionary containing model configuration parameters
            Must include: opset_version, batch_size
        """

    @abstractmethod
    def create_model(self) -> torch.nn.Module:
        """
        Create the PyTorch model.

        Returns:
            PyTorch model ready for export
        """

    @abstractmethod
    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the input tensor shape for the model.

        Returns:
            Tuple representing input shape (batch_size, channels, height, width) or similar
        """

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """
        Get list of trainable parameter names.

        Default: all parameters are trainable.
        Override for fine-tuning or frozen layers.

        Args:
            all_param_names: List of all parameter names in the model

        Returns:
            List of parameter names that should be trainable
        """
        return all_param_names

    def get_inference_pipeline(self) -> "OptimizationPipeline":
        """
        Get the optimization pipeline for inference mode.

        Subclasses can override this to customize the optimization pipeline.
        For example, CCT overrides this to add transformer-specific optimizations.

        Returns:
            OptimizationPipeline configured for this model's inference optimization
        """
        from .optimization_passes import create_inference_pipeline

        return create_inference_pipeline()

    def get_training_pipeline(self) -> "OptimizationPipeline":
        """
        Get the optimization pipeline for training mode.

        Subclasses can override this to customize the optimization pipeline.

        Returns:
            OptimizationPipeline configured for this model's training optimization
        """
        from .optimization_passes import create_training_pipeline

        return create_training_pipeline()

    def run_inference_optimization(self, onnx_file: str, output_file: str):
        """
        Run ONNX optimizations for inference mode using optimization pipeline.

        Default implementation uses a standard inference pipeline with:
        - Node renaming for C compatibility
        - Identity node removal
        - Reshape fusion
        - GEMM input dimension unification
        - BiasGelu optimization
        - Shape operation optimization

        Subclasses can override get_inference_pipeline() to customize the pipeline
        (e.g., CCT adds transformer-specific ONNX Runtime optimizations).

        Args:
            onnx_file: Path to input ONNX file
            output_file: Path to save optimized ONNX file
        """
        # Get the optimization pipeline for this model
        pipeline = self.get_inference_pipeline()

        # Copy to output if different files
        if onnx_file != output_file:
            shutil.copy(onnx_file, output_file)

        # Run the pipeline
        try:
            pipeline.run(output_file, output_file)
        except Exception as e:
            print(f"   ⚠️  Pipeline execution failed: {e}")

    def run_training_optimization(self, onnx_file: str, output_file: str):
        """
        Run ONNX optimizations for training mode using optimization pipeline.

        Default: uses comprehensive training optimization pipeline from train_optimizer.
        Subclasses can override get_training_pipeline() for model-specific optimizations.

        Args:
            onnx_file: Path to input ONNX file
            output_file: Path to save optimized ONNX file
        """
        # Get the optimization pipeline for this model
        pipeline = self.get_training_pipeline()

        # Copy to output if different files
        if onnx_file != output_file:
            shutil.copy(onnx_file, output_file)

        # Run the pipeline
        try:
            pipeline.run(output_file, output_file)
        except Exception as e:
            print(f"   ⚠️  Pipeline execution failed: {e}")

    def get_model_name(self) -> str:
        """
        Get the model name for file naming.

        Returns:
            Model name string
        """
        return self.__class__.__name__.replace("Exporter", "").replace("ONNX", "")

    def setup_paths(self, mode: ExportMode) -> Dict[str, str]:
        """
        Setup output directory and file paths.

        Args:
            mode: Export mode (training or inference)

        Returns:
            Dictionary of file paths
        """
        model_name = self.get_model_name()
        config_str = self._get_config_string()
        base_name = f"{model_name}_{mode.value}{config_str}"

        if self.save_path:
            output_dir = self.save_path
        else:
            output_dir = os.path.join(os.getcwd(), "onnx", base_name)

        os.makedirs(output_dir, exist_ok=True)

        paths = {
            "output_dir": output_dir,
            "network": os.path.join(output_dir, "network.onnx"),
        }

        if mode == ExportMode.TRAINING:
            paths.update(
                {
                    "network_infer": os.path.join(output_dir, "network_infer.onnx"),
                    "network_train": os.path.join(output_dir, "network_train.onnx"),
                    "network_train_optim": os.path.join(output_dir, "network_train_optim.onnx"),
                    "network_pre_sgd": os.path.join(output_dir, "network_pre_sgd.onnx"),
                }
            )
        
        if mode == ExportMode.ZO_TRAINING:
            paths.update(
                {
                    "network_infer": os.path.join(output_dir, "network_infer.onnx"),
                    "network_zo_train": os.path.join(output_dir, "network_zo_train.onnx"),
                }
            )

        return paths

    def _get_config_string(self) -> str:
        """
        Get configuration string for folder naming.

        Subclasses can override this to customize folder names.
        Example: "_32_128_2_2" for CCT config

        Returns:
            Configuration string
        """
        return ""

    def _export_to_onnx(
        self, model: torch.nn.Module, input_tensor: torch.Tensor, opset_version: int = 12
    ) -> onnx.ModelProto:
        """
        Export PyTorch model to ONNX.

        Args:
            model: PyTorch model
            input_tensor: Sample input tensor
            opset_version: ONNX opset version

        Returns:
            ONNX model
        """
        f = io.BytesIO()
        torch.onnx.export(
            model,
            input_tensor,
            f,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            do_constant_folding=True,  # Enable constant folding for cleaner graphs
            export_params=True,
            keep_initializers_as_inputs=False,
        )

        onnx_model = onnx.load_model_from_string(f.getvalue())
        return onnx_model

    def export_inference(self, save_path: Optional[str] = None, quant: bool = False) -> str:
        """
        Export model in inference mode.

        Args:
            save_path: Optional custom save path

        Returns:
            Path to the exported ONNX file
        """
        if save_path:
            self.save_path = save_path

        # Load configuration
        self.config = self.load_config()
        self.paths = self.setup_paths(ExportMode.INFERENCE)

        print(f"\n{'='*60}")
        print(f"🚀 Exporting {self.get_model_name()} to ONNX (Inference Mode)")
        print(f"{'='*60}\n")

        # Create PyTorch model
        print("📦 Creating PyTorch model...")
        model = self.create_model()
        model.eval()  # Inference mode

        # Generate input
        input_shape = self.get_input_shape()
        input_tensor = torch.randn(*input_shape, dtype=torch.float32)
        print(f"   Input shape: {input_shape}")

        if not quant:
            # initialize weights and biases for testing
            for param in model.parameters():
                if param.requires_grad:
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
            # Export to ONNX
            print("\n📤 Exporting to ONNX...")
            opset_version = self.config.get("opset_version", 12)
            onnx_model = self._export_to_onnx(model, input_tensor, opset_version)
        
        elif quant:
            # load weights.
            state_dict = torch.load(os.path.join(os.getcwd(), self.config.get("weights_path", "model_weights.pth")), map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print("\n📤 Exporting to ONNX with quantization...")
            # use DeepQuant to export to ONNX
            onnx_model = exportBrevitas(model, input_tensor, debug=False)
                    
        # Save
        onnx.save(onnx_model, self.paths["network"])
        print(f"✅ ONNX model saved: {self.paths['network']}")

        # Run inference optimizations
        print("\n🔧 Running inference optimizations...")
        self.run_inference_optimization(self.paths["network"], self.paths["network"])

        # Run shape inference
        print("\n🔍 Running shape inference...")
        from ..optimization.shape_optimizer import infer_shapes_with_custom_ops

        infer_shapes_with_custom_ops(self.paths["network"], self.paths["network"])

        # Save test input/output data if method is implemented
        if hasattr(self, "save_test_data"):
            try:
                self.save_test_data(model, self.paths["output_dir"])
            except Exception as e:
                print(f"⚠️  Failed to save test data: {e}")

        print_model_info(self.paths["network"])

        print(f"\n{'='*60}")
        print("✅ Export Complete!")
        print(f"   Final model: {self.paths['network']}")
        print(f"   Test data: {self.paths['output_dir']}/test_*.npy")
        print(f"{'='*60}\n")

        return self.paths["network"]

    def export_training(self, save_path: Optional[str] = None) -> str:
        """
        Export model in training mode with training artifacts.

        Args:
            save_path: Optional custom save path

        Returns:
            Path to the exported training ONNX file
        """
        if save_path:
            self.save_path = save_path

        # Load configuration
        self.config = self.load_config()
        self.paths = self.setup_paths(ExportMode.TRAINING)

        print(f"\n{'='*60}")
        print(f"🚀 Exporting {self.get_model_name()} to ONNX (Training Mode)")
        print(f"{'='*60}\n")

        # Create PyTorch model
        print("📦 Creating PyTorch model...")
        model = self.create_model()
        model.train()  # Training mode

        # Store model for test data generation
        self._model = model

        # Generate input
        input_shape = self.get_input_shape()
        input_tensor = torch.randn(*input_shape, dtype=torch.float32)
        print(f"   Input shape: {input_shape}")

        # Export to ONNX
        print("\n📤 Exporting to ONNX...")
        opset_version = self.config.get("opset_version", 12)
        onnx_model = self._export_to_onnx(model, input_tensor, opset_version)

        # Randomize initializers for testing
        onnx_model = randomize_onnx_initializers(onnx_model)

        # Save inference model
        onnx.save(onnx_model, self.paths["network_infer"])
        print(f"✅ Inference ONNX saved: {self.paths['network_infer']}")

        # Run inference optimizations
        print("\n🔧 Running inference optimizations...")
        self.run_inference_optimization(self.paths["network_infer"], self.paths["network_infer"])

        # Reload optimized model
        onnx_model = onnx.load(self.paths["network_infer"])
        print_model_info(self.paths["network_infer"])

        # Get trainable parameters
        all_param_names = [init.name for init in onnx_model.graph.initializer]
        requires_grad = self.get_trainable_params(all_param_names)
        frozen_params = [name for name in all_param_names if name not in requires_grad]

        print(f"\n🔹 Trainable parameters: {len(requires_grad)}")
        print(f"🔹 Frozen parameters: {len(frozen_params)}")

        # Generate training artifacts
        print("\n🏋️ Generating training artifacts...")
        artifacts.generate_artifacts(
            onnx_model,
            optimizer=artifacts.OptimType.SGD,
            loss=artifacts.LossType.CrossEntropyLoss,
            requires_grad=requires_grad,
            frozen_params=frozen_params,
            artifact_directory=self.paths["output_dir"],
        )

        # Rename training_model.onnx to network_train.onnx
        training_model_path = os.path.join(self.paths["output_dir"], "training_model.onnx")
        if os.path.exists(training_model_path):
            os.rename(training_model_path, self.paths["network_train"])
            print(f"✅ Training model saved: {self.paths['network_train']}")

        # Load training model and add gradient outputs
        onnx_model = onnx.load(self.paths["network_train"])
        graph = onnx_model.graph
        grad_tensor_names = [name + "_grad" for name in requires_grad]

        for grad_name in grad_tensor_names:
            if not any(output.name == grad_name for output in graph.output):
                grad_output = helper.make_tensor_value_info(grad_name, onnx.TensorProto.FLOAT, None)
                graph.output.append(grad_output)

        # Save with gradient outputs
        onnx.save(onnx_model, self.paths["network_train_optim"])
        onnx.save(onnx_model, self.paths["network_train"])

        # Run shape inference for training model (handles Microsoft custom ops)
        print("\n🔍 Running shape inference...")
        from ..optimization.shape_optimizer import infer_shapes_with_custom_ops

        infer_shapes_with_custom_ops(
            self.paths["network_train_optim"], self.paths["network_train_optim"]
        )

        # Run training-specific optimizations
        print("\n🔧 Running training optimizations...")
        self.run_training_optimization(self.paths["network_train_optim"], self.paths["network"])

        # Save pre-SGD model
        shutil.copy(self.paths["network"], self.paths["network_pre_sgd"])
        print(f"✅ Pre-SGD model saved: {self.paths['network_pre_sgd']}")

        # Create test input/output
        print("\n🧪 Creating test input/output...")
        self._create_test_data()

        # Add optimizer (SGD) nodes
        print("\n➕ Adding SGD optimizer nodes...")
        self._add_optimizer_nodes()

        print(f"\n{'='*60}")
        print("✅ Export Complete!")
        print(f"   Final model: {self.paths['network']}")
        print(f"{'='*60}\n")

        return self.paths["network"]

    def export_zo_training(self, save_path: Optional[str] = None, quant: bool = False) -> str:
        """
        Export model in zeroth-order training mode.

        Args:
            save_path: Optional custom save path"""
        if save_path:
            self.save_path = save_path

        # Load configuration
        self.config = self.load_config()
        self.paths = self.setup_paths(ExportMode.ZO_TRAINING)

        print(f"\n{'='*60}")
        print(f"🚀 Exporting {self.get_model_name()} to ONNX (Zeroth-Order Training Mode)")
        print(f"{'='*60}\n")

        # Create PyTorch model
        print("📦 Creating PyTorch model...")
        model = self.create_model()
        model.eval()  # Zeroth-Order Training mode

        # Generate input
        input_shape = self.get_input_shape()
        input_tensor = torch.randn(*input_shape, dtype=torch.float32)
        print(f"   Input shape: {input_shape}")

        # Export to ONNX
        print("\n📤 Exporting to ONNX...")
        opset_version = self.config.get("opset_version", 12)
        onnx_model = self._export_to_onnx(model, input_tensor, opset_version)

        # Save
        onnx.save(onnx_model, self.paths["network_infer"])
        print(f"✅ ONNX model saved: {self.paths['network_infer']}")

        # Run inference optimizations
        print("\n🔧 Running inference optimizations...")
        self.run_inference_optimization(self.paths["network_infer"], self.paths["network_infer"])

        # Run shape inference
        print("\n🔍 Running shape inference...")
        from ..optimization.shape_optimizer import infer_shapes_with_custom_ops

        infer_shapes_with_custom_ops(self.paths["network_infer"], self.paths["network_infer"])

        # Save test input/output data if method is implemented
        if hasattr(self, "save_test_data"):
            try:
                self.save_test_data(model, self.paths["output_dir"])
            except Exception as e:
                print(f"⚠️  Failed to save test data: {e}")

        # Reload optimized model
        onnx_model = onnx.load(self.paths["network_infer"])
        print_model_info(self.paths["network_infer"])

        # Get trainable parameters
        all_param_names = [init.name for init in onnx_model.graph.initializer]
        requires_grad = self.get_trainable_params(all_param_names)
        frozen_params = [name for name in all_param_names if name not in requires_grad]

        print(f"\n🔹 Trainable parameters: {len(requires_grad)}")
        print(f"🔹 Frozen parameters: {len(frozen_params)}")

        # Transform model for zeroth-order training (e.g., add noise nodes, modify outputs)
        print("\n🔧 Transforming model for zeroth-order training...")
        # Randomize initializers for testing
        onnx_model = randomize_onnx_initializers(onnx_model)
        generate_zo_graph(
            inference_onnx=self.paths["network_infer"],
            output_onnx=self.paths["network_zo_train"],
            zo_config=self.config["zo"],
        )

        # # Load training model and add gradient outputs
        # onnx_model = onnx.load(self.paths["network_train"])
        # graph = onnx_model.graph
        # grad_tensor_names = [name + "_grad" for name in requires_grad]

        # for grad_name in grad_tensor_names:
        #     if not any(output.name == grad_name for output in graph.output):
        #         grad_output = helper.make_tensor_value_info(grad_name, onnx.TensorProto.FLOAT, None)
        #         graph.output.append(grad_output)

        # # Save with gradient outputs
        # onnx.save(onnx_model, self.paths["network_train_optim"])
        # onnx.save(onnx_model, self.paths["network_train"])

        # Run shape inference for training model (handles Microsoft custom ops)
        print("\n🔍 Running shape inference...")
        from ..optimization.shape_optimizer import infer_shapes_with_custom_ops
        infer_shapes_with_custom_ops(
            self.paths["network_zo_train"]
        )

        # # Run training-specific optimizations
        # print("\n🔧 Running training optimizations...")
        # self.run_training_optimization(self.paths["network_train_optim"], self.paths["network"])

        # # Save pre-SGD model
        # shutil.copy(self.paths["network"], self.paths["network_pre_sgd"])
        # print(f"✅ Pre-SGD model saved: {self.paths['network_pre_sgd']}")

        # Create test input/output
        print("\n🧪 Creating test input/output...")
        self._create_test_data()

        # # Add optimizer (SGD) nodes
        # print("\n➕ Adding SGD optimizer nodes...")
        # self._add_optimizer_nodes()

        print(f"\n{'='*60}")
        print("✅ Export Complete!")
        print(f"   Final model: {self.paths['network']}")
        print(f"{'='*60}\n")

    def _create_test_data(self):
        """
        Create test input/output data for training.

        Uses ONNX Runtime to generate reference output from the (potentially randomized) ONNX model.
        This ensures test data matches the actual ONNX model weights.
        """
        # Generate test data using ONNX Runtime
        try:
            from pathlib import Path

            import numpy as np
            import onnxruntime as ort

            print("💾 Generating test input/output data from ONNX model...")

            # Create test input
            input_shape = self.get_input_shape()
            test_input = np.random.randn(*input_shape).astype(np.float32)

            # Run ONNX inference to get output
            session = ort.InferenceSession(self.paths["network_infer"])
            input_name = session.get_inputs()[0].name
            test_output = session.run(None, {input_name: test_input})[0]

            # Save as .npz files
            save_path = Path(self.paths["output_dir"])
            save_path.mkdir(parents=True, exist_ok=True)

            np.savez(save_path / "inputs.npz", input=test_input)
            np.savez(save_path / "outputs.npz", output=test_output)

            print("  ✅ Saved test data (ONNX reference):")
            print(f"     Input:  {save_path / 'inputs.npz'} shape={test_input.shape}")
            print(f"     Output: {save_path / 'outputs.npz'} shape={test_output.shape}")
        except Exception as e:
            print(f"⚠️  Failed to create test data: {e}")

    def _add_optimizer_nodes(self):
        """
        Add optimizer (SGD/Adam) nodes to the model.

        Subclasses can override this to customize optimizer node addition.
        """
        # Default implementation - can be overridden

    def export(self, mode: str = "train", save_path: Optional[str] = None) -> str:
        """
        Main export entry point.

        Args:
            mode: Export mode - "train" or "infer"
            save_path: Optional custom save path

        Returns:
            Path to the exported ONNX file
        """
        if mode == "train":
            return self.export_training(save_path)
        elif mode == "infer":
            return self.export_inference(save_path)
        elif mode == "zo-train":
            return self.export_zo_training(save_path)
        elif mode =="q-infer":
            return self.export_inference(save_path, quant=True)
        elif mode == "q-zo-train":
            return self.export_zo_training(save_path, quant=True)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'infer'")
