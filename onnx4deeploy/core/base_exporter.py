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
from onnxruntime.training import artifacts

from .onnx_utils import print_model_info


class ExportMode(Enum):
    """Export mode: training or inference."""

    TRAINING = "train"
    INFERENCE = "infer"


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

    def get_data_source(self) -> "DataSource":
        """
        Return the data source used to generate (input, label) pairs for
        create_training_test_data().

        Default: RandomDataSource (preserves original behaviour).
        Override in subclasses to use real datasets (e.g. MNISTDataSource).
        """
        from ..data.random_datasource import RandomDataSource

        return RandomDataSource()

    def get_training_pipeline(self) -> "OptimizationPipeline":
        """
        Get the optimization pipeline for training mode.

        Subclasses can override this to customize the optimization pipeline.

        Returns:
            OptimizationPipeline configured for this model's training optimization
        """
        from .optimization_passes import create_training_pipeline

        return create_training_pipeline()

    def run_training_optimization(self, onnx_file: str, output_file: str):
        """
        Run ONNX optimizations for training mode.

        Args:
            onnx_file: Path to input training ONNX file
            output_file: Path to save optimized ONNX file
        """
        from ..optimization.train_optimizer import run_train_onnx_optimization

        # Pass the inference model so frozen params can be sourced from its initializers.
        infer_file = self.paths.get("network_infer") if self.paths else None
        run_train_onnx_optimization(onnx_file, output_file, onnx_infer_file=infer_file)

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
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        opset_version: int = 12,
        training_mode: bool = False,
    ) -> onnx.ModelProto:
        """
        Export PyTorch model to ONNX.

        Args:
            model: PyTorch model
            input_tensor: Sample input tensor
            opset_version: ONNX opset version
            training_mode: If True, export with TrainingMode.TRAINING so that ops like
                LayerNorm, Dropout, and BatchNorm are exported with their training-specific
                outputs (e.g. saved_mean / inv_std_var for LayerNorm).  These intermediate
                values are required by ORT's gradient builders and are *not* present in a
                default (eval-mode) ONNX export.

        Returns:
            ONNX model
        """
        f = io.BytesIO()
        export_training = (
            torch.onnx.TrainingMode.TRAINING if training_mode else torch.onnx.TrainingMode.EVAL
        )

        # For opset ≥ 17, LayerNormalization is a standard ONNX op, so PyTorch exports it
        # with only 1 output (Y).  ORT's gradient builder needs O(1)=mean and O(2)=inv_std_var.
        # Fix: override aten::layer_norm to declare outputs=3.  The extra two outputs are
        # "dangling" in the forward graph but ORT preserves them through get_optimized_model()
        # and stashes them for LayerNormalizationGrad.
        # For opset ≤ 16, PyTorch decomposes LayerNorm to individual ops and ORT's
        # LayerNormFusion re-creates the node with 3 outputs automatically — no override needed.
        if training_mode and opset_version >= 17:
            from torch.onnx import symbolic_helper

            @symbolic_helper.parse_args("v", "is", "v", "v", "f", "i")
            def _layer_norm_training(g, input, normalized_shape, weight, bias, eps, cudnn_enable):
                y, _mean, _inv_std = g.op(
                    "LayerNormalization",
                    input,
                    weight,
                    bias,
                    outputs=3,
                    axis_i=-len(normalized_shape),
                    epsilon_f=eps,
                    stash_type_i=1,
                )
                return y

            torch.onnx.register_custom_op_symbolic(
                "aten::layer_norm", _layer_norm_training, opset_version=opset_version
            )

        torch.onnx.export(
            model,
            input_tensor,
            f,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            do_constant_folding=not training_mode,
            export_params=True,
            keep_initializers_as_inputs=False,
            training=export_training,
        )

        onnx_model = onnx.load_model_from_string(f.getvalue())
        return onnx_model

    def export_inference(self, save_path: Optional[str] = None) -> str:
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

        # Export to ONNX
        print("\n📤 Exporting to ONNX...")
        opset_version = self.config.get("opset_version", 12)
        onnx_model = self._export_to_onnx(model, input_tensor, opset_version)

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

        print("📦 Creating PyTorch model...")
        model = self.create_model()
        model.train()  # training=TrainingMode.TRAINING export requires train() mode
        self._model = model

        input_shape = self.get_input_shape()
        input_tensor = torch.randn(*input_shape, dtype=torch.float32)
        print(f"   Input shape: {input_shape}")

        # ort-training ≥ 1.14 requires opset ≥ 13.
        # In opset 13 the Squeeze/Unsqueeze 'axes' became an input tensor (not an
        # attribute).  Any pass that converts axes back to an attribute must NOT
        # run before generate_artifacts.
        opset_version = max(self.config.get("opset_version", 13), 13)
        print(f"\n📤 Exporting to ONNX (opset {opset_version}, training mode)...")
        onnx_model = self._export_to_onnx(model, input_tensor, opset_version, training_mode=True)
        onnx.save(onnx_model, self.paths["network_infer"])
        print(f"✅ Inference ONNX saved: {self.paths['network_infer']}")

        # Run inference optimizations.
        # Set _for_training=True so subclasses (e.g. SleepConViTExporter) can skip
        # ORT transformer fusion, which creates com.microsoft custom ops that are
        # incompatible with generate_artifacts' internal ONNX shape inference.
        print("\n🔧 Running inference optimizations...")
        self._for_training = True
        try:
            self.run_inference_optimization(
                self.paths["network_infer"], self.paths["network_infer"]
            )
        finally:
            self._for_training = False

        # Reload and validate before passing to generate_artifacts.
        # generate_artifacts calls onnx.checker.check_model(model, True) internally,
        # so an invalid model raises here with a clear error message.
        onnx_model = onnx.load(self.paths["network_infer"])
        try:
            onnx.checker.check_model(onnx_model)
            print("✅ Model validation passed")
        except Exception as e:
            raise RuntimeError(
                f"Model failed ONNX validation before generate_artifacts: {e}"
            ) from e
        print_model_info(self.paths["network_infer"])

        # Determine trainable / frozen parameters
        all_param_names = [init.name for init in onnx_model.graph.initializer]
        requires_grad = self.get_trainable_params(all_param_names)
        frozen_params = [name for name in all_param_names if name not in requires_grad]

        print(f"\n🔹 Trainable parameters: {len(requires_grad)}")
        print(f"🔹 Frozen parameters: {len(frozen_params)}")

        # Generate training artifacts.
        # Produces inside artifact_directory:
        #   training_model.onnx  — forward + loss + backward graph
        #   eval_model.onnx      — forward + loss (no gradients)
        #   optimizer_model.onnx — SGD parameter-update graph
        #   checkpoint/          — initial parameter values
        print("\n🏋️ Generating training artifacts...")
        artifacts.generate_artifacts(
            onnx_model,
            optimizer=artifacts.OptimType.SGD,
            loss=artifacts.LossType.CrossEntropyLoss,
            requires_grad=requires_grad,
            frozen_params=frozen_params,
            artifact_directory=self.paths["output_dir"],
        )

        # Rename default artifact name → project convention
        training_model_src = os.path.join(self.paths["output_dir"], "training_model.onnx")
        if not os.path.exists(training_model_src):
            raise RuntimeError(f"generate_artifacts did not produce: {training_model_src}")
        os.rename(training_model_src, self.paths["network_train"])
        print(f"✅ Training model: {self.paths['network_train']}")

        # Run training-specific optimizations.
        # convert_squeeze_unsqueeze_input_to_attr (and other Deeploy transforms)
        # are applied here — AFTER generate_artifacts — so ORT validation is done.
        print("\n🔧 Running training optimizations...")
        self.run_training_optimization(
            self.paths["network_train"], self.paths["network_train_optim"]
        )

        print("\n🔍 Running shape inference on training model...")
        from ..optimization.shape_optimizer import infer_shapes_with_custom_ops

        infer_shapes_with_custom_ops(
            self.paths["network_train_optim"], self.paths["network_train_optim"]
        )

        # Final model = Deeploy-optimized training model
        shutil.copy(self.paths["network_train_optim"], self.paths["network"])
        print(f"✅ Final model: {self.paths['network']}")

        # Generate reference test input/output via ORT on-device training API
        print("\n🧪 Creating test input/output...")
        self.create_training_test_data()

        print(f"\n{'='*60}")
        print("✅ Export Complete!")
        print(f"   Training model:  {self.paths['network_train']}")
        print(f"   Final model:     {self.paths['network']}")
        print(f"   Output dir:      {self.paths['output_dir']}")
        print(f"{'='*60}\n")

        return self.paths["network"]

    def create_training_test_data(self) -> None:
        """
        Generate reference test data for one complete training step.

        Uses ORT's InferenceSession to run the training model with ALL graph inputs
        (data input + labels + all initial weight/bias parameters), then applies
        SGD manually to compute updated parameter values.

        Initial parameter values are read from network_infer.onnx initializers,
        which exactly match the checkpoint values produced by generate_artifacts.

        Saved files
        -----------
        inputs.npz  : {<input_name>: float32, <labels_name>: int64,
                       <param_name>: float32, ...}   ← ALL graph inputs
        outputs.npz : {<param_name>: float32, ..., loss: float32}
                       param tensors  — updated via SGD (param - lr * grad)
                       loss           — scalar cross-entropy loss
        """
        from pathlib import Path

        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import numpy_helper

        try:
            input_shape = self.get_input_shape()
            input_shape[0]
            num_classes = self.config.get("num_classes", 2)
            learning_rate = float(self.config.get("learning_rate", 0.001))

            data_source = self.get_data_source()
            _inputs, _labels = data_source.load_batches(1, input_shape, num_classes, seed=42)
            test_input = _inputs[0]
            labels = _labels[0]

            save_dir = Path(self.paths["output_dir"])

            # Read initial parameter values from the inference model.
            # generate_artifacts uses these initializers as the checkpoint initial state,
            # so they are guaranteed to match the training model's parameter inputs.
            infer_model = onnx.load(self.paths["network_infer"])
            init_map: dict = {
                init.name: numpy_helper.to_array(init) for init in infer_model.graph.initializer
            }

            # Run the training model with ORT InferenceSession.
            # network_train.onnx is pre-optimization and ORT-compatible.
            session = ort.InferenceSession(
                self.paths["network_train"], providers=["CPUExecutionProvider"]
            )

            # Print all graph input names for visibility
            all_input_names = [inp.name for inp in session.get_inputs()]
            print(f"   Training model inputs ({len(all_input_names)}): {all_input_names}")

            # Build a complete input feed for every graph input:
            #   tensor(int64)                -> labels
            #   tensor(bool)                 -> lazy_reset_grad = True (first step)
            #   name in init_map             -> initial parameter value
            #   *_grad.accumulation.buffer   -> zeros (gradient accum buffer init)
            #   shape == input_shape         -> data input
            #   anything else                -> zeros with correct shape
            inputs_dict: dict = {}
            for inp in session.get_inputs():
                name = inp.name
                shape = [d for d in inp.shape if isinstance(d, int) and d > 0]
                if inp.type == "tensor(int64)":
                    inputs_dict[name] = labels
                elif inp.type == "tensor(bool)":
                    # lazy_reset_grad=True resets accumulator at the start of each step
                    inputs_dict[name] = np.array([True])
                elif name in init_map:
                    inputs_dict[name] = init_map[name]
                elif "_grad.accumulation.buffer" in name:
                    # Gradient accumulation buffers are initialized to zero
                    inputs_dict[name] = np.zeros(shape, dtype=np.float32)
                elif shape == list(input_shape):
                    inputs_dict[name] = test_input
                else:
                    inputs_dict[name] = np.zeros(shape, dtype=np.float32)

            # Execute forward + backward
            raw_outputs = session.run(None, inputs_dict)
            output_names = [o.name for o in session.get_outputs()]
            outputs_raw = dict(zip(output_names, raw_outputs))

            # Compute updated parameters: updated = param - lr * grad
            # ORT names gradient outputs as "<param_name>_grad"
            outputs_dict: dict = {}
            for param_name, param_val in init_map.items():
                grad_name = param_name + "_grad"
                if grad_name in outputs_raw:
                    outputs_dict[param_name] = param_val - learning_rate * outputs_raw[grad_name]

            # Include scalar loss
            for out_name, out_val in outputs_raw.items():
                if "loss" in out_name.lower() and "grad" not in out_name.lower():
                    outputs_dict["loss"] = np.atleast_1d(np.array(out_val, dtype=np.float32))
                    break

            if not outputs_dict:
                # Fallback: save raw outputs if no gradient pattern matched
                outputs_dict = {k: v for k, v in outputs_raw.items()}

            # Save: inputs include ALL graph inputs (data + labels + all params)
            np.savez(save_dir / "inputs.npz", **inputs_dict)
            n_params = sum(1 for k in inputs_dict if k in init_map)
            print(
                f"   ✅ inputs.npz  — {len(inputs_dict)} tensors "
                f"(data + labels + {n_params} params)"
            )

            np.savez(save_dir / "outputs.npz", **outputs_dict)
            n_updated = sum(1 for k in outputs_dict if k in init_map)
            print(
                f"   ✅ outputs.npz — {len(outputs_dict)} tensors "
                f"({n_updated} updated params + loss)"
            )

        except Exception as e:
            print(f"   ⚠️  ORT InferenceSession failed ({e}); using fallback...")
            self._create_test_data_fallback()

    def _create_test_data_fallback(self) -> None:
        """
        Fallback: load initial params from network_train.onnx initializers and
        save all graph inputs including parameters.

        Saved files
        -----------
        inputs.npz  : {<name>: tensor, ...}  — ALL graph inputs including params
        outputs.npz : {<param_name>: float32, ...}  — updated params or raw outputs
        """
        from pathlib import Path

        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import numpy_helper

        try:
            input_shape = self.get_input_shape()
            input_shape[0]
            num_classes = self.config.get("num_classes", 2)
            learning_rate = float(self.config.get("learning_rate", 0.001))

            data_source = self.get_data_source()
            _inputs, _labels = data_source.load_batches(1, input_shape, num_classes, seed=42)
            test_input = _inputs[0]
            labels = _labels[0]

            save_dir = Path(self.paths["output_dir"])

            # Read initial param values from network_train initializers
            train_model = onnx.load(self.paths["network_train"])
            init_map = {
                init.name: numpy_helper.to_array(init) for init in train_model.graph.initializer
            }

            session = ort.InferenceSession(
                self.paths["network_train"], providers=["CPUExecutionProvider"]
            )

            # Build complete input feed including all parameters
            inputs_dict: dict = {}
            for inp in session.get_inputs():
                name = inp.name
                if inp.type == "tensor(int64)":
                    inputs_dict[name] = labels
                elif name in init_map:
                    inputs_dict[name] = init_map[name]
                else:
                    inputs_dict[name] = test_input

            raw_outputs = session.run(None, inputs_dict)
            output_names = [o.name for o in session.get_outputs()]
            outputs_raw = dict(zip(output_names, raw_outputs))

            # Apply SGD where gradient outputs are found
            outputs_dict: dict = {}
            for param_name, param_val in init_map.items():
                grad_name = param_name + "_grad"
                if grad_name in outputs_raw:
                    outputs_dict[param_name] = param_val - learning_rate * outputs_raw[grad_name]

            for out_name, out_val in outputs_raw.items():
                if "loss" in out_name.lower() and "grad" not in out_name.lower():
                    outputs_dict["loss"] = np.atleast_1d(np.array(out_val, dtype=np.float32))
                    break

            if not outputs_dict:
                outputs_dict = {k: v for k, v in outputs_raw.items()}

            np.savez(save_dir / "inputs.npz", **inputs_dict)
            n_params = sum(1 for k in inputs_dict if k in init_map)
            print(f"   ✅ inputs.npz  — {len(inputs_dict)} tensors ({n_params} params)")

            np.savez(save_dir / "outputs.npz", **outputs_dict)
            print(f"   ✅ outputs.npz — {len(outputs_dict)} tensors")

        except Exception as e:
            print(f"   ⚠️  Fallback test data generation failed: {e}")

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
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'infer'")
