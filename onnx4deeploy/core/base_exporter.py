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

from .onnx_utils import print_model_info

# onnxruntime.training is only required by export_training (artifact generation).
# Import lazily inside that method so single_step / inference / pytorch-only
# workflows can run on systems without the onnxruntime-training package.


def _fold_conv_bn_inplace(model: "torch.nn.Module") -> int:
    """Fold every Conv+BatchNorm2d pair in ``model`` into a single biased Conv.

    Required before Brevitas/DeepQuant export so the resulting QCDQ ONNX has no
    standalone ``BatchNormalization`` op (Deeploy's Siracusa target does not
    map it; it expects BN to be absorbed at quant time).

    Approach: walk every parent module, pair each ``BatchNorm2d`` child with
    the immediately preceding ``Conv*`` child (sibling attribute, by attribute
    declaration order). For each pair, use ``torch.nn.utils.fusion.fuse_conv_bn_eval``
    to produce a Conv whose weight+bias absorbs gamma/beta/running_mean/var,
    write it back in place of the original Conv, and replace the BN with
    ``nn.Identity()``. This works on plain ``nn.Conv2d`` and on Brevitas
    ``QuantConv2d`` (which inherits from ``nn.Conv2d`` and exposes the same
    weight/bias parameters; the quantization proxies will re-wrap automatically).

    Returns the number of pairs folded.
    """
    import torch.nn as nn
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    n_folded = 0
    for parent in model.modules():
        # Children in declaration order. Pair each BN with its immediate
        # predecessor Conv sibling (works for both Sequential and the
        # ``self.conv1 = ...; self.bn1 = ...`` flat style).
        children = list(parent.named_children())
        for i, (bn_name, bn) in enumerate(children):
            if not isinstance(bn, nn.BatchNorm2d):
                continue
            if i == 0:
                continue
            prev_name, prev = children[i - 1]
            # ``QuantConv2d`` (Brevitas) subclasses ``nn.Conv2d``.
            if not isinstance(prev, nn.Conv2d):
                continue
            try:
                fused = fuse_conv_bn_eval(prev.eval(), bn.eval())
            except Exception:
                # Skip pairs where folding is not safe (e.g. shared params).
                continue
            # Write the fused weight/bias into the existing conv module so any
            # Brevitas quant proxies attached to it stay wired up.
            with torch.no_grad():
                prev.weight.copy_(fused.weight.detach())
                if fused.bias is not None:
                    if prev.bias is None:
                        prev.bias = nn.Parameter(fused.bias.detach().clone())
                    else:
                        prev.bias.copy_(fused.bias.detach())
            # Replace BN with identity so the forward pass skips it cleanly.
            setattr(parent, bn_name, nn.Identity())
            n_folded += 1
    return n_folded


class ExportMode(Enum):
    """Export mode: training, inference, or single-step training-as-inference."""

    TRAINING = "train"
    INFERENCE = "infer"
    # Single-step training-as-inference: same fwd+bwd+InPlaceAccumulator graph as
    # train, but lazy_reset_grad is pinned to True (constant initializer) so each
    # InPlaceAccumulator output equals the pure batch dW (no historical accum).
    # outputs.npz holds the raw ORT-computed grad for every graph output, letting
    # `deeployRunner_*.py` (inference path) flag any per-tensor grad divergence.
    SINGLE_STEP = "train_single_step"


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

    # ------------------------------------------------------------------ #
    # Quantized export (optional, per-exporter opt-in)                    #
    # ------------------------------------------------------------------ #

    def create_brevitas_model(self) -> torch.nn.Module:
        """
        Return a Brevitas-quantized version of the model.

        Each exporter that wants to support `-mode quant` must override this.
        See `docs/Quantization_Integration.md` for the Brevitas substitution
        recipe and a worked example.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_brevitas_model(). "
            f"See docs/Quantization_Integration.md for the recipe."
        )

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

    def get_loss_type(self):
        """
        Return the ORT training loss type used by generate_artifacts().

        Default: CrossEntropyLoss (for all classification models).
        Override to return artifacts.LossType.MSELoss for reconstruction tasks
        (e.g., autoencoders for the MLperf Tiny Anomaly Detection benchmark),
        or artifacts.LossType.BCEWithLogitsLoss for binary classification.

        Available types: CrossEntropyLoss | MSELoss | BCEWithLogitsLoss | L1Loss

        Returns:
            onnxruntime.training.artifacts.LossType
        """
        from onnxruntime.training import artifacts

        return artifacts.LossType.CrossEntropyLoss

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

        if mode in (ExportMode.TRAINING, ExportMode.SINGLE_STEP):
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

        # Determine trainable / frozen parameters.
        # BatchNorm running statistics (running_mean, running_var, num_batches_tracked)
        # are non-differentiable buffers updated via EMA, not backprop.  They must go
        # into frozen_params (not requires_grad).  Omitting them from both lists causes
        # ORT to treat them as trainable by default → tries to build gradient nodes → crash.
        _BN_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")
        all_initializer_names = [init.name for init in onnx_model.graph.initializer]
        bn_buffer_names = [
            n for n in all_initializer_names if any(n.endswith(s) for s in _BN_BUFFERS)
        ]
        all_param_names = [n for n in all_initializer_names if n not in bn_buffer_names]
        requires_grad = self.get_trainable_params(all_param_names)
        frozen_params = [n for n in all_param_names if n not in requires_grad] + bn_buffer_names

        print(f"\n🔹 Trainable parameters: {len(requires_grad)}")
        print(f"🔹 Frozen parameters: {len(frozen_params)}")

        # Generate training artifacts.
        # Produces inside artifact_directory:
        #   training_model.onnx  — forward + loss + backward graph
        #   eval_model.onnx      — forward + loss (no gradients)
        #   optimizer_model.onnx — SGD parameter-update graph
        #   checkpoint/          — initial parameter values
        print("\n🏋️ Generating training artifacts...")
        from onnxruntime.training import artifacts

        artifacts.generate_artifacts(
            onnx_model,
            optimizer=artifacts.OptimType.SGD,
            loss=self.get_loss_type(),
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

        # Build the SGD optimizer ONNX graph (reads network.onnx to detect trainable params)
        self.create_optimizer()

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

    # ---------------------------------------------------------------------- #
    # Training test-data helpers                                             #
    # ---------------------------------------------------------------------- #

    _GRAD_ACC_SUFFIX = "_grad.accumulation.buffer"

    def _load_init_map(self, onnx_path: str) -> dict:
        """
        Load model initializers from an ONNX file into a ``{name: np.ndarray}`` dict.

        Used by ``create_training_test_data`` (and subclass overrides) to retrieve
        the initial parameter values that match the checkpoint produced by
        ``generate_artifacts``.

        Args:
            onnx_path: Path to the ONNX model whose initializers should be loaded.

        Returns:
            Dict mapping initializer name → numpy array.
        """
        import onnx
        from onnx import numpy_helper

        model = onnx.load(onnx_path)
        return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

    def _build_input_feed(
        self,
        session: "ort.InferenceSession",
        param_values: dict,
        test_input: "np.ndarray",
        labels: "np.ndarray",
        lazy_reset_grad: bool = True,
    ) -> dict:
        """
        Build a complete ORT input feed dict for one forward+backward pass.

        Assignment rules applied in priority order:

        1. ``tensor(int64)``             → *labels*
        2. ``tensor(bool)``              → ``[lazy_reset_grad]``  (InPlaceAccumulatorV2 ctrl)
        3. name in *param_values*        → current parameter value
        4. name ends with ``_grad.accumulation.buffer`` → zeros (accumulator init)
        5. shape matches ``get_input_shape()``           → *test_input*
        6. anything else                 → zeros with the correct shape

        Args:
            session:          Active ORT InferenceSession for the training model.
            param_values:     Dict of current parameter tensors (may be initial weights
                              or mid-training weights for gradient-accumulation loops).
            test_input:       Data input array for this mini-batch.
            labels:           Label array for this mini-batch.
            lazy_reset_grad:  Value written to any ``tensor(bool)`` graph input.
                              Pass ``True`` on the first accumulation step, ``False``
                              on subsequent steps.

        Returns:
            Dict mapping every session input name → numpy array.
        """
        import numpy as np

        input_shape = self.get_input_shape()
        feed: dict = {}
        for inp in session.get_inputs():
            name = inp.name
            shape = [d for d in inp.shape if isinstance(d, int) and d > 0]
            if inp.type == "tensor(int64)":
                feed[name] = labels
            elif inp.type == "tensor(bool)":
                feed[name] = np.array([lazy_reset_grad])
            elif name in param_values:
                feed[name] = param_values[name]
            elif self._GRAD_ACC_SUFFIX in name:
                feed[name] = np.zeros(shape, dtype=np.float32)
            elif shape == list(input_shape):
                feed[name] = test_input
            else:
                feed[name] = np.zeros(shape, dtype=np.float32)
        return feed

    def create_training_test_data(self) -> None:
        """
        Generate reference test data for one complete training step.

        Uses ORT's InferenceSession to run the training model with ALL graph inputs
        (data input + labels + all initial weight/bias parameters), then applies
        SGD manually to compute updated parameter values.

        Initial parameter values are read from ``network_infer.onnx`` initializers,
        which exactly match the checkpoint values produced by ``generate_artifacts``.
        If ``network_infer.onnx`` is unavailable the initializers are taken from
        ``network_train.onnx`` instead.

        Saved files
        -----------
        inputs.npz  : ALL graph inputs — data, labels, initial params, ctrl tensors
        outputs.npz : SGD-updated parameter tensors + scalar ``loss``
        """
        from pathlib import Path

        import numpy as np
        import onnxruntime as ort

        input_shape = self.get_input_shape()
        num_classes = self.config.get("num_classes", 2)
        learning_rate = float(self.config.get("learning_rate", 0.001))
        save_dir = Path(self.paths["output_dir"])

        data_source = self.get_data_source()
        test_inputs, labels_list = data_source.load_batches(1, input_shape, num_classes, seed=42)
        test_input, labels = test_inputs[0], labels_list[0]

        # Prefer network_infer.onnx: its initializers are guaranteed to match the
        # checkpoint produced by generate_artifacts.  Fall back to network_train.onnx
        # initializers when network_infer.onnx is not available (e.g. custom workflows).
        infer_path = self.paths.get("network_infer", "")
        init_source = (
            infer_path if infer_path and os.path.exists(infer_path) else self.paths["network_train"]
        )
        init_map = self._load_init_map(init_source)

        session = ort.InferenceSession(
            self.paths["network_train"], providers=["CPUExecutionProvider"]
        )
        print(
            f"   Training model inputs ({len(session.get_inputs())}): "
            f"{[i.name for i in session.get_inputs()]}"
        )

        feed = self._build_input_feed(session, init_map, test_input, labels)
        outputs_raw = dict(zip([o.name for o in session.get_outputs()], session.run(None, feed)))

        # SGD update: updated = param - lr * grad
        # ORT names gradient outputs as "<param_name>_grad".
        outputs_dict: dict = {}
        for param_name, param_val in init_map.items():
            if (param_name + "_grad") in outputs_raw:
                outputs_dict[param_name] = (
                    param_val - learning_rate * outputs_raw[param_name + "_grad"]
                )
        for out_name, out_val in outputs_raw.items():
            if "loss" in out_name.lower() and "grad" not in out_name.lower():
                outputs_dict["loss"] = np.atleast_1d(np.array(out_val, dtype=np.float32))
                break
        if not outputs_dict:
            outputs_dict = dict(outputs_raw)

        np.savez(save_dir / "inputs.npz", **feed)
        n_params = sum(1 for k in feed if k in init_map)
        print(f"   ✅ inputs.npz  — {len(feed)} tensors (data + labels + {n_params} params)")

        np.savez(save_dir / "outputs.npz", **outputs_dict)
        n_updated = sum(1 for k in outputs_dict if k in init_map)
        print(
            f"   ✅ outputs.npz — {len(outputs_dict)} tensors ({n_updated} updated params + loss)"
        )

    def create_optimizer(self) -> Optional[str]:
        """
        Build and save the SGD optimizer ONNX graph alongside the training export.

        Auto-detects trainable parameters via ``<param>_grad.accumulation.buffer``
        inputs in the final network.onnx, then writes a minimal SGD graph to the
        conventional optimizer directory next to the training directory:

            <base>/<model>_train  →  <base>/<model>_optimizer/network.onnx

        The learning rate is read from ``config["learning_rate"]`` (default 0.001).

        Returns:
            Path to the saved optimizer ONNX, or None if the output directory
            does not follow the ``_train`` naming convention.
        """
        from pathlib import Path

        from .optimizer_onnx import create_optimizer_onnx, derive_optimizer_dir

        train_dir = self.paths["output_dir"]
        opt_dir = derive_optimizer_dir(train_dir)
        if opt_dir is None:
            print("   ⚠️  Skipping optimizer ONNX: output dir must end with '_train'")
            return None

        lr = float(self.config.get("learning_rate", 0.001)) if self.config else 0.001
        opt_path = str(Path(opt_dir) / "network.onnx")

        print(f"\n⚙️  Building optimizer ONNX (lr={lr}) → {opt_path}")
        try:
            create_optimizer_onnx(train_dir=train_dir, output_path=opt_path, lr=lr)
            return opt_path
        except Exception as e:
            print(f"   ⚠️  Optimizer ONNX generation skipped: {e}")
            return None

    def export(self, mode: str = "train", save_path: Optional[str] = None) -> str:
        """
        Main export entry point.

        Args:
            mode: Export mode - "train", "infer", "train_single_step", or "quant"
            save_path: Optional custom save path

        Returns:
            Path to the exported ONNX file
        """
        if mode == "train":
            return self.export_training(save_path)
        elif mode == "infer":
            return self.export_inference(save_path)
        elif mode == "train_single_step":
            return self.export_training_single_step(save_path)
        elif mode == "quant":
            return self.export_quantized(save_path)
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'train', 'infer', 'train_single_step', or 'quant'"
            )

    # ---------------------------------------------------------------------- #
    # Quantized export via DeepQuant (Brevitas → QCDQ ONNX)                   #
    # ---------------------------------------------------------------------- #

    def export_quantized(self, save_path: Optional[str] = None) -> str:
        """
        Export the model to QCDQ ONNX via DeepQuant.

        Requires the exporter subclass to implement ``create_brevitas_model``.
        Calls ``DeepQuant.ExportBrevitas.exportBrevitas`` which produces an ONNX
        with decomposed Quant (Div/Add/Round/Clip) and Dequant (Sub/Mul) nodes.
        See ``docs/Quantization_Integration.md``.
        """
        try:
            from DeepQuant.ExportBrevitas import exportBrevitas
        except ImportError as exc:
            raise ImportError(
                "Quantized export requires DeepQuant. Install with:\n"
                "  git clone https://github.com/pulp-platform/DeepQuant.git\n"
                "  pip install -e DeepQuant\n"
                "and ensure 'brevitas' is installed."
            ) from exc

        if save_path:
            self.save_path = save_path

        self.config = self.load_config()
        self.paths = self.setup_paths(ExportMode.INFERENCE)

        print(f"\n{'='*60}")
        print(f"🚀 Exporting {self.get_model_name()} to QCDQ ONNX (Quantized Mode)")
        print(f"{'='*60}\n")

        print("📦 Creating Brevitas-quantized PyTorch model...")
        model = self.create_brevitas_model()
        model.eval()

        # Fold Conv → BatchNorm2d into a single biased Conv. Brevitas-quantized
        # models keep ``nn.BatchNorm2d`` as a separate module (Brevitas does
        # not auto-fuse), so the exported ONNX has a bare ``BatchNormalization``
        # op which Deeploy targets like Siracusa do not map. Folding here
        # produces a Conv that absorbs gamma/beta/running_mean/running_var
        # into its weight+bias before quantization, eliminating the BN node
        # from the final QCDQ graph.
        n_folded = _fold_conv_bn_inplace(model)
        if n_folded:
            print(f"   Folded {n_folded} Conv+BatchNorm pair(s) into Conv weights/bias.")

        input_shape = self.get_input_shape()
        example = torch.randn(*input_shape, dtype=torch.float32)
        print(f"   Input shape: {input_shape}")

        # One forward pass on random data initializes Brevitas's per-tensor
        # statistics. For production accuracy, replace this with a real PTQ
        # calibration loop (see docs/Quantization_Integration.md §9).
        print("\n📐 Running calibration forward pass (random input)...")
        with torch.no_grad():
            _ = model(example)

        print("\n📤 Exporting via DeepQuant.exportBrevitas...")
        # exportBrevitas writes to cwd; chdir to the output dir so the
        # network.onnx + inputs.npz + outputs.npz land alongside.
        from pathlib import Path
        import os

        out_dir = Path(self.paths["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        cwd_before = os.getcwd()
        try:
            os.chdir(out_dir)
            exportBrevitas(model, example)
        finally:
            os.chdir(cwd_before)

        # DeepQuant emits ``4_model_dequant_moved.onnx`` by default. Promote it
        # to the standard ``network.onnx`` filename so it slots into the rest
        # of the Onnx4Deeploy pipeline.
        deepquant_out = out_dir / "4_model_dequant_moved.onnx"
        target = Path(self.paths["network"])
        if deepquant_out.exists():
            import shutil

            shutil.copyfile(deepquant_out, target)
            print(f"✅ Renamed {deepquant_out.name} → {target.name}")

        print(f"\n{'='*60}")
        print("✅ Quantized Export Complete!")
        print(f"   Final model: {self.paths['network']}")
        print(f"   I/O fixtures: {out_dir / 'inputs.npz'}, {out_dir / 'outputs.npz'}")
        print(f"{'='*60}\n")

        return str(target)

    # ---------------------------------------------------------------------- #
    # Single-step training-as-inference                                       #
    # ---------------------------------------------------------------------- #

    def export_training_single_step(self, save_path: Optional[str] = None) -> str:
        """
        Export the training graph for per-tensor gradient verification.

        Reuses ``export_training`` end-to-end, then post-processes ``network.onnx``:
          1. Pin every ``tensor(bool)`` graph input (lazy_reset_grad and friends)
             to a constant initializer ``[True]`` so each InPlaceAccumulator
             output equals the pure batch dW (no historical accum).
          2. Regenerate ``inputs.npz`` (drop bool entries since they are now
             initializers) and ``outputs.npz`` (raw ORT-computed grad per
             graph output, instead of SGD-updated parameter values).

        Run via the inference path (``deeployRunner_*.py``) — Deeploy will
        compare every graph output (loss + each ``<param>_grad.accumulation.out``)
        against ORT and print per-tensor errors, pinpointing which gradient
        diverges in the integrated execution.
        """
        # 1. Standard training export — produces network_train.onnx, network.onnx,
        #    and the conventional inputs.npz / outputs.npz (which we will overwrite).
        self.export_training(save_path)

        # 2. Pin bool inputs as constant initializers in the deployed network.
        print("\n🪛 Single-step post-process: pinning bool inputs as constants...")
        self._pin_bool_inputs_as_constant(self.paths["network"], value=True)

        # 3. Regenerate inputs.npz / outputs.npz for inference-runner-style
        #    per-tensor verification.
        print("\n🧪 Single-step post-process: regenerating inputs/outputs for inference runner...")
        self._create_single_step_test_data()

        print(f"\n{'='*60}")
        print("✅ Single-step training-as-inference export complete")
        print(f"   network.onnx (lazy_reset_grad pinned True)")
        print(f"   inputs.npz   (no bool entries — match graph inputs)")
        print(f"   outputs.npz  (raw ORT grads — match graph outputs)")
        print(f"{'='*60}\n")
        return self.paths["network"]

    def _pin_bool_inputs_as_constant(self, onnx_path: str, value: bool = True) -> None:
        """
        Convert every ``tensor(bool)`` graph input into a constant initializer.

        Removes the input entry and adds an initializer with the same name
        carrying the scalar ``value`` (broadcast to the input's declared shape,
        defaulting to ``[1]`` when a 1-D shape is missing).
        """
        import numpy as np
        import onnx
        from onnx import TensorProto, numpy_helper

        model = onnx.load(onnx_path)
        bool_input_names = []
        kept_inputs = []
        for inp in model.graph.input:
            if inp.type.tensor_type.elem_type == TensorProto.BOOL:
                bool_input_names.append(inp.name)
            else:
                kept_inputs.append(inp)

        if not bool_input_names:
            print("   (no tensor(bool) inputs found; nothing to pin)")
            return

        # Rewrite graph.input in-place (clear+extend; ProtoBuf RepeatedField
        # disallows direct assignment).
        del model.graph.input[:]
        model.graph.input.extend(kept_inputs)

        for name in bool_input_names:
            # Hard-code as scalar [value]; matches lazy_reset_grad shape [1].
            const_arr = np.array([value], dtype=bool)
            init = numpy_helper.from_array(const_arr, name=name)
            model.graph.initializer.append(init)
            print(f"   pinned bool input '{name}' = [{value}]")

        onnx.save(model, onnx_path)

        # Re-run shape inference so downstream Deeploy sees consistent shapes
        # for the now-initialized lazy_reset_grad.
        try:
            from ..optimization.shape_optimizer import infer_shapes_with_custom_ops

            infer_shapes_with_custom_ops(onnx_path, onnx_path)
        except Exception as e:
            print(f"   ⚠️  Shape inference after pinning skipped: {e}")

    def _create_single_step_test_data(self) -> None:
        """
        Generate ``inputs.npz`` and ``outputs.npz`` for the single-step
        inference-style test:

        - ``inputs.npz``: every non-bool graph input (data, labels, params,
          grad-accumulation buffers initialized to zeros), keyed by input name
          in graph order. Bool inputs are now constant initializers, so they
          are NOT written here.
        - ``outputs.npz``: raw ORT-computed value for every graph output
          (loss + each ``<param>_grad.accumulation.out``), keyed by output
          name in graph output order.

        ORT runs against ``network_train.onnx`` (which still has the bool
        input) with ``lazy_reset_grad=True``, so the recorded grads are pure
        batch dW with no historical accumulation.
        """
        from pathlib import Path

        import numpy as np
        import onnxruntime as ort

        input_shape = self.get_input_shape()
        num_classes = self.config.get("num_classes", 2)
        save_dir = Path(self.paths["output_dir"])

        data_source = self.get_data_source()
        test_inputs, labels_list = data_source.load_batches(1, input_shape, num_classes, seed=42)
        test_input, labels = test_inputs[0], labels_list[0]

        # Initial parameter values from network_infer.onnx (matches checkpoint).
        infer_path = self.paths.get("network_infer", "")
        init_source = (
            infer_path if infer_path and os.path.exists(infer_path) else self.paths["network_train"]
        )
        init_map = self._load_init_map(init_source)

        # Run ORT against the original training graph (bool input still present)
        # with lazy_reset_grad=True → first-step semantics.
        session = ort.InferenceSession(
            self.paths["network_train"], providers=["CPUExecutionProvider"]
        )
        feed = self._build_input_feed(session, init_map, test_input, labels, lazy_reset_grad=True)
        output_names = [o.name for o in session.get_outputs()]
        output_values = session.run(None, feed)

        # inputs.npz — drop bool-typed entries (they are constants in network.onnx now).
        # Iterate in session.get_inputs() order so insertion order matches the
        # post-pinning graph input order.
        feed_no_bool: dict = {}
        for inp in session.get_inputs():
            if inp.type == "tensor(bool)":
                continue
            feed_no_bool[inp.name] = feed[inp.name]
        np.savez(save_dir / "inputs.npz", **feed_no_bool)
        print(
            f"   ✅ inputs.npz  — {len(feed_no_bool)} tensors "
            f"(bool inputs pinned as constants in network.onnx)"
        )

        # outputs.npz — raw grads, in graph output order.
        outputs_dict = dict(zip(output_names, output_values))
        np.savez(save_dir / "outputs.npz", **outputs_dict)
        print(
            f"   ✅ outputs.npz — {len(outputs_dict)} tensors "
            f"(loss + per-parameter raw dW from ORT)"
        )
