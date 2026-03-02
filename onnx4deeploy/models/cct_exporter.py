# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""CCT (Compact Convolutional Transformer) Model Exporter."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter
from ..transform.model_transform import randomize_layernorm_params

# Import CCT PyTorch models from new location
from .pytorch_models.cct import cct_test


class CCTExporter(BaseONNXExporter):
    """ONNX exporter for CCT (Compact Convolutional Transformer) model."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        """
        Initialize CCT exporter.

        Args:
            save_path: Optional custom path to save ONNX files
            config_file: Path to configuration YAML file
        """
        super().__init__(save_path, config_file)
        self.model_config = {}

    def load_config(self) -> Dict[str, Any]:
        """
        Load CCT configuration.

        Returns:
            Dictionary containing CCT configuration parameters
        """
        # Default CCT configuration for testing
        config = {
            "batch_size": 1,
            "img_size": 8,  # Small image for L2-constrained testing (32 → 8)
            "embedding_dim": 32,  # Reduced from 128: avoids PULP L2 overflow with frozen weights
            "num_heads": 2,
            "num_layers": 2,
            "num_classes": 10,
            "opset_version": 17,  # LayerNormalization requires opset 17+
            "n_conv_layers": 1,
            "kernel_size": 3,
            "positional_embedding": "learnable",
            # Training configuration
            "training_strategy": "no_tokenizer",  # Options: "no_tokenizer", "last_block", "linear", "full", "custom"
            "custom_trainable_params": [],  # Used when training_strategy = "custom"
            # Training loop configuration
            "learning_rate": 0.001,
            "n_batches": 4,
            "n_accum": 1,
        }

        # Apply any CLI overrides stored before export_training() was called.
        if hasattr(self, "_config_overrides") and self._config_overrides:
            config.update(self._config_overrides)

        self.model_config = config
        return config

    def create_model(self) -> torch.nn.Module:
        """
        Create CCT PyTorch model.

        Returns:
            CCT model ready for export
        """
        model = cct_test(
            pretrained=False,
            img_size=self.model_config["img_size"],
            num_classes=self.model_config["num_classes"],
            embedding_dim=self.model_config["embedding_dim"],
            num_heads=self.model_config["num_heads"],
            num_layers=self.model_config["num_layers"],
            n_conv_layers=self.model_config.get("n_conv_layers", 1),
            positional_embedding=self.model_config.get("positional_embedding", "learnable"),
            stochastic_depth=0.0,  # Disable DropPath: no RandomUniformLike in ONNX
            dropout=0.0,  # Disable Dropout: no Dropout op in ONNX
            attention_dropout=0.0,  # Disable attention Dropout: no Dropout op in ONNX
        )

        # Randomize LayerNorm parameters (for testing)
        model = randomize_layernorm_params(model)

        return model

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the input tensor shape for CCT.

        Returns:
            Tuple representing input shape (batch_size, channels, height, width)
        """
        batch_size = self.config["batch_size"]
        img_size = self.config["img_size"]
        return (batch_size, 3, img_size, img_size)

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """
        Get list of trainable parameter names for CCT based on training strategy.

        Uses pattern-based filtering on ONNX initializer names (robust across pipeline
        versions, unlike hardcoded lists).

        Strategies:
        - "no_tokenizer" (default): Freeze only Conv tokenizer; train all transformer
          blocks (attention + MLP + norms + FC + positional_emb + attention_pool).
          Frozen constants: only tokenizer_conv_layers_* (~3 KB) — fits in PULP L2.
        - "last_block": Freeze tokenizer + positional_emb + blocks_0_*; train block 1
          + attention_pool + norm + FC.
        - "linear": Only train the final FC layer (classifier_fc_weight/bias).
        - "full": Train all parameters (no frozen constants).
        - "custom": Use custom_trainable_params list from config.

        Args:
            all_param_names: List of all ONNX initializer names

        Returns:
            List of parameter names that should be trainable
        """
        strategy = self.config.get("training_strategy", "no_tokenizer")

        # Pattern-based strategies — freeze params whose names match the freeze pattern.
        # This is robust to ONNX pipeline renames (no hardcoded node_0_* names).
        _FREEZE = {
            # freeze only Conv tokenizer → train all transformer layers
            "no_tokenizer": lambda n: "tokenizer" in n,
            # freeze tokenizer + positional emb + first block → train last block only
            "last_block": lambda n: "tokenizer" in n or "positional_emb" in n or "blocks_0" in n,
            # freeze everything except FC
            "linear": lambda n: n not in {"classifier_fc_weight", "classifier_fc_bias"},
            # freeze nothing
            "full": lambda n: False,
            # freeze nothing from the explicit list (keep user-supplied names)
            "custom": lambda n: n not in self.config.get("custom_trainable_params", []),
        }

        if strategy not in _FREEZE:
            print(f"⚠️  Unknown training strategy '{strategy}', using 'no_tokenizer' as fallback")
            strategy = "no_tokenizer"

        requires_grad = [n for n in all_param_names if not _FREEZE[strategy](n)]

        frozen = [n for n in all_param_names if _FREEZE[strategy](n)]
        print(f"\n🎯 Training Strategy: '{strategy}'")
        print(
            f"   Total params: {len(all_param_names)}  |  Trainable: {len(requires_grad)}  |  Frozen: {len(frozen)}"
        )
        if frozen:
            print(f"   Frozen (→ constant): {frozen}")

        return requires_grad

    def _get_config_string(self) -> str:
        """
        Get configuration string for folder naming.

        Returns:
            Configuration string like "_32_128_2_2"
        """
        return f"_{self.config['img_size']}_{self.config['embedding_dim']}_{self.config['num_heads']}_{self.config['num_layers']}"

    def get_inference_pipeline(self):
        """
        Get CCT-specific inference optimization pipeline.

        CCT uses transformer-specific optimizations including:
        - Randomize initializers (for testing)
        - ONNX Runtime transformer optimizer (includes LayerNorm fusion)
        - Standard inference optimizations (GEMM, Identity, etc.)

        When called during a training export (_for_training=True), the ORT transformer
        optimizer is skipped. That optimizer fuses ops into com.microsoft custom ops
        (FusedMatMul, BiasGelu) which lack standard ONNX shape inference support;
        generate_artifacts' internal infer_shapes_on_base() then cannot properly build
        the backward pass with InPlaceAccumulatorV2, resulting in only 2 explicit graph
        inputs instead of the required (data + weights + grad_acc_bufs) structure.

        Returns:
            OptimizationPipeline configured for CCT inference
        """
        from ..core.optimization_passes import create_transformer_inference_pipeline

        # Skip ORT transformer fusion when exporting for training.
        skip_ort = getattr(self, "_for_training", False)

        # Create transformer-specific pipeline with model parameters
        pipeline = create_transformer_inference_pipeline(
            embedding_dim=self.config["embedding_dim"],
            num_heads=self.config["num_heads"],
            input_shape=self.get_input_shape(),
            skip_ort_transformer=skip_ort,
        )

        return pipeline

    # run_training_optimization() is intentionally NOT overridden here.
    # The base class implementation calls run_train_onnx_optimization() which runs
    # the full 18-pass standard pipeline — including frozen-param folding,
    # BiasGelu unfusion, SCELossGrad 3→2 input, ReduceSum axes→attribute, etc.
    # This is the same approach used by SleepConViT.

    def save_test_data(self, model: torch.nn.Module, save_dir: str):
        """
        Save test input/output data for validation.

        Uses PyTorch model to generate reference output for validating ONNX correctness.

        Args:
            model: PyTorch model to run inference with
            save_dir: Directory to save test data
        """
        print("💾 Saving test input/output data...")

        # Create test input
        input_shape = self.get_input_shape()
        test_input = np.random.randn(*input_shape).astype(np.float32)

        # Get PyTorch output (reference for validating ONNX)
        # Temporarily switch to eval mode if in training mode
        was_training = model.training
        model.eval()

        with torch.no_grad():
            input_tensor = torch.from_numpy(test_input)
            output_tensor = model(input_tensor)
            test_output = output_tensor.numpy()

        # Restore training mode if needed
        if was_training:
            model.train()

        # Save as .npz files
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        np.savez(save_path / "inputs.npz", input=test_input)
        np.savez(save_path / "outputs.npz", output=test_output)

        print("  ✅ Saved test data (PyTorch reference):")
        print(f"     Input:  {save_path / 'inputs.npz'} shape={test_input.shape}")
        print(f"     Output: {save_path / 'outputs.npz'} shape={test_output.shape}")

    def create_training_test_data(
        self, n_batches: int = None, num_data_inputs: int = 2, n_accum: int = None
    ) -> None:
        """
        Save test input/output data for training mode validation.

        Saves inputs.npz WITHOUT grad accumulation buffer entries (those are
        zero-initialised by the C harness via memset after InitTrainingNetwork).
        Generates ``n_batches`` distinct (input, labels) pairs so the harness
        can exercise different data on every mini-batch step.

        Gradient accumulation is supported via ``n_accum``: n_batches must be
        divisible by n_accum.  SGD is applied once every n_accum mini-batches.

        inputs.npz layout
        -----------------
        Base entries (positional over non-grad-buf graph inputs):
          arr_0000 … arr_{M-1}  — all non-grad-buf graph inputs in order
                                  (data[0], data[1], weights…, ctrl)

        Per-mini-batch DATA entries (mb 1 … n_batches-1):
          mb{I}_arr_{J:04d}     — DATA input J for mini-batch I
                                  (only the first ``num_data_inputs`` buffers)

        outputs.npz layout
        ------------------
          <param_name>  — SGD-updated weight tensors (after final optimizer step)
          loss          — reference loss for each mini-batch, shape (n_batches,)
        """
        import onnx
        import onnxruntime as ort
        from onnx import numpy_helper

        _GRAD_ACC = "_grad.accumulation.buffer"

        # Resolve n_batches / n_accum from explicit args > config > defaults.
        if n_batches is None:
            n_batches = self.config.get("n_batches", 4)
        if n_accum is None:
            n_accum = int(self.config.get("n_accum", 1))
        if n_batches % n_accum != 0:
            raise ValueError(f"n_batches={n_batches} must be divisible by n_accum={n_accum}")
        n_steps = n_batches // n_accum

        # Determine effective_data_size: only unique samples stored in NPZ/C header.
        _data_size_cfg = self.config.get("data_size", None)
        effective_data_size = (
            int(_data_size_cfg)
            if (_data_size_cfg and int(_data_size_cfg) < n_batches)
            else n_batches
        )

        save_dir = Path(self.paths["output_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(42)
        input_shape = self.get_input_shape()
        batch_size = input_shape[0]
        num_classes = self.config.get("num_classes", 10)
        learning_rate = float(self.config.get("learning_rate", 0.001))

        print(
            f"   Training sim: n_batches={n_batches}  n_accum={n_accum}  n_steps={n_steps}  lr={learning_rate}"
            + (
                f"  data_size={effective_data_size} (cycling)"
                if effective_data_size < n_batches
                else ""
            )
        )

        # Generate effective_data_size distinct (input, labels) pairs; cycle via modulo for ORT.
        test_inputs = [
            np.random.randn(*input_shape).astype(np.float32) for _ in range(effective_data_size)
        ]
        labels_list = [
            np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int64)
            for _ in range(effective_data_size)
        ]

        # Read initial parameter values from the inference model.
        infer_model = onnx.load(self.paths["network_infer"])
        init_map: dict = {
            init.name: numpy_helper.to_array(init) for init in infer_model.graph.initializer
        }

        # Load network_train.onnx and expose per-step gradient tensors as extra outputs
        # so we can manually accumulate them across n_accum mini-batches.
        # InPlaceAccumulatorV2 nodes: input[1] = per-step gradient tensor name.
        train_model_onnx = onnx.load(self.paths["network_train"])
        grad_tensor_map: dict = {}  # param_name -> grad_tensor_name (intermediate)
        for node in train_model_onnx.graph.node:
            if "InPlaceAccumulator" in node.op_type and len(node.input) >= 2:
                grad_tensor_name = node.input[1]  # e.g. "classifier_fc_weight_grad"
                if grad_tensor_name.endswith("_grad"):
                    param_name = grad_tensor_name[:-5]  # strip "_grad"
                    grad_tensor_map[param_name] = grad_tensor_name

        # Add gradient tensors as explicit graph outputs in an in-memory copy.
        for grad_name in grad_tensor_map.values():
            vi = onnx.helper.make_tensor_value_info(grad_name, onnx.TensorProto.FLOAT, None)
            train_model_onnx.graph.output.append(vi)

        session = ort.InferenceSession(
            train_model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        session_output_names = [o.name for o in session.get_outputs()]
        print(f"   Training model inputs:  {[i.name for i in session.get_inputs()]}")
        print(f"   Training model outputs: {session_output_names}")

        current_weights = {k: v.copy() for k, v in init_map.items()}
        all_losses: list = []
        feed_mb0: dict = {}  # snapshot of mb=0 feed (initial weights + mb0 data)

        for update_step in range(n_steps):
            # Zero-initialise accumulated gradients at the start of each update step.
            accumulated_grads = {
                pname: np.zeros_like(current_weights[pname])
                for pname in grad_tensor_map
                if pname in current_weights
            }

            for accum_step in range(n_accum):
                mb = update_step * n_accum + accum_step

                feed = {}
                for inp in session.get_inputs():
                    name = inp.name
                    shape = [d for d in inp.shape if isinstance(d, int) and d > 0]
                    if inp.type == "tensor(int64)":
                        feed[name] = labels_list[mb % effective_data_size]
                    elif inp.type == "tensor(bool)":
                        # lazy_reset_grad: True on first accum step, False otherwise.
                        feed[name] = np.array([accum_step == 0])
                    elif name in current_weights:
                        feed[name] = current_weights[name]
                    elif _GRAD_ACC in name:
                        feed[name] = np.zeros(shape, dtype=np.float32)
                    elif shape == list(input_shape):
                        feed[name] = test_inputs[mb % effective_data_size]
                    else:
                        feed[name] = np.zeros(shape, dtype=np.float32)

                if mb == 0:
                    feed_mb0 = dict(feed)  # snapshot mb=0 feed (initial weights)

                raw_outputs = session.run(None, feed)
                outputs_raw = dict(zip(session_output_names, raw_outputs))

                # Collect loss.
                for out_name, out_val in outputs_raw.items():
                    if "loss" in out_name.lower() and "grad" not in out_name.lower():
                        all_losses.append(float(np.array(out_val).flatten()[0]))
                        break

                # Accumulate per-step gradients manually.
                for pname, grad_name in grad_tensor_map.items():
                    if grad_name in outputs_raw and pname in accumulated_grads:
                        accumulated_grads[pname] = accumulated_grads[pname] + outputs_raw[grad_name]

            # SGD weight update after n_accum steps using accumulated gradients.
            for pname, acc_grad in accumulated_grads.items():
                current_weights[pname] = current_weights[pname] - learning_rate * acc_grad

        # outputs_dict: weights after all optimizer steps + all losses.
        outputs_dict: dict = {k: v for k, v in current_weights.items()}
        outputs_dict["loss"] = np.array(all_losses, dtype=np.float32)
        print(f"   Collected {len(all_losses)} reference losses: {all_losses}")

        # ------------------------------------------------------------------ #
        # Build inputs.npz: re-order by network.onnx input order and SKIP    #
        # grad accumulation buffer entries (they are zero-init'd by harness). #
        # ------------------------------------------------------------------ #
        final_model = onnx.load(self.paths["network"])
        final_input_names = [inp.name for inp in final_model.graph.input]

        # Separate grad-acc-buf names from the rest.
        grad_acc_names = {n for n in final_input_names if _GRAD_ACC in n}
        non_grad_names = [n for n in final_input_names if n not in grad_acc_names]

        # Base entries (sequential arr_0000 … arr_{M-1} over non-grad inputs).
        # Use feed_mb0: mb=0 data + INITIAL weights (not the loop-end updated weights).
        save_dict: dict = {}
        for npz_idx, name in enumerate(non_grad_names):
            if name in feed_mb0:
                save_dict[f"arr_{npz_idx:04d}"] = feed_mb0[name]
            else:
                print(f"   ⚠️  network.onnx non-grad input '{name}' not found in feed — skipping")

        # Per-mini-batch DATA entries for mb 1 … effective_data_size-1 (unique samples only).
        # The C harness cycles via mb % TRAINING_DATA_SIZE, so no duplicates are stored.
        session_type: dict = {inp.name: inp.type for inp in session.get_inputs()}
        data_names = non_grad_names[:num_data_inputs]
        for mb in range(1, effective_data_size):
            for buf_idx, data_name in enumerate(data_names):
                inp_type = session_type.get(data_name, "tensor(float)")
                if inp_type == "tensor(int64)":
                    save_dict[f"mb{mb}_arr_{buf_idx:04d}"] = labels_list[mb]
                else:
                    save_dict[f"mb{mb}_arr_{buf_idx:04d}"] = test_inputs[mb]

        save_dict["meta_data_size"] = np.array([effective_data_size], dtype=np.int32)
        save_dict["meta_n_batches"] = np.array([n_batches], dtype=np.int32)
        np.savez(save_dir / "inputs.npz", **save_dict)
        n_params = sum(1 for n in non_grad_names if n in init_map)
        n_grad = len(grad_acc_names)
        print(
            f"   ✅ inputs.npz  — {len(non_grad_names)} base tensors "
            f"(data + {n_params} params + ctrl; {n_grad} grad-acc-buf(s) omitted) "
            f"+ {(effective_data_size - 1) * num_data_inputs} unique DATA entries "
            f"({effective_data_size} unique samples, {n_batches} total mini-batches with cycling)"
        )

        np.savez(save_dir / "outputs.npz", **outputs_dict)
        n_updated = sum(1 for k in outputs_dict if k in init_map)
        print(
            f"   ✅ outputs.npz — {len(outputs_dict)} tensors ({n_updated} updated params + loss)"
        )
