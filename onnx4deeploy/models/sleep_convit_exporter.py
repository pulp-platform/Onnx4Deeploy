# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""SleepConViT Model Exporter."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter

# Import SleepConViT PyTorch model from new location
from .pytorch_models.sleep_convit import SleepConViT

# Note: We use F.gelu in the model (functional interface) which has
# built-in ONNX export support.
# CascadedConcat uses torch.autograd.Function.symbolic for ONNX export,
# so no manual registration is needed.


class SleepConViTExporter(BaseONNXExporter):
    """ONNX exporter for SleepConViT model."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        """
        Initialize SleepConViT exporter.

        Args:
            save_path: Optional custom path to save ONNX files
            config_file: Path to configuration YAML file
        """
        super().__init__(save_path, config_file)
        self.model_config = {}

    def get_inference_pipeline(self):
        """
        Get SleepConViT-specific inference optimization pipeline.

        Uses transformer-specific optimizations (same as CCT) including:
        - ONNX Runtime transformer optimizer (fuses GELU, LayerNorm, etc.)
        - Standard inference optimizations

        When called during a training export (_for_training=True), the ORT transformer
        optimizer is skipped. That optimizer fuses ops into com.microsoft custom ops
        which lack standard ONNX shape inference support; generate_artifacts' internal
        infer_shapes_on_base() then cannot infer the SoftmaxCrossEntropyLoss output
        shape, causing a LookupError (onnx::loss::N is not a graph value info).

        Returns:
            OptimizationPipeline configured for SleepConViT inference
        """
        from ..core.optimization_passes import create_transformer_inference_pipeline

        # Use transformer pipeline with SleepConViT parameters.
        # Input shape is already 4D: (B, C, H, W).
        # Skip ORT transformer fusion when exporting for training.
        skip_ort = getattr(self, "_for_training", False)
        return create_transformer_inference_pipeline(
            embedding_dim=self.config["model_dim"],
            num_heads=self.config["num_heads"],
            input_shape=self.get_input_shape(),  # Already 4D: (B, 1, 1, 3000)
            skip_ort_transformer=skip_ort,
        )

    def load_config(self) -> Dict[str, Any]:
        """
        Load SleepConViT configuration.

        Returns:
            Dictionary containing SleepConViT configuration parameters
        """
        # Default SleepConViT configuration
        config = {
            "batch_size": 1,
            "input_channels": 1,
            "input_length": 3000,  # Time-series sequence length
            "model_dim": 48,
            "num_heads": 6,
            "num_patches": 94,  # Computed from ConvStem output
            "seq_len": 95,  # num_patches + 1 (CLS token)
            "attention_dropout": 0.0,  # No dropout for inference
            "mlp_head_hidden_dim": 48,
            "encoder_ff_dropout": 0.0,  # No dropout for inference
            "num_classes": 5,  # Sleep stages: Wake, N1, N2, N3, REM
            "opset_version": 17,  # Match CCT opset version for compatibility
            # Training configuration
            "training_strategy": "no_tokenizer",  # Options: "full", "no_tokenizer", "last_layer", "custom"
            "custom_trainable_params": [],
            # Training loop configuration
            "n_accum": 1,  # mini-batches per SGD update (gradient accumulation)
            # ZO training configuration
            "zo": {
                "epsilon": 0.1,
                "seed": 42,
                "noise_type": "uniform",
            },
        }

        # Apply any CLI overrides stored before export_training() was called.
        if hasattr(self, "_config_overrides") and self._config_overrides:
            config.update(self._config_overrides)

        self.model_config = config
        return config

    def create_model(self) -> torch.nn.Module:
        """
        Create SleepConViT PyTorch model.

        Returns:
            SleepConViT model ready for export
        """
        model = SleepConViT(config=self.model_config)
        return model

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the input tensor shape for SleepConViT.

        Returns:
            Tuple representing input shape (batch_size, channels, height, width)
            Shape: (B, 1, 1, 3000) for compatibility with ViT/transformer optimizer
        """
        batch_size = self.config["batch_size"]
        channels = self.config["input_channels"]
        length = self.config["input_length"]
        return (batch_size, channels, 1, length)  # 4D: (B, C, H, W)

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """
        Get list of trainable parameter names for SleepConViT.

        Supports multiple training strategies:
        - "full": Train all parameters (default)
        - "last_layer": Only train the final classification layer
        - "custom": Use custom_trainable_params from config

        Args:
            all_param_names: List of all parameter names in the model

        Returns:
            List of parameter names that should be trainable
        """
        strategy = self.config.get("training_strategy", "full")

        # Define training strategies
        strategy_params = {
            "full": all_param_names,  # Train everything
            "no_tokenizer": [name for name in all_param_names if "conv_stem" not in name],
            "last_layer": [
                "classifier.lin1.weight",
                "classifier.lin1.bias",
            ],
            "custom": self.config.get("custom_trainable_params", []),
        }

        # Get trainable params based on strategy
        if strategy not in strategy_params:
            print(f"⚠️  Unknown training strategy '{strategy}', using 'full' as fallback")
            strategy = "full"

        trainable_params = strategy_params[strategy]

        # Filter to only include params that exist in the model
        requires_grad = [name for name in all_param_names if name in trainable_params]

        # Print strategy info
        print(f"\n🎯 Training Strategy: '{strategy}'")
        print(f"   Total params in model: {len(all_param_names)}")
        print(f"   Params to train: {len(requires_grad)}")
        print(f"   Frozen params: {len(all_param_names) - len(requires_grad)}")

        return requires_grad

    def _get_config_string(self) -> str:
        """
        Get configuration string for folder naming.

        Returns:
            Configuration string like "_3000_48d_8h_5cls"
        """
        return (
            f"_{self.config['input_length']}"
            f"_{self.config['model_dim']}d"
            f"_{self.config['num_heads']}h"
            f"_{self.config['num_classes']}cls"
        )

    def save_test_data(self, model: torch.nn.Module, save_dir: str):
        """
        Save test input/output data for validation.

        Loads ONNX weights into PyTorch model to ensure outputs.npz matches ONNX exactly.
        This handles ONNX-specific optimizations like MatMul transpose and weight-sharing.

        Args:
            model: PyTorch model to load ONNX weights into
            save_dir: Directory to save test data
        """
        import onnx

        print("💾 Saving test input/output data...")

        # Create test input
        input_shape = self.get_input_shape()
        test_input = np.random.randn(*input_shape).astype(np.float32)

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Load ONNX weights
        onnx_path = save_path / "network.onnx"

        if not onnx_path.exists():
            print(f"  ⚠️  ONNX model not found at {onnx_path}")
            print("  ⚠️  Run export() first, then call save_test_data()")
            return

        # Load ONNX weights
        print(f"  📦 Loading ONNX weights from {onnx_path.name}...")
        onnx_model = onnx.load(str(onnx_path))
        onnx_weights = {}

        for init in onnx_model.graph.initializer:
            if init.data_type == 1:  # FLOAT
                data = np.frombuffer(init.raw_data, dtype=np.float32)
            elif init.data_type == 7:  # INT64
                data = np.frombuffer(init.raw_data, dtype=np.int64)
            elif init.data_type == 6:  # INT32
                data = np.frombuffer(init.raw_data, dtype=np.int32)
            else:
                continue

            try:
                onnx_weights[init.name] = data.reshape(init.dims) if init.dims else data
            except:
                onnx_weights[init.name] = data

        print(f"  ✅ Loaded {len(onnx_weights)} weights from ONNX")

        # Map ONNX weights to PyTorch (with MatMul transpose handling)
        state_dict = {}
        pytorch_params = {name: param for name, param in model.named_parameters()}

        onnx_to_pytorch_map = {
            "conv_stem_branch1_0_weight": "conv_stem.branch1.0.weight",
            "conv_stem_branch1_3_weight": "conv_stem.branch1.3.weight",
            "conv_stem_branch2_0_weight": "conv_stem.branch2.0.weight",
            "conv_stem_branch2_3_weight": "conv_stem.branch2.3.weight",
            "conv_stem_branch3_0_weight": "conv_stem.branch3.0.weight",
            "conv_stem_branch3_3_weight": "conv_stem.branch3.3.weight",
            "pos_embed": "pos_embed",
            "node_0_Expand__0": "cls_token",
            "cls_selector": "cls_selector",
            "encoder_ln_1_weight": "encoder.ln_1.weight",
            "encoder_ln_1_bias": "encoder.ln_1.bias",
            "onnx__MatMul_109": "encoder.mha.q_proj.weight",
            "onnx__MatMul_110": "encoder.mha.k_proj.weight",
            "onnx__MatMul_111": "encoder.mha.v_proj.weight",
            "onnx__MatMul_112": "encoder.mha.out_proj.weight",
            "onnx__MatMul_113": "encoder.ff.ff1.weight",
            "onnx__MatMul_114": "encoder.ff.ff2.weight",
            "classifier_lin1_weight": "classifier.lin1.weight",
            "classifier_lin1_bias": "classifier.lin1.bias",
        }

        for onnx_name, pytorch_name in onnx_to_pytorch_map.items():
            if onnx_name in onnx_weights and pytorch_name in pytorch_params:
                onnx_weight = onnx_weights[onnx_name]

                # CRITICAL: ONNX MatMul weights must be transposed for PyTorch Linear
                needs_transpose = "MatMul" in onnx_name and ".weight" in pytorch_name

                if needs_transpose and len(onnx_weight.shape) == 2:
                    state_dict[pytorch_name] = torch.from_numpy(onnx_weight.T.copy()).float()
                else:
                    state_dict[pytorch_name] = torch.from_numpy(onnx_weight.copy()).float()

        # Load mapped weights
        model.load_state_dict(state_dict, strict=False)
        print(f"  ✅ Loaded {len(state_dict)} weights into PyTorch model")

        # Apply ONNX weight-sharing optimization
        if "encoder.ln_1.weight" in state_dict and "encoder.ln_1.bias" in state_dict:
            ln_weight = state_dict["encoder.ln_1.weight"]
            ln_bias = state_dict["encoder.ln_1.bias"]

            # All LayerNorms reuse encoder.ln_1 weights in ONNX
            if hasattr(model.encoder, "ln_2"):
                model.encoder.ln_2.weight.data.copy_(ln_weight)
                model.encoder.ln_2.bias.data.copy_(ln_bias)

            if hasattr(model, "norm"):
                model.norm.weight.data.copy_(ln_weight)
                model.norm.bias.data.copy_(ln_bias)

            # All Linear biases use encoder.ln_1.bias in ONNX
            if (
                hasattr(model.encoder.mha.out_proj, "bias")
                and model.encoder.mha.out_proj.bias is not None
            ):
                model.encoder.mha.out_proj.bias.data.copy_(ln_bias)

            if hasattr(model.encoder.ff.ff1, "bias") and model.encoder.ff.ff1.bias is not None:
                model.encoder.ff.ff1.bias.data.copy_(ln_bias)

            if hasattr(model.encoder.ff.ff2, "bias") and model.encoder.ff.ff2.bias is not None:
                model.encoder.ff.ff2.bias.data.copy_(ln_bias)

            print(f"  ✅ Applied ONNX weight-sharing (ln_1 → ln_2, norm, Linear biases)")

        # Generate output with ONNX weights
        model.eval()
        with torch.no_grad():
            input_tensor = torch.from_numpy(test_input).float()
            output_tensor = model(input_tensor)
            test_output = output_tensor.numpy()

        # Save as .npz files
        np.savez(save_path / "inputs.npz", input=test_input)
        np.savez(save_path / "outputs.npz", output=test_output)

        print(f"  ✅ Saved: {save_path / 'inputs.npz'} shape={test_input.shape}")
        print(f"  ✅ Saved: {save_path / 'outputs.npz'} shape={test_output.shape}")
        print(f"  📊 Output values: {test_output.flatten()}")

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
        num_classes = self.config.get("num_classes", 5)
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
                grad_tensor_name = node.input[1]  # e.g. "fc1_weight_grad"
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
                    feed_mb0 = {
                        k: v.copy() if hasattr(v, "copy") else v for k, v in feed.items()
                    }  # snapshot mb=0 feed (initial weights)

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
        save_dict["meta_n_accum"] = np.array([n_accum], dtype=np.int32)
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
