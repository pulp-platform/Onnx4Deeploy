# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""FC Autoencoder Exporter — MLperf Tiny Anomaly Detection (AD) benchmark.

Supports inference and full training-graph generation.

Loss: MSELoss (reconstruction: output ≈ input).
  ORT generate_artifacts() receives the "labels" tensor which equals the input
  features — the model is trained to reconstruct its own input.

Anomaly detection at runtime:
  reconstruction_error = MSE(model(x), x)
  anomaly = reconstruction_error > threshold

Default configuration (PULP-deployable "tiny" variant):
  Input:  (1, 128)  — 128-dim log-mel spectrogram feature vector
  Hidden: [64, 32, 64]  — symmetric encoder-decoder
  ~26 K parameters, fits in PULP L2

Full MLperf Tiny AD reference variant ("mlperf"):
  Hidden: [128, 128, 128]
  ~100 K parameters — may require training_strategy="last_layer" on PULP
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter


class AutoencoderExporter(BaseONNXExporter):
    """ONNX exporter for FC Autoencoder (MLperf Tiny Anomaly Detection)."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        super().__init__(save_path, config_file)
        self.model_config = {}

    # ------------------------------------------------------------------ #
    # Loss type override — MSE for reconstruction                          #
    # ------------------------------------------------------------------ #

    def get_loss_type(self):
        """Override to use MSELoss for autoencoder reconstruction training."""
        from onnxruntime.training import artifacts

        return artifacts.LossType.MSELoss

    # ------------------------------------------------------------------ #
    # Configuration                                                        #
    # ------------------------------------------------------------------ #

    def load_config(self) -> Dict[str, Any]:
        config = {
            "batch_size": 1,
            "input_dim": 128,  # Feature vector length (128 for MLperf Tiny AD)
            "hidden_dims": [64, 32, 64],  # [128, 128, 128] for full MLperf Tiny reference
            "variant": "tiny",  # "tiny" (PULP) | "mlperf" (reference)
            "opset_version": 17,
            # Training
            "training_strategy": "full",  # "full" | "encoder_only" | "decoder_only" | "custom"
            "custom_trainable_params": [],
            "learning_rate": 0.001,
            "n_batches": 4,
            "n_accum": 1,
            "data_size": None,
        }

        if hasattr(self, "_config_overrides") and self._config_overrides:
            config.update(self._config_overrides)

        # Resolve hidden_dims from variant if not explicitly set
        overrides = getattr(self, "_config_overrides", {}) or {}
        if "hidden_dims" not in overrides:
            if config.get("variant") == "mlperf":
                config["hidden_dims"] = [128, 128, 128]
            else:
                config["hidden_dims"] = [64, 32, 64]

        self.model_config = config
        return config

    # ------------------------------------------------------------------ #
    # Model factory                                                        #
    # ------------------------------------------------------------------ #

    def create_model(self) -> torch.nn.Module:
        from .pytorch_models.autoencoder.autoencoder import FCAutoencoder

        return FCAutoencoder(
            input_dim=self.model_config["input_dim"],
            hidden_dims=self.model_config["hidden_dims"],
        )

    # ------------------------------------------------------------------ #
    # Shape helpers                                                        #
    # ------------------------------------------------------------------ #

    def get_input_shape(self) -> Tuple[int, ...]:
        return (self.config["batch_size"], self.config["input_dim"])

    def _get_config_string(self) -> str:
        v = self.config.get("variant", "tiny")
        dims = "x".join(str(d) for d in self.config["hidden_dims"])
        return f"_ae_{v}_{self.config['input_dim']}_{dims}"

    # ------------------------------------------------------------------ #
    # Training strategy                                                   #
    # ------------------------------------------------------------------ #

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """
        Pattern-based trainable parameter selection.

        Strategies:
        - "full":         Train all parameters (default).
        - "encoder_only": Freeze decoder; train only encoder layers.
        - "decoder_only": Freeze encoder; train only decoder layers.
        - "custom":       Explicit list from config["custom_trainable_params"].
        """
        strategy = self.config.get("training_strategy", "full")

        _FREEZE = {
            "full": lambda n: False,
            "encoder_only": lambda n: "decoder" in n,
            "decoder_only": lambda n: "encoder" in n,
            "custom": lambda n: n not in self.config.get("custom_trainable_params", []),
        }

        if strategy not in _FREEZE:
            print(f"⚠️  Unknown strategy '{strategy}', using 'full'")
            strategy = "full"

        requires_grad = [n for n in all_param_names if not _FREEZE[strategy](n)]
        frozen = [n for n in all_param_names if _FREEZE[strategy](n)]

        print(f"\n🎯 Training Strategy: '{strategy}'")
        print(
            f"   Total: {len(all_param_names)}  Trainable: {len(requires_grad)}  Frozen: {len(frozen)}"
        )
        if frozen:
            print(f"   Frozen (→ constant): {frozen[:5]}{'…' if len(frozen) > 5 else ''}")
        return requires_grad

    # ------------------------------------------------------------------ #
    # Inference test data                                                  #
    # ------------------------------------------------------------------ #

    def save_test_data(self, model: torch.nn.Module, save_dir: str):
        print("💾 Saving inference test data...")
        input_shape = self.get_input_shape()
        test_input = np.random.randn(*input_shape).astype(np.float32)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            test_output = model(torch.from_numpy(test_input)).numpy()
        if was_training:
            model.train()

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        np.savez(save_path / "inputs.npz", input=test_input)
        np.savez(save_path / "outputs.npz", output=test_output)
        print(f"   Input: {test_input.shape}  Output: {test_output.shape}")

    # ------------------------------------------------------------------ #
    # Training test data                                                   #
    # ------------------------------------------------------------------ #

    def create_training_test_data(
        self, n_batches: int = None, num_data_inputs: int = 2, n_accum: int = None
    ) -> None:
        """
        Save inputs.npz / outputs.npz for autoencoder training validation.

        Key difference from classification exporters:
        - No int64 label input. ORT MSELoss expects a float "labels" tensor == input features.
        - The "label" fed to ORT is the input itself (reconstruction target).
        - outputs.npz contains updated weights + per-mini-batch MSE loss values.
        """
        import onnx
        import onnxruntime as ort

        if n_batches is None:
            n_batches = self.config.get("n_batches", 4)
        if n_accum is None:
            n_accum = int(self.config.get("n_accum", 1))
        if n_batches % n_accum != 0:
            n_batches = max((n_batches // n_accum) * n_accum, n_accum)
            print(f"   n_batches adjusted to {n_batches} (must be divisible by n_accum={n_accum})")
        n_steps = n_batches // n_accum

        save_dir = Path(self.paths["output_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        input_shape = self.get_input_shape()
        learning_rate = float(self.config.get("learning_rate", 0.001))

        print(
            f"   Training sim: n_batches={n_batches}  n_accum={n_accum}  n_steps={n_steps}  lr={learning_rate}"
        )

        _data_size_cfg = self.config.get("data_size", None)
        effective_data_size = (
            int(_data_size_cfg)
            if (_data_size_cfg and int(_data_size_cfg) < n_batches)
            else n_batches
        )

        # Autoencoder: labels = inputs (reconstruction target)
        rng = np.random.default_rng(42)
        test_inputs = [
            rng.standard_normal(input_shape).astype(np.float32) for _ in range(effective_data_size)
        ]
        # No separate label list — labels are the inputs themselves

        init_map: dict = self._load_init_map(self.paths["network_infer"])

        train_model_onnx = onnx.load(self.paths["network_train"])
        grad_tensor_map: dict = {}
        for node in train_model_onnx.graph.node:
            if "InPlaceAccumulator" in node.op_type and len(node.input) >= 2:
                grad_tensor_name = node.input[1]
                if grad_tensor_name.endswith("_grad"):
                    grad_tensor_map[grad_tensor_name[:-5]] = grad_tensor_name

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
        feed_mb0: dict = {}

        for update_step in range(n_steps):
            accumulated_grads = {
                pname: np.zeros_like(current_weights[pname])
                for pname in grad_tensor_map
                if pname in current_weights
            }

            for accum_step in range(n_accum):
                mb = update_step * n_accum + accum_step
                data = test_inputs[mb % effective_data_size]

                # Autoencoder: reconstruction target == input features.
                # No int64 labels — _build_input_feed rule 1 never fires.
                # Both the data input and the MSE "labels" input share the same shape,
                # so rule 5 assigns `data` to both (passing data as labels too).
                feed = self._build_input_feed(
                    session,
                    param_values=current_weights,
                    test_input=data,
                    labels=data,
                    lazy_reset_grad=(accum_step == 0),
                )

                if mb == 0:
                    feed_mb0 = {k: v.copy() if hasattr(v, "copy") else v for k, v in feed.items()}

                raw_outputs = session.run(None, feed)
                outputs_raw = dict(zip(session_output_names, raw_outputs))

                for out_name, out_val in outputs_raw.items():
                    if "loss" in out_name.lower() and "grad" not in out_name.lower():
                        all_losses.append(float(np.array(out_val).flatten()[0]))
                        break

                for pname, grad_name in grad_tensor_map.items():
                    if grad_name in outputs_raw and pname in accumulated_grads:
                        accumulated_grads[pname] += outputs_raw[grad_name]

            for pname, acc_grad in accumulated_grads.items():
                current_weights[pname] -= learning_rate * acc_grad

        outputs_dict: dict = {k: v for k, v in current_weights.items()}
        outputs_dict["loss"] = np.array(all_losses, dtype=np.float32)
        print(f"   Reference losses: {all_losses}")

        final_model = onnx.load(self.paths["network"])
        final_input_names = [inp.name for inp in final_model.graph.input]
        grad_acc_names = {n for n in final_input_names if self._GRAD_ACC_SUFFIX in n}
        non_grad_names = [n for n in final_input_names if n not in grad_acc_names]

        save_dict: dict = {}
        for npz_idx, name in enumerate(non_grad_names):
            if name in feed_mb0:
                save_dict[f"arr_{npz_idx:04d}"] = feed_mb0[name]
            else:
                print(f"   non-grad input '{name}' not found in feed — skipping")

        # For autoencoder: both data slots (input + labels) use the same feature vector
        session_type: dict = {inp.name: inp.type for inp in session.get_inputs()}
        data_names = non_grad_names[:num_data_inputs]
        for mb in range(1, effective_data_size):
            data = test_inputs[mb]
            for buf_idx, data_name in enumerate(data_names):
                # Both input and labels slots get the same feature vector
                save_dict[f"mb{mb}_arr_{buf_idx:04d}"] = data

        save_dict["meta_data_size"] = np.array([effective_data_size], dtype=np.int32)
        save_dict["meta_n_batches"] = np.array([n_batches], dtype=np.int32)
        save_dict["meta_n_accum"] = np.array([n_accum], dtype=np.int32)
        np.savez(save_dir / "inputs.npz", **save_dict)

        n_params = sum(1 for n in non_grad_names if n in init_map)
        n_grad = len(grad_acc_names)
        print(
            f"   inputs.npz: {len(non_grad_names)} base tensors "
            f"(data + {n_params} params; {n_grad} grad-acc-buf(s) omitted) "
            f"+ {(effective_data_size - 1) * num_data_inputs} DATA entries"
        )

        np.savez(save_dir / "outputs.npz", **outputs_dict)
        n_updated = sum(1 for k in outputs_dict if k in init_map)
        print(f"   outputs.npz: {len(outputs_dict)} tensors ({n_updated} updated params + loss)")
