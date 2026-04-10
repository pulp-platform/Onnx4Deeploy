# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MobileNetV2 Model Exporter — inference + training graph support."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter
from .pytorch_models.mobilenet import mobilenet_v2


class MobileNetV2Exporter(BaseONNXExporter):
    """ONNX exporter for MobileNetV2 model (MLPerf Mobile / VWW benchmark)."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        super().__init__(save_path, config_file)
        self.model_config = {}

    # ------------------------------------------------------------------ #
    # Configuration                                                        #
    # ------------------------------------------------------------------ #

    def load_config(self) -> Dict[str, Any]:
        config = {
            "batch_size": 1,
            "img_size": 224,
            "input_channels": 3,
            "num_classes": 1000,
            "width_mult": 1.0,  # 0.35 for MLperf Tiny VWW
            "opset_version": 17,
            # Training
            "training_strategy": "full",  # "full" | "last_layer" | "no_features" | "custom"
            "custom_trainable_params": [],
            "learning_rate": 0.001,
            "n_batches": 4,
            "n_accum": 1,
            "data_size": None,
        }

        if hasattr(self, "_config_overrides") and self._config_overrides:
            config.update(self._config_overrides)

        self.model_config = config
        return config

    # ------------------------------------------------------------------ #
    # Model factory                                                        #
    # ------------------------------------------------------------------ #

    def create_model(self) -> torch.nn.Module:
        return mobilenet_v2(
            num_classes=self.model_config["num_classes"],
            width_mult=self.model_config["width_mult"],
            input_channels=self.model_config["input_channels"],
        )

    # ------------------------------------------------------------------ #
    # Shape helpers                                                        #
    # ------------------------------------------------------------------ #

    def get_input_shape(self) -> Tuple[int, ...]:
        return (
            self.config["batch_size"],
            self.config["input_channels"],
            self.config["img_size"],
            self.config["img_size"],
        )

    def _get_config_string(self) -> str:
        width = self.config["width_mult"]
        return f"_mobilenetv2_{width}_{self.config['img_size']}_{self.config['num_classes']}"

    # ------------------------------------------------------------------ #
    # Training strategy                                                   #
    # ------------------------------------------------------------------ #

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """
        Pattern-based trainable parameter selection.

        Strategies:
        - "full":         Train all parameters (default).
        - "last_layer":   Only the final classifier FC.
        - "no_features":  Freeze features backbone; train classifier only.
        - "custom":       Explicit list from config["custom_trainable_params"].
        """
        strategy = self.config.get("training_strategy", "full")

        _FREEZE = {
            "full": lambda n: False,
            "last_layer": lambda n: "classifier" not in n,
            "no_features": lambda n: "features" in n,
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
        Save inputs.npz / outputs.npz for training-mode validation.

        Follows the SimpleCnnExporter layout (see simple_cnn_exporter.py for details).
        Grad-accumulation buffers are excluded; the C harness zero-inits them.
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
        num_classes = self.config.get("num_classes", 10)
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

        rng = np.random.default_rng(42)
        test_inputs = [
            rng.standard_normal(input_shape).astype(np.float32) for _ in range(effective_data_size)
        ]
        labels_list = [
            rng.integers(0, num_classes, size=(input_shape[0],)).astype(np.int64)
            for _ in range(effective_data_size)
        ]

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

                feed = self._build_input_feed(
                    session,
                    param_values=current_weights,
                    test_input=test_inputs[mb % effective_data_size],
                    labels=labels_list[mb % effective_data_size],
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
            f"   inputs.npz: {len(non_grad_names)} base tensors "
            f"(data + {n_params} params; {n_grad} grad-acc-buf(s) omitted) "
            f"+ {(effective_data_size - 1) * num_data_inputs} DATA entries"
        )

        np.savez(save_dir / "outputs.npz", **outputs_dict)
        n_updated = sum(1 for k in outputs_dict if k in init_map)
        print(f"   outputs.npz: {len(outputs_dict)} tensors ({n_updated} updated params + loss)")
