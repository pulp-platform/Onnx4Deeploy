# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Simple CNN Model Exporter with full training-graph support."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter
from .pytorch_models.simple_cnn import SimpleCNN


class SimpleCnnExporter(BaseONNXExporter):
    """ONNX exporter for Simple CNN model (inference + training graph)."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        super().__init__(save_path, config_file)
        self.model_config = {}

    # ---------------------------------------------------------------------- #
    # Configuration                                                           #
    # ---------------------------------------------------------------------- #

    def load_config(self) -> Dict[str, Any]:
        config = {
            "batch_size": 1,
            "input_channels": 1,
            "input_height": 16,
            "input_width": 16,
            "hidden_channels": 16,  # Conv2 output channels; Conv1 = hidden_channels // 2
            "num_classes": 10,
            "opset_version": 17,
            # Training
            "learning_rate": 0.001,
            "training_strategy": "full",  # "full" | "conv_only" | "fc_only" | "no_bias" | "custom"
            "custom_trainable_params": [],
            "n_accum": 1,
            # Data
            "dataset": "random",  # "random" | "mnist"
            "data_path": None,
            "data_split": "train",
            "data_size": None,
        }

        if hasattr(self, "_config_overrides") and self._config_overrides:
            config.update(self._config_overrides)

        # When MNIST is selected, auto-adjust spatial dims to 28×28.
        if config.get("dataset") == "mnist":
            overrides = getattr(self, "_config_overrides", {}) or {}
            if "input_height" not in overrides:
                config["input_height"] = 28
            if "input_width" not in overrides:
                config["input_width"] = 28
            if "num_classes" not in overrides:
                config["num_classes"] = 10

        self.model_config = config
        return config

    # ---------------------------------------------------------------------- #
    # Model factory                                                           #
    # ---------------------------------------------------------------------- #

    def create_model(self) -> torch.nn.Module:
        return SimpleCNN(
            input_channels=self.model_config["input_channels"],
            input_height=self.model_config["input_height"],
            input_width=self.model_config["input_width"],
            hidden_channels=self.model_config["hidden_channels"],
            num_classes=self.model_config["num_classes"],
        )

    # ---------------------------------------------------------------------- #
    # Shape helpers                                                           #
    # ---------------------------------------------------------------------- #

    def get_input_shape(self) -> Tuple[int, ...]:
        return (
            self.config["batch_size"],
            self.config["input_channels"],
            self.config["input_height"],
            self.config["input_width"],
        )

    def _get_config_string(self) -> str:
        return (
            f"_{self.config['input_height']}x{self.config['input_width']}"
            f"_{self.config['hidden_channels']}ch_{self.config['num_classes']}"
        )

    # ---------------------------------------------------------------------- #
    # Training strategy                                                      #
    # ---------------------------------------------------------------------- #

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """
        Return the subset of parameters to train.

        Strategies:
        - "full":      Train all parameters (weights + biases).
        - "no_bias":   Train weights only, freeze all biases.
        - "conv_only": Freeze FC layer; train only conv layers.
        - "fc_only":   Freeze conv layers; train only FC layer.
        - "custom":    Use custom_trainable_params list from config.
        """
        strategy = self.config.get("training_strategy", "full")

        strategy_params = {
            "full": all_param_names,
            "no_bias": [n for n in all_param_names if "bias" not in n],
            "conv_only": [n for n in all_param_names if n.startswith("conv")],
            "fc_only": [n for n in all_param_names if n.startswith("fc")],
            "custom": self.config.get("custom_trainable_params", []),
        }

        if strategy not in strategy_params:
            print(f"   Unknown training strategy '{strategy}', using 'full' as fallback")
            strategy = "full"

        requires_grad = [n for n in all_param_names if n in strategy_params[strategy]]

        print(f"\n   Training Strategy: '{strategy}'")
        print(f"   Total params: {len(all_param_names)}  trainable: {len(requires_grad)}")
        return requires_grad

    # ---------------------------------------------------------------------- #
    # Data source                                                             #
    # ---------------------------------------------------------------------- #

    def get_data_source(self):
        dataset = (self.config or self.model_config or {}).get("dataset", "random")
        if dataset == "mnist":
            from ..data.mnist_datasource import MNISTDataSource

            cfg = self.config or self.model_config or {}
            return MNISTDataSource(
                data_path=cfg.get("data_path", None),
                split=cfg.get("data_split", "train"),
                data_size=cfg.get("data_size", None),
            )
        from ..data.random_datasource import RandomDataSource

        return RandomDataSource()

    # ---------------------------------------------------------------------- #
    # Inference test data                                                     #
    # ---------------------------------------------------------------------- #

    def save_test_data(self, model: torch.nn.Module, save_dir: str):
        print("   Saving inference test data...")
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
        print(f"   Input:  {test_input.shape}  Output: {test_output.shape}")

    # ---------------------------------------------------------------------- #
    # Training test data                                                      #
    # ---------------------------------------------------------------------- #

    def create_training_test_data(
        self, n_batches: int = None, num_data_inputs: int = 2, n_accum: int = None
    ) -> None:
        """
        Save inputs.npz / outputs.npz for training-mode validation.

        Follows the same layout as SimpleMlpExporter.create_training_test_data():
        - inputs.npz: arr_0000…arr_{M-1} (base), mb{I}_arr_{J:04d} (per-batch data),
                      meta_data_size, meta_n_batches.
        - outputs.npz: updated weight tensors + loss array.

        Grad-accumulation buffers are intentionally excluded; the C harness
        zero-initialises them after InitTrainingNetwork().
        """
        import onnx
        import onnxruntime as ort
        from onnx import numpy_helper

        _GRAD_ACC = "_grad.accumulation.buffer"

        # Resolve n_batches / n_accum.
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
            f"   Training sim: n_batches={n_batches}  n_accum={n_accum}"
            f"  n_steps={n_steps}  lr={learning_rate}"
        )

        # Effective data size (unique samples stored; C harness cycles via modulo).
        _data_size_cfg = self.config.get("data_size", None)
        effective_data_size = (
            int(_data_size_cfg)
            if (_data_size_cfg and int(_data_size_cfg) < n_batches)
            else n_batches
        )

        data_source = self.get_data_source()
        test_inputs, labels_list = data_source.load_batches(
            effective_data_size, input_shape, num_classes, seed=42
        )

        # Read initial weights from inference ONNX.
        infer_model = onnx.load(self.paths["network_infer"])
        init_map: dict = {
            init.name: numpy_helper.to_array(init) for init in infer_model.graph.initializer
        }

        # Load training model and expose per-step gradients as extra outputs.
        train_model_onnx = onnx.load(self.paths["network_train"])
        grad_tensor_map: dict = {}
        for node in train_model_onnx.graph.node:
            if "InPlaceAccumulator" in node.op_type and len(node.input) >= 2:
                grad_tensor_name = node.input[1]
                if grad_tensor_name.endswith("_grad"):
                    param_name = grad_tensor_name[:-5]
                    grad_tensor_map[param_name] = grad_tensor_name

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

                feed = {}
                for inp in session.get_inputs():
                    name = inp.name
                    shape = [d for d in inp.shape if isinstance(d, int) and d > 0]
                    if inp.type == "tensor(int64)":
                        feed[name] = labels_list[mb % effective_data_size]
                    elif inp.type == "tensor(bool)":
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
                    feed_mb0 = {k: v.copy() if hasattr(v, "copy") else v for k, v in feed.items()}

                raw_outputs = session.run(None, feed)
                outputs_raw = dict(zip(session_output_names, raw_outputs))

                for out_name, out_val in outputs_raw.items():
                    if "loss" in out_name.lower() and "grad" not in out_name.lower():
                        all_losses.append(float(np.array(out_val).flatten()[0]))
                        break

                for pname, grad_name in grad_tensor_map.items():
                    if grad_name in outputs_raw and pname in accumulated_grads:
                        accumulated_grads[pname] = accumulated_grads[pname] + outputs_raw[grad_name]

            for pname, acc_grad in accumulated_grads.items():
                current_weights[pname] = current_weights[pname] - learning_rate * acc_grad

        outputs_dict: dict = {k: v for k, v in current_weights.items()}
        outputs_dict["loss"] = np.array(all_losses, dtype=np.float32)
        print(f"   Reference losses: {all_losses}")

        # ------------------------------------------------------------------ #
        # Build inputs.npz                                                   #
        # ------------------------------------------------------------------ #
        final_model = onnx.load(self.paths["network"])
        final_input_names = [inp.name for inp in final_model.graph.input]

        grad_acc_names = {n for n in final_input_names if _GRAD_ACC in n}
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
