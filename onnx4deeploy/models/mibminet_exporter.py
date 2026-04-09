# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MI-BMInet (Motor Imagery BMI Network) Model Exporter."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter

# MIBMINetDeploy: LayerNorm-based, no dropout — ONNX training compatible.
from .pytorch_models.mibminet import MIBMINetDeploy


class MIBMInetExporter(BaseONNXExporter):
    """ONNX exporter for MI-BMInet (Motor Imagery BMI Network) model."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        """
        Initialize MI-BMInet exporter.

        Args:
            save_path: Optional custom path to save ONNX files
            config_file: Path to configuration YAML file
        """
        super().__init__(save_path, config_file)
        self.model_config = {}

    def load_config(self) -> Dict[str, Any]:
        """
        Load MI-BMInet configuration.

        Returns:
            Dictionary containing MI-BMInet configuration parameters
        """
        config = {
            "batch_size": 1,
            "channels": 8,  # C: number of EEG channels
            "time_steps": 2000,  # T: number of time samples
            "num_classes": 2,
            "F1": 8,  # Number of spectral filters
            "D": 2,  # Spatial filter multiplier (F2 = F1 * D)
            "Nf": 64,  # Temporal filter size (block 1)
            "Nf2": 16,  # Temporal filter size (block 2, separable)
            "activation": "relu",
            "opset_version": 17,
            # Training configuration
            "training_strategy": "full",  # Options: "full", "norm_only", "last_layer", "custom"
            "custom_trainable_params": [],
            "learning_rate": 0.001,
            "n_accum": 1,  # mini-batches per SGD update (gradient accumulation)
            # Data source configuration
            "dataset": "random",  # "random" | "eeg" (use "random" for synthetic data)
            "data_path": None,
            "data_split": "train",
            "data_size": None,
        }

        # Apply CLI overrides (e.g. --n-batches, --n-accum, --batch-size).
        if hasattr(self, "_config_overrides") and self._config_overrides:
            config.update(self._config_overrides)

        self.model_config = config
        return config

    def create_model(self) -> torch.nn.Module:
        """
        Create MIBMINetDeploy PyTorch model.

        MIBMINetDeploy uses LayerNorm (instead of BatchNorm) and no dropout,
        making it fully compatible with ONNX training graph generation.

        Returns:
            MIBMINetDeploy model ready for export
        """
        model = MIBMINetDeploy(
            F1=self.model_config["F1"],
            D=self.model_config["D"],
            C=self.model_config["channels"],
            T=self.model_config["time_steps"],
            N=self.model_config["num_classes"],
            Nf=self.model_config["Nf"],
            Nf2=self.model_config["Nf2"],
            activation=self.model_config.get("activation", "relu"),
        )
        return model

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the input tensor shape for MI-BMInet.

        Returns:
            Tuple representing input shape (batch_size, 1, channels, time_steps)
        """
        batch_size = self.config["batch_size"]
        channels = self.config["channels"]
        time_steps = self.config["time_steps"]
        return (batch_size, 1, channels, time_steps)

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """
        Get list of trainable parameter names for MI-BMInet.

        Uses pattern-based lambda filtering (never hardcoded ONNX names, which
        change across pipeline versions).

        Strategies:
        - "full":       Train everything — no frozen params (default, PULP-safe)
        - "norm_only":  Freeze all conv/sep_conv weights; train LayerNorm affines + classifier
        - "last_layer": Freeze everything except the final classifier (fc*)
        - "custom":     Explicit list from config["custom_trainable_params"]

        Args:
            all_param_names: List of all ONNX initializer names in the model

        Returns:
            List of parameter names that should be trainable
        """
        strategy = self.config.get("training_strategy", "full")

        # Lambda returns True → freeze (exclude from training graph)
        _FREEZE = {
            "full": lambda n: False,  # train all
            "norm_only": lambda n: "conv" in n,  # freeze conv* + sep_conv*
            "last_layer": lambda n: not n.startswith("fc_"),  # only fc*
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
            print(f"   Frozen (→ constant): {frozen}")

        return requires_grad

    def get_data_source(self):
        """
        Return the data source for training mini-batch generation.

        MI-BMInet uses random data by default (EEG signals are subject-specific).
        """
        from ..data.random_datasource import RandomDataSource

        return RandomDataSource()

    def _get_config_string(self) -> str:
        """
        Get configuration string for folder naming.

        Returns:
            Configuration string like "_8_2000_2"
        """
        return (
            f"_{self.config['channels']}_{self.config['time_steps']}_{self.config['num_classes']}"
        )

    def save_test_data(self, model: torch.nn.Module, save_dir: str):
        """
        Save test input/output data for validation.

        Args:
            model: PyTorch model to run inference with
            save_dir: Directory to save test data
        """
        print("💾 Saving test input/output data...")

        input_shape = self.get_input_shape()
        test_input = np.random.randn(*input_shape).astype(np.float32)

        was_training = model.training
        model.eval()

        with torch.no_grad():
            input_tensor = torch.from_numpy(test_input)
            output_tensor = model(input_tensor)
            test_output = output_tensor.numpy()

        if was_training:
            model.train()

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

        Generates n_batches distinct (input, labels) pairs and simulates
        the training loop (with optional gradient accumulation) to produce
        reference weight updates and losses.

        inputs.npz layout
        -----------------
        Base entries (non-grad-buf graph inputs):
          arr_0000 … arr_{M-1}  — data + weights + ctrl inputs for mb=0
        Per-mini-batch DATA entries (mb 1 … effective_data_size-1):
          mb{I}_arr_{J:04d}     — DATA input J for mini-batch I

        outputs.npz layout
        ------------------
          <param_name>  — SGD-updated weight tensors
          loss          — reference loss for each mini-batch, shape (n_batches,)
        """
        import onnx
        import onnxruntime as ort

        if n_batches is None:
            n_batches = self.config.get("n_batches", 4)
        if n_accum is None:
            n_accum = int(self.config.get("n_accum", 1))
        if n_batches % n_accum != 0:
            n_batches = (n_batches // n_accum) * n_accum
            if n_batches == 0:
                n_batches = n_accum
            print(
                f"   ⚠️  n_batches adjusted to {n_batches} (must be divisible by n_accum={n_accum})"
            )
        n_steps = n_batches // n_accum

        save_dir = Path(self.paths["output_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        input_shape = self.get_input_shape()
        num_classes = self.config.get("num_classes", 2)
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

        data_source = self.get_data_source()
        test_inputs, labels_list = data_source.load_batches(
            effective_data_size, input_shape, num_classes, seed=42
        )

        init_map: dict = self._load_init_map(self.paths["network_infer"])

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

                feed = self._build_input_feed(
                    session,
                    param_values=current_weights,
                    test_input=test_inputs[mb % effective_data_size],
                    labels=labels_list[mb % effective_data_size],
                    lazy_reset_grad=(accum_step == 0),
                )

                if mb == 0:
                    feed_mb0 = dict(feed)

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
        print(f"   Collected {len(all_losses)} reference losses: {all_losses}")

        final_model = onnx.load(self.paths["network"])
        final_input_names = [inp.name for inp in final_model.graph.input]

        grad_acc_names = {n for n in final_input_names if self._GRAD_ACC_SUFFIX in n}
        non_grad_names = [n for n in final_input_names if n not in grad_acc_names]

        save_dict: dict = {}
        for npz_idx, name in enumerate(non_grad_names):
            if name in feed_mb0:
                save_dict[f"arr_{npz_idx:04d}"] = feed_mb0[name]
            else:
                print(f"   ⚠️  network.onnx non-grad input '{name}' not found in feed — skipping")

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
            f"({effective_data_size} unique samples, {n_batches} total mini-batches)"
        )

        np.savez(save_dir / "outputs.npz", **outputs_dict)
        n_updated = sum(1 for k in outputs_dict if k in init_map)
        print(
            f"   ✅ outputs.npz — {len(outputs_dict)} tensors ({n_updated} updated params + loss)"
        )
