# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Simple MLP Model Exporter."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter

# Import SimpleFlatMLP PyTorch model (pre-flattened input, no Flatten op in ONNX)
from .pytorch_models.simple_mlp import SimpleFlatMLP


class SimpleMlpExporter(BaseONNXExporter):
    """ONNX exporter for Simple MLP model."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        """
        Initialize Simple MLP exporter.

        Args:
            save_path: Optional custom path to save ONNX files
            config_file: Path to configuration YAML file
        """
        super().__init__(save_path, config_file)
        self.model_config = {}

    def load_config(self) -> Dict[str, Any]:
        """
        Load Simple MLP configuration.

        Returns:
            Dictionary containing Simple MLP configuration parameters
        """
        # Default Simple MLP configuration
        config = {
            "batch_size": 1,
            "input_height": 8,
            "input_width": 8,
            "hidden_size": 8,
            "num_classes": 10,
            "opset_version": 17,
            "dropout": 0.0,  # No dropout for inference
            # Training configuration
            "training_strategy": "no_bias",  # Options: "full", "no_bias", "last_layer", "custom"
            "custom_trainable_params": [],
            # Training loop configuration
            "n_accum": 1,  # mini-batches per SGD update (gradient accumulation)
            # Data source configuration
            "dataset": "random",  # "random" | "mnist"
            "data_path": None,  # path for dataset files (None → auto-download)
            "data_split": "train",  # "train" | "test" (for MNIST)
        }

        # Apply any CLI overrides stored before export_training() was called.
        # This lets Onnx4Deeploy.py pass --n-batches (and future flags) without
        # modifying base_exporter.py's export_training() signature.
        if hasattr(self, "_config_overrides") and self._config_overrides:
            config.update(self._config_overrides)

        # When using MNIST, auto-adjust input dimensions and num_classes to match
        # the dataset (28×28 grayscale, 10-class) unless the user already overrode them.
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

    def create_model(self) -> torch.nn.Module:
        """
        Create Simple MLP PyTorch model.

        Returns:
            SimpleFlatMLP model ready for export (accepts pre-flattened 2-D input
            so that no Flatten node is emitted in the ONNX graph)
        """
        input_size = self.model_config["input_height"] * self.model_config["input_width"]

        return SimpleFlatMLP(
            input_size=input_size,
            hidden_size=self.model_config["hidden_size"],
            num_classes=self.model_config["num_classes"],
            dropout=self.model_config["dropout"],
        )

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the input tensor shape for Simple MLP.

        Returns:
            Tuple representing input shape (batch_size, input_size) – already
            flat so that the exporter does not need a Flatten op.
        """
        batch_size = self.config["batch_size"]
        input_size = self.config["input_height"] * self.config["input_width"]
        return (batch_size, input_size)

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """
        Get list of trainable parameter names for Simple MLP.

        Supports multiple training strategies:
        - "full":       Train all parameters (weights + biases)
        - "no_bias":    Train all weights, freeze all biases (default)
        - "last_layer": Only train the final classification layer (fc3.weight + fc3.bias)
        - "custom":     Use custom_trainable_params from config

        Args:
            all_param_names: List of all parameter names in the model

        Returns:
            List of parameter names that should be trainable
        """
        strategy = self.config.get("training_strategy", "full")

        # Define training strategies
        strategy_params = {
            "full": all_param_names,  # Train everything
            "no_bias": [name for name in all_param_names if "bias" not in name],  # Weights only
            "last_layer": [
                "fc2.weight",
                "fc2.bias",
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

    def get_data_source(self):
        """
        Return the data source for training mini-batch generation.

        Selects based on ``config["dataset"]``:
        - ``"random"`` (default): random Gaussian inputs, uniform-random labels.
        - ``"mnist"``: real MNIST images (downloaded automatically if needed).
        """
        dataset = (self.config or self.model_config or {}).get("dataset", "random")
        if dataset == "mnist":
            from ..data.mnist_datasource import MNISTDataSource

            cfg = self.config or self.model_config or {}
            return MNISTDataSource(
                data_path=cfg.get("data_path", None),
                split=cfg.get("data_split", "train"),
            )
        from ..data.random_datasource import RandomDataSource

        return RandomDataSource()

    def _get_config_string(self) -> str:
        """
        Get configuration string for folder naming.

        Returns:
            Configuration string like "_28x28_128_10"
        """
        return f"_{self.config['input_height']}x{self.config['input_width']}_{self.config['hidden_size']}_{self.config['num_classes']}"

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
        Within each accumulation group, all losses are computed with the same
        (pre-update) weights, matching the C harness behaviour.

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

        save_dir = Path(self.paths["output_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        input_shape = self.get_input_shape()
        input_shape[0]
        num_classes = self.config.get("num_classes", 2)
        learning_rate = float(self.config.get("learning_rate", 0.001))

        print(
            f"   Training sim: n_batches={n_batches}  n_accum={n_accum}  n_steps={n_steps}  lr={learning_rate}"
        )

        # Load n_batches distinct (input, labels) pairs via the configured DataSource.
        data_source = self.get_data_source()
        test_inputs, labels_list = data_source.load_batches(
            n_batches, input_shape, num_classes, seed=42
        )

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
                        feed[name] = labels_list[mb]
                    elif inp.type == "tensor(bool)":
                        # lazy_reset_grad: True on first accum step, False otherwise.
                        feed[name] = np.array([accum_step == 0])
                    elif name in current_weights:
                        feed[name] = current_weights[name]
                    elif _GRAD_ACC in name:
                        feed[name] = np.zeros(shape, dtype=np.float32)
                    elif shape == list(input_shape):
                        feed[name] = test_inputs[mb]
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

        # Per-mini-batch DATA entries for mb 1 … n_batches-1.
        # Build a dtype look-up from the session inputs.
        session_type: dict = {inp.name: inp.type for inp in session.get_inputs()}
        data_names = non_grad_names[:num_data_inputs]
        for mb in range(1, n_batches):
            for buf_idx, data_name in enumerate(data_names):
                inp_type = session_type.get(data_name, "tensor(float)")
                if inp_type == "tensor(int64)":
                    save_dict[f"mb{mb}_arr_{buf_idx:04d}"] = labels_list[mb]
                else:
                    save_dict[f"mb{mb}_arr_{buf_idx:04d}"] = test_inputs[mb]

        np.savez(save_dir / "inputs.npz", **save_dict)
        n_params = sum(1 for n in non_grad_names if n in init_map)
        n_grad = len(grad_acc_names)
        print(
            f"   ✅ inputs.npz  — {len(non_grad_names)} base tensors "
            f"(data + {n_params} params + ctrl; {n_grad} grad-acc-buf(s) omitted) "
            f"+ {(n_batches - 1) * num_data_inputs} per-mb DATA entries "
            f"({n_batches} mini-batches total)"
        )

        np.savez(save_dir / "outputs.npz", **outputs_dict)
        n_updated = sum(1 for k in outputs_dict if k in init_map)
        print(
            f"   ✅ outputs.npz — {len(outputs_dict)} tensors ({n_updated} updated params + loss)"
        )
