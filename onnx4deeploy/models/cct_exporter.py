# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""CCT (Compact Convolutional Transformer) Model Exporter."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
import torch

from ..core.base_exporter import BaseONNXExporter
from ..optimization import remove_identity_nodes
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
            "img_size": 32,
            "embedding_dim": 128,
            "num_heads": 2,
            "num_layers": 2,
            "num_classes": 10,
            "opset_version": 17,  # LayerNormalization requires opset 17+
            "n_conv_layers": 1,
            "kernel_size": 3,
            "positional_embedding": "learnable",
            # Training configuration
            "training_strategy": "linear",  # Options: "linear", "last_attention", "last_2_attention", "lora_block1", "lora_block2", "full", "custom"
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

        Supports multiple training strategies for different fine-tuning scenarios:
        - "linear": Only train final classification layer (default)
        - "last_attention": Train last attention block + classifier
        - "last_2_attention": Train last 2 attention blocks + classifier
        - "lora_block1": LoRA-style training for block 1
        - "lora_block2": LoRA-style training for both blocks
        - "full": Train all parameters
        - "custom": Use custom_trainable_params from config

        Args:
            all_param_names: List of all parameter names in the model

        Returns:
            List of parameter names that should be trainable
        """
        strategy = self.config.get("training_strategy", "linear")

        # Define training strategies
        strategy_params = {
            "linear": [
                "classifier_fc_weight",
                "classifier_fc_bias",
            ],
            "last_attention": [
                "classifier_fc_weight",
                "classifier_fc_bias",
                "node_0_classifier_attention_pool_Transpose__0",
                "classifier_attention_pool_bias",
                "node_0_classifier_blocks_1_linear2_Transpose__0",
                "classifier_blocks_0_self_attn_proj_bias_Identity_31",
                "node_0_classifier_blocks_1_linear1_Transpose__0",
                "classifier_blocks_0_self_attn_proj_bias_Identity_33",
                "node_0_classifier_blocks_1_self_attn_proj_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_v_proj_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_k_proj_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_q_proj_Transpose__0",
            ],
            "last_2_attention": [
                "classifier_fc_weight",
                "classifier_fc_bias",
                "node_0_classifier_attention_pool_Transpose__0",
                "classifier_attention_pool_bias",
                "node_0_classifier_blocks_1_linear2_Transpose__0",
                "classifier_blocks_0_self_attn_proj_bias_Identity_31",
                "node_0_classifier_blocks_1_linear1_Transpose__0",
                "classifier_blocks_0_self_attn_proj_bias_Identity_33",
                "node_0_classifier_blocks_1_self_attn_proj_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_v_proj_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_k_proj_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_q_proj_Transpose__0",
                "node_0_classifier_blocks_0_linear2_Transpose__0",
                "classifier_blocks_0_self_attn_proj_bias_Identity_34",
                "node_0_classifier_blocks_0_linear1_Transpose__0",
                "node_0_classifier_blocks_0_self_attn_proj_Transpose__0",
                "classifier_blocks_0_self_attn_proj_bias__classifier_blocks_0_self_attn_proj_Add",
                "node_0_classifier_blocks_0_self_attn_k_proj_Transpose__0",
                "node_0_classifier_blocks_0_self_attn_q_proj_Transpose__0",
                "node_0_classifier_blocks_0_self_attn_v_proj_Transpose__0",
            ],
            "lora_block1": [
                "classifier_fc_weight",
                "classifier_fc_bias",
                "node_0_classifier_attention_pool_Transpose__0",
                "classifier_attention_pool_bias",
                "node_0_classifier_blocks_1_linear2_Transpose_1__0",
                "node_0_classifier_blocks_1_linear2_Transpose__0",
                "node_0_classifier_blocks_1_linear1_Transpose_1__0",
                "node_0_classifier_blocks_1_linear1_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_11__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_10__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_4__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_3__0",
                "node_0_classifier_blocks_1_self_attn_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_1__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_5__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_6__0",
            ],
            "lora_block2": [
                "classifier_fc_weight",
                "classifier_fc_bias",
                "node_0_classifier_attention_pool_Transpose__0",
                "classifier_attention_pool_bias",
                # Block 1
                "node_0_classifier_blocks_1_linear2_Transpose_1__0",
                "node_0_classifier_blocks_1_linear2_Transpose__0",
                "node_0_classifier_blocks_1_linear1_Transpose_1__0",
                "node_0_classifier_blocks_1_linear1_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_11__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_10__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_4__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_3__0",
                "node_0_classifier_blocks_1_self_attn_Transpose__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_1__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_5__0",
                "node_0_classifier_blocks_1_self_attn_Transpose_6__0",
                # Block 0
                "node_0_classifier_blocks_0_linear2_Transpose_1__0",
                "node_0_classifier_blocks_0_linear2_Transpose__0",
                "node_0_classifier_blocks_0_linear1_Transpose_1__0",
                "node_0_classifier_blocks_0_linear1_Transpose__0",
                "node_0_classifier_blocks_0_self_attn_Transpose_11__0",
                "node_0_classifier_blocks_0_self_attn_Transpose_10__0",
                "node_0_classifier_blocks_0_self_attn_Transpose_4__0",
                "node_0_classifier_blocks_0_self_attn_Transpose_3__0",
                "node_0_classifier_blocks_0_self_attn_Transpose__0",
                "node_0_classifier_blocks_0_self_attn_Transpose_1__0",
                "node_0_classifier_blocks_0_self_attn_Transpose_5__0",
                "node_0_classifier_blocks_0_self_attn_Transpose_6__0",
            ],
            "full": all_param_names,  # Train everything
            "custom": self.config.get("custom_trainable_params", []),
        }

        # Get trainable params based on strategy
        if strategy not in strategy_params:
            print(f"⚠️  Unknown training strategy '{strategy}', using 'linear' as fallback")
            strategy = "linear"

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

    def run_training_optimization(self, onnx_file: str, output_file: str):
        """
        Run ONNX optimizations for CCT training mode.

        Training-specific optimizations:
        - Fold frozen parameters into ONNX initializers (constants)
        - Shape inference for training graph
        - Identity node removal

        ORT's generate_artifacts() exposes ALL parameters (frozen + trainable) as
        graph inputs. Deeploy's NCHW→NHWC layout-conversion pass requires Conv weights
        to be gs.Constant (ONNX initializer), not gs.Variable (graph input). This pass
        folds frozen params (those without a _grad.accumulation.buffer counterpart) back
        into ONNX initializers using their initial values from network_infer.onnx.

        Args:
            onnx_file: Path to input ONNX file
            output_file: Path to save optimized ONNX file
        """

        print("🔧 Running CCT-specific training optimizations...")

        # Load training model and inference model (for initial frozen param values).
        model = onnx.load(onnx_file)
        infer_model = onnx.load(self.paths["network_infer"])
        init_map = {init.name: init for init in infer_model.graph.initializer}

        # Detect trainable param names: those with a _grad.accumulation.buffer input.
        _GRAD_ACC = "_grad.accumulation.buffer"
        input_names = {inp.name for inp in model.graph.input}
        trainable_names = set()
        for inp_name in input_names:
            if _GRAD_ACC in inp_name:
                param_name = inp_name[: -len(_GRAD_ACC)]
                trainable_names.add(param_name)

        # Fold frozen params back into ONNX initializers.
        # Frozen = float32 graph inputs that are not:
        #   - data inputs (no initial value in infer model)
        #   - trainable params (have a grad_acc_buf counterpart)
        #   - grad acc buffers (_grad.accumulation.buffer suffix)
        #   - lazy_reset_grad (bool)
        #   - labels (int64)
        graph_inputs_to_keep = []
        frozen_folded = 0
        for inp in model.graph.input:
            name = inp.name
            elem_type = inp.type.tensor_type.elem_type if inp.type.HasField("tensor_type") else 0
            is_grad_acc = _GRAD_ACC in name
            is_trainable = name in trainable_names
            is_bool = elem_type == 9  # TensorProto.BOOL
            is_int = elem_type == 7  # TensorProto.INT64

            if is_grad_acc or is_trainable or is_bool or is_int:
                graph_inputs_to_keep.append(inp)
            elif name in init_map:
                # Fold as constant initializer (frozen param with known initial value).
                model.graph.initializer.append(init_map[name])
                frozen_folded += 1
            else:
                # Data input (image) — no initial value, must stay as graph input.
                graph_inputs_to_keep.append(inp)

        del model.graph.input[:]
        model.graph.input.extend(graph_inputs_to_keep)
        print(
            f"  ➤ Folded {frozen_folded} frozen params into graph constants "
            f"(kept {len(graph_inputs_to_keep)} explicit inputs)"
        )

        # 1. Unfuse BiasGelu → Add + Gelu
        # ORT's generate_artifacts() fuses Add(x, bias) + Gelu into BiasGelu.
        # Deeploy has no BiasGelu mapping but supports separate Add and Gelu ops.
        # Transform: BiasGelu(x, bias) → intermediate = Add(x, bias); Gelu(intermediate)
        import onnx_graphsurgeon as gs

        gs_model = gs.import_onnx(model)
        bias_gelu_nodes = [n for n in gs_model.nodes if n.op == "BiasGelu"]
        if bias_gelu_nodes:
            print(f"  ➤ Unfusing {len(bias_gelu_nodes)} BiasGelu node(s) → Add + Gelu...")
            for node in bias_gelu_nodes:
                x_in, bias_in = node.inputs[0], node.inputs[1]
                gelu_out = node.outputs[0]
                add_out = gs.Variable(
                    name=node.name + "_add_out",
                    dtype=gelu_out.dtype,
                    shape=gelu_out.shape,
                )
                add_node = gs.Node(
                    op="Add",
                    name=node.name + "_add",
                    inputs=[x_in, bias_in],
                    outputs=[add_out],
                )
                gelu_node = gs.Node(
                    op="Gelu",
                    name=node.name + "_gelu",
                    inputs=[add_out],
                    outputs=[gelu_out],
                )
                gs_model.nodes.append(add_node)
                gs_model.nodes.append(gelu_node)
                node.inputs.clear()
                node.outputs.clear()
            gs_model.cleanup().toposort()
            model = gs.export_onnx(gs_model)
            print(f"  ✔ BiasGelu unfused successfully")

        # 2. Shape inference
        print("  ➤ Shape inference for training graph...")
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as e:
            print(f"    Warning: Shape inference failed (expected for training): {e}")

        # Save intermediate result
        onnx.save(model, output_file)

        # 2. Remove identity nodes
        print("  ➤ Removing identity nodes...")
        remove_identity_nodes(output_file, output_file)

        print("  ✅ Training optimization complete")

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

        save_dir = Path(self.paths["output_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(42)
        input_shape = self.get_input_shape()
        batch_size = input_shape[0]
        num_classes = self.config.get("num_classes", 10)
        learning_rate = float(self.config.get("learning_rate", 0.001))

        print(
            f"   Training sim: n_batches={n_batches}  n_accum={n_accum}  n_steps={n_steps}  lr={learning_rate}"
        )

        # Generate n_batches distinct (input, labels) pairs.
        test_inputs = [np.random.randn(*input_shape).astype(np.float32) for _ in range(n_batches)]
        labels_list = [
            np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int64)
            for _ in range(n_batches)
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
