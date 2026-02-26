import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
import json
import torch
from pathlib import Path
from onnx import TensorProto, helper, shape_inference

from onnx4deeploy.transform.model_transform import ensure_all_tensor_shapes

def generate_zo_graph(inference_onnx:str, output_onnx:str, zo_config:dict) -> None:
    """ Generate MeZO ONNX graph for model based on its inference onnx"""

    epsilon, seed, noise_type, exceptions = zo_config["epsilon"], zo_config["seed"], zo_config["noise_type"], zo_config.get("exceptions", [])

    base_path = os.path.dirname(output_onnx)
    os.makedirs(base_path, exist_ok=True)
    inject_perturbation_nodes(inference_onnx,
                              output_path=output_onnx,
                              epsilon=epsilon,
                              seed=seed,
                              noise_type=noise_type,
                              exceptions=exceptions)

    ensure_all_tensor_shapes(model_path=output_onnx, output_path=output_onnx)
    # append_cross_entropy_loss(output_onnx, output_onnx, label_name='label')

def inject_perturbation_nodes(
    onnx_path: str,
    output_path: str,
    epsilon: float = 0.01,
    seed: float = 42.0,
    noise_type: str = "gaussian",
    exceptions: list[str] = []
) -> None:
    """
    This function inserts statically-seeded random operators. The unique seed for each
    operator serves as an identifier that a custom hardware runtime can override with
    a dynamic, runtime-provided seed.

    Args:
        onnx_path: Path to the original ONNX model.
        epsilon: The magnitude of the perturbation.
                For the negative forward pass, just reverse the sign.
                For inference, set to 0.
        seed: A base seed to generate unique, deterministic seeds for each operator.
        noise_type: The type of random distribution to use ('gaussian' or 'uniform').
    """
    # Load original ONNX model
    p = Path(onnx_path)

    # --- 1. Identify target weights and biases ---
    model = onnx.load(onnx_path)
    weights_and_biases = {
        init.name
        for init in model.graph.initializer
        if "weight" in init.name or "bias" in init.name
    }

    if not weights_and_biases:
        print("Warning: No weights or biases containing 'weight' or 'bias' in their names were found to perturb.")
        return

    print(f"Found {len(weights_and_biases)} weight/bias tensors to perturb.")

    def modify_graph(original_model: onnx.ModelProto, output_path: str, exceptions: list[str]):
        new_nodes = []
        extra_value_infos = []

        # Keep track of all initializers. We will add to this list.
        new_initializers = list(original_model.graph.initializer)

        # Create a set of initializer names for quick lookups
        initializer_names = {init.name for init in new_initializers}

        base_seed = int(seed)
        perturbation_counter = 0

        # Prepare a fast lookup for initializer names
        initializer_names = {init.name for init in new_initializers}

        for node in original_model.graph.node:
            # Check if this is a node we want to modify
            if node.op_type in ["Conv", "Gemm", "MatMul"] and node.name not in exceptions:
                print(F"node: {node.name}, op_type: {node.op_type}")
                modified_inputs = list(node.input)
                made_change = False

                for i, input_name in enumerate(node.input):
                    # Check if the input is a weight/bias initializer
                    if input_name in initializer_names:
                        made_change = True

                        print(f"input_name {i}: {input_name}")
                        # Find the original weight tensor to get its properties
                        original_weight_tensor = next(t for t in new_initializers if t.name == input_name)
                        dtype = TensorProto.DataType.Name(original_weight_tensor.data_type)  # "FLOAT"
                        noise_shape = original_weight_tensor.dims
                        noise_shape = [int(x) for x in noise_shape]

                        # --- This is the core logic for injecting nodes ---

                          # 1. Define names for the new intermediate tensors
                        perturbed_tensor_name = f"{perturbation_counter}_{input_name}"
                        # 2. Create the RandomNormal/RandomUniform node
                        unique_seed = float(base_seed + perturbation_counter)

                        if noise_type == "gaussian":
                            perturbation_node = helper.make_node(
                                "PerturbNormal",
                                inputs=[input_name],
                                outputs=[perturbed_tensor_name],
                                name=f"perturbnormal_{perturbed_tensor_name}",
                                domain="mezo",
                                seed=seed,
                                eps=epsilon,
                                idx=perturbation_counter,
                                # dtype=dtype,
                                doc_string="y = x + epsilon * RandomNormal(x, seed)"
                            )
                            new_nodes.append(perturbation_node)

                        elif noise_type == "uniform":
                            perturbation_node = helper.make_node(
                                "PerturbUniform",
                                inputs=[input_name],
                                outputs=[perturbed_tensor_name],
                                name=f"perturbuniform_{perturbed_tensor_name}",
                                domain="mezo",
                                idx=perturbation_counter,
                                seed=seed,
                                eps=epsilon,
                                low=-np.sqrt(3),
                                high=np.sqrt(3),
                                # dtype=dtype,
                                doc_string="y = x + epsilon * RandomUniform(x, seed)"
                            )
                            new_nodes.append(perturbation_node)

                        elif noise_type == "triangle":
                            perturbation_node = helper.make_node(
                                "PerturbTriangle",
                                inputs=[input_name],
                                outputs=[perturbed_tensor_name],
                                name=f"perturbtriangle_{perturbed_tensor_name}",
                                domain="mezo",
                                idx=perturbation_counter,
                                seed=seed,
                                eps=epsilon,
                                low=-np.sqrt(6),
                                high=np.sqrt(6),
                                # dtype=dtype,
                                doc_string="y = x + epsilon * RandomTriangle(x, seed)"
                            )
                            new_nodes.append(perturbation_node)

                        elif noise_type == "rademacher":
                            perturbation_node = helper.make_node(
                                "PerturbRademacher",
                                inputs=[input_name],
                                outputs=[perturbed_tensor_name],
                                name=f"perturbrademacher_{perturbed_tensor_name}",
                                domain="mezo",
                                idx=perturbation_counter,
                                seed=seed,
                                eps=epsilon,
                                # dtype=dtype,
                                doc_string="y = x + epsilon * RandomRademacher(x, seed)"
                            )
                            new_nodes.append(perturbation_node)

                        elif noise_type == "eggroll":
                            # Shape annotation for intermediate outputs
                            a_shape = [noise_shape[0], 1]
                            b_shape = [int(np.prod(noise_shape[1:])), 1]
                            shape_input_name = helper.make_tensor(name=f"shape_{input_name}", data_type=TensorProto.INT64, dims=[len(noise_shape)],
                                                vals=np.array(noise_shape, dtype=np.int64))
                            new_initializers.append(shape_input_name)

                            if len(noise_shape) > 2:

                                shape_flat_name = helper.make_tensor(name=f"shape_{input_name}_flat", data_type=TensorProto.INT64, dims=[2],
                                                                      vals=np.array([a_shape[0], b_shape[0]], dtype=np.int64))
                               
                                new_initializers.append(shape_flat_name)

                                # insert flattening nodes
                                flatten_node = helper.make_node(
                                    "Reshape",
                                    inputs=[input_name, f"shape_{input_name}_flat"],
                                    outputs=[f"flattened_{input_name}"],
                                    name=f"flatten_{input_name}"
                                )
                                new_nodes.append(flatten_node)
                                extra_value_infos.append(helper.make_tensor_value_info(
                                    f"flattened_{input_name}", TensorProto.FLOAT,[noise_shape[0], int(np.prod(noise_shape[1:]))]
                                ))

                                unflatten_node = helper.make_node(
                                    "Reshape",
                                    inputs=[f"flattened_{perturbed_tensor_name}", f"shape_{input_name}"],
                                    outputs=[perturbed_tensor_name],
                                    name=f"unflatten_{perturbed_tensor_name}"
                                )
                                new_nodes.append(unflatten_node)
                                extra_value_infos.append(helper.make_tensor_value_info(
                                    f"flattened_{perturbed_tensor_name}", TensorProto.FLOAT, [noise_shape[0], int(np.prod(noise_shape[1:]))]
                                ))
                                eggroll_input = f"flattened_{input_name}"
                                eggroll_output = f"flattened_{perturbed_tensor_name}"
                            else:
                                eggroll_input = input_name
                                eggroll_output = perturbed_tensor_name

                            extra_value_infos.append(helper.make_tensor_value_info(
                                f"a_{perturbed_tensor_name}", TensorProto.FLOAT, a_shape
                            ))
                            extra_value_infos.append(helper.make_tensor_value_info(
                                f"b_{perturbed_tensor_name}", TensorProto.FLOAT, b_shape
                            ))

                            # Eggroll noise node (without loss_grad input)
                            noise_node_a = helper.make_node(
                                "PerturbEggroll",
                                inputs=[f"shape_{input_name}"],
                                outputs=[f"a_{perturbed_tensor_name}"],
                                name=f"gen_eggroll_noise_a_{perturbed_tensor_name}",
                                seed=seed,
                                idx=perturbation_counter,
                                domain="com.microsoft",
                                doc_string="a = RandomRademacher(x[0], seed)"
                            )
                            
                            noise_node_b = helper.make_node(
                                "PerturbEggroll",
                                inputs=[f"shape_{input_name}"],
                                outputs=[f"b_{perturbed_tensor_name}"],
                                name=f"gen_eggroll_noise_b_{perturbed_tensor_name}",
                                seed=seed,
                                idx=perturbation_counter,
                                domain="com.microsoft",
                                doc_string="b = RandomRademacher(x[1:], seed)"
                            )

                            gemm_node = helper.make_node(
                                "Gemm",
                                inputs=[f"a_{perturbed_tensor_name}", f"b_{perturbed_tensor_name}", eggroll_input],
                                outputs=[eggroll_output],
                                name=f"eggroll_gemm_{perturbed_tensor_name}",
                                transA=0,
                                transB=1,
                                alpha=epsilon,
                                beta=0
                            )

                            new_nodes.append(noise_node_a)
                            new_nodes.append(noise_node_b)
                            new_nodes.append(gemm_node)

                        # **CRITICAL**: annotate perturbed edge with same dtype/shape as weight
                        if len(original_weight_tensor.dims) == 1:
                            out_shape = (original_weight_tensor.dims[0], )
                            print(f"out_shape: {out_shape}")
                        else:
                            out_shape = original_weight_tensor.dims
                        extra_value_infos.append(
                            helper.make_tensor_value_info(perturbed_tensor_name,
                                                          elem_type=TensorProto.FLOAT,
                                                           shape=out_shape)
                        )

                        # 5. Update the input list for the *original* node
                        modified_inputs[i] = perturbed_tensor_name
                        perturbation_counter += 1

                if made_change:

                    # handle attributes
                    kwargs = {}
                    for attr in node.attribute:
                        # Use get_attribute_value to extract the python value from the AttributeProto
                        kwargs[attr.name] = helper.get_attribute_value(attr)

                    # Create a new version of the Conv/Gemm node with the modified inputs
                    new_original_node = helper.make_node(
                        node.op_type,
                        modified_inputs, # Use the updated input list
                        node.output,
                        name=node.name,
                        domain=node.domain,
                        **kwargs
                    )
                    new_nodes.append(new_original_node)
                else:
                    # If no weights were perturbed, add the original node back unchanged
                    new_nodes.append(node)
            else:
                # This node is not a target, so add it to our new list as-is
                new_nodes.append(node)

        new_value_info = list(original_model.graph.value_info) + extra_value_infos

        # Create a new graph with the new list of nodes and initializers
        new_graph = helper.make_graph(
            nodes=new_nodes,
            name=f"{original_model.graph.name}-{node.op_type}",
            inputs=original_model.graph.input,
            outputs=original_model.graph.output,
            initializer=new_initializers,
            value_info=new_value_info
        )

        # Create and save the new model
        for op in original_model.opset_import:
            if op.domain == "":
                standard_opset_version = op.version
                break

        opset_list = [
            # Add the standard opset with the version we found
            helper.make_opsetid("", standard_opset_version),

            # Addcustom domain
            helper.make_opsetid("mezo", 1)
        ]
        new_model = helper.make_model(new_graph, producer_name="mezo-graph-generator",
                    opset_imports=opset_list)

        # onnx.checker.check_model(new_model)
        onnx.save(new_model, output_path)

    # --- Main execution ---
    original_model = onnx.load(onnx_path)

    print(f"Found {len(original_model.graph.initializer)} initializers. Perturbing weights/biases in Conv, MatMul, Gemm nodes.")

    modify_graph(original_model, output_path, exceptions=exceptions)

    print(f"Saved perturbed models to:\n- {output_path}")
    return output_path

def append_cross_entropy_loss(onnx_path, output_path, label_name='y', logits_output_idx=0, reduction='mean'):
    """
    Adds a brand-new label input (INT64, shape ['batch_size']) -- guaranteed to be a new input name --
    appends a SoftmaxCrossEntropyLoss node consuming the model logits and the new label input,
    and replaces the graph output with the scalar loss.
    """
    model = onnx.load(onnx_path)
    graph = model.graph

    if len(graph.output) == 0:
        raise RuntimeError("Model has no outputs to attach the loss to.")

    # resolve logits tensor name (default: first graph output)
    if logits_output_idx < 0 or logits_output_idx >= len(graph.output):
        raise RuntimeError(f"Invalid logits_output_idx {logits_output_idx}")
    logits_name = graph.output[logits_output_idx].name

    existing_inputs = {inp.name for inp in graph.input}
    label_input_name = label_name
    suffix = 0
    while label_input_name in existing_inputs:
        label_input_name = f"{label_name}_mezo{suffix}"
        suffix += 1

    # get batch size from the first graph input (no initializer checks)
    batch_dim = "batch_size"  # fallback symbolic name
    if graph.input:
        first_inp = graph.input[0]
        if first_inp.type.HasField("tensor_type") and first_inp.type.tensor_type.shape.dim:
            first_dim = first_inp.type.tensor_type.shape.dim[0]
            if first_dim.HasField("dim_value") and first_dim.dim_value > 0:
                batch_dim = int(first_dim.dim_value)
            elif first_dim.HasField("dim_param") and first_dim.dim_param:
                batch_dim = first_dim.dim_param

    # add the new label input using resolved batch_dim
    label_vi = helper.make_tensor_value_info(label_input_name, TensorProto.INT8, [batch_dim])
    graph.input.append(label_vi)

    # create loss node (standard SoftmaxCrossEntropyLoss) with proper attribute
    loss_name = "loss"
    loss_node = helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=[logits_name, label_input_name],
        outputs=[loss_name],
        name="CrossEntropyLoss",
        reduction=reduction,
    )
    graph.node.append(loss_node)

    # replace graph outputs with the scalar loss
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info(loss_name, TensorProto.FLOAT, []))

    # try to infer shapes and save
    try:
        inferred = shape_inference.infer_shapes(model)
        onnx.save(inferred, output_path)
    except Exception:
        onnx.save(model, output_path)
