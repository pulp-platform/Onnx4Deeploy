# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Generic optimizer ONNX graph builder for Deeploy training networks.

Builds a Deeploy SGD optimizer graph from a training network.onnx by
auto-detecting trainable parameters via their grad accumulation buffer inputs.

Convention: a graph input named ``<param>_grad.accumulation.buffer`` implies
``<param>`` is a trainable parameter. The resulting optimizer graph has one
``SGD`` node per trainable parameter:

    SGD(param, grad_acc_buf, attrs={lr}) → param_updated

Input ordering (interleaved pairs, matching deeploytraintest.c expectations):
    [2*i  ] param_i
    [2*i+1] param_i_grad.accumulation.buffer

Output ordering:
    [i] param_i_updated
"""

from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import onnx_graphsurgeon as gs

_GRAD_ACC = "_grad.accumulation.buffer"


def create_optimizer_onnx(
    train_dir: str,
    output_path: str,
    lr: float = 0.001,
) -> None:
    """Build and save a Deeploy SGD optimizer ONNX graph.

    Reads the training ``network.onnx`` from *train_dir* to discover the
    shapes of all trainable parameters, then writes a minimal SGD graph to
    *output_path*.

    Parameters
    ----------
    train_dir:
        Directory that contains the training ``network.onnx``.
    output_path:
        Destination path for the optimizer ``network.onnx``.
    lr:
        SGD learning rate embedded as the ``lr`` node attribute.
    """
    train_onnx_path = Path(train_dir) / "network.onnx"
    if not train_onnx_path.exists():
        raise FileNotFoundError(f"Training ONNX not found: {train_onnx_path}")

    train_model = onnx.load(str(train_onnx_path))
    train_graph = gs.import_onnx(train_model)

    # Build a name→variable map for all graph inputs.
    input_by_name = {inp.name: inp for inp in train_graph.inputs}

    # Detect trainable params: any input with a matching grad acc buffer.
    trainable_params = []
    for inp in train_graph.inputs:
        if _GRAD_ACC in inp.name:
            param_name = inp.name[: -len(_GRAD_ACC)]
            if param_name in input_by_name:
                trainable_params.append(input_by_name[param_name])

    if not trainable_params:
        raise ValueError(
            "No trainable parameters detected in network.onnx. "
            "Ensure the training graph contains <param>_grad.accumulation.buffer inputs."
        )

    print(f"Found {len(trainable_params)} trainable parameter(s):")
    for p in trainable_params:
        print(f"  {p.name}  shape={list(p.shape)}  dtype={p.dtype}")

    graph_inputs: list = []
    graph_outputs: list = []
    nodes: list = []

    for p_inp in trainable_params:
        p_name = p_inp.name
        shape = list(p_inp.shape)
        grad_name = p_name + _GRAD_ACC

        p_var = gs.Variable(p_name, dtype=np.float32, shape=shape)
        g_var = gs.Variable(grad_name, dtype=np.float32, shape=shape)
        p_out_var = gs.Variable(f"{p_name}_updated", dtype=np.float32, shape=shape)

        node = gs.Node(
            op="SGD",
            name=f"sgd_{p_name}",
            inputs=[p_var, g_var],
            outputs=[p_out_var],
            attrs={"lr": float(lr)},
        )

        graph_inputs.extend([p_var, g_var])
        graph_outputs.append(p_out_var)
        nodes.append(node)

    graph = gs.Graph(
        nodes=nodes,
        inputs=graph_inputs,
        outputs=graph_outputs,
        name="optimizer",
    )
    graph.cleanup().toposort()

    model = gs.export_onnx(graph)
    model.opset_import[0].version = 17  # SGD is a Deeploy custom op

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))

    print(f"Saved optimizer ONNX → {output_path}")
    print(f"  Inputs  ({len(graph_inputs)}): " + ", ".join(v.name for v in graph_inputs))
    print(f"  Outputs ({len(graph_outputs)}): " + ", ".join(v.name for v in graph_outputs))
    print(f"  SGD lr = {lr}")


def derive_optimizer_dir(train_dir: str) -> Optional[str]:
    """Return the conventional optimizer directory next to a training directory.

    Convention: ``<base>/<model>_train`` → ``<base>/<model>_optimizer``.
    Returns *None* if *train_dir* does not contain ``_train``.
    """
    p = Path(train_dir)
    if "_train" not in p.name:
        return None
    opt_name = p.name.replace("_train", "_optimizer")
    return str(p.parent / opt_name)
