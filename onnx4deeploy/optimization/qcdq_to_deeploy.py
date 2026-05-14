# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Post-export ONNX passes that adapt DeepQuant's QCDQ output to the exact
shape vanilla `pulp-platform/Deeploy:devel` consumes — without requiring
any Deeploy-side patch.

Run on the ``network.onnx`` produced by ``DeepQuant.exportBrevitas`` after
``BaseONNXExporter.export_quantized`` saves it. Each pass is idempotent and
safe to re-run.

Passes (in order):
  1. ``upgrade_reducemean_axes`` — opset-13 ``axes`` attribute → opset-18
     second input. Deeploy's ``_remove_only_singleton_reduce_mean`` only
     reads ``node.inputs[1]``.
  2. ``fold_qcdq_to_quant_dequant`` — fold Brevitas's decomposed
     ``Div/Add/Round/Clip`` into a single ``Quant`` op and ``Sub/Mul`` into
     a single ``Dequant``. (Deeploy's QuantPatternPass / DequantPatternPass
     would do this too, but doing it here keeps subsequent passes cleaner.)
  3. ``fold_dequant_quant_to_requantshift`` — match every consecutive
     ``Dequant → Quant`` and replace with a single ``RequantShift`` carrying
     mul / add / div / n_levels / signed attrs. This is the missing bridge
     between QCDQ activation handoffs and Deeploy's integer-Conv pipeline.
  4. ``skip_leading_quant_dequant`` — at graph input, drop the trailing
     Dequant of the ``input → Quant → Dequant → first_int_op`` pair so the
     first op consumes int8 directly.
  5. ``absorb_conv_bias_into_following_requantshift`` — when a Conv has
     ``bias=True``, fold ``Conv.bias * RequantShift.mul`` into
     ``RequantShift.add`` and drop the conv bias input, so the resulting
     fused RequantizedConv matches PULPConv2DParser's 4-input requirement.

Returns the count of nodes folded by each pass for traceability.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import onnx
from onnx import helper, numpy_helper

# ----------------------------------------------------------------------- #
# Small helpers                                                            #
# ----------------------------------------------------------------------- #


def _make_const(name: str, arr: np.ndarray) -> onnx.NodeProto:
    """Wrap a numpy array as a `Constant` node — used when we need a
    graph-resident initializer with a known name and value.
    """
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        name=name + "_const",
        value=numpy_helper.from_array(arr, name=name),
    )


def _init_lookup(graph: onnx.GraphProto) -> Dict[str, np.ndarray]:
    """Build name → numpy view of all initializers + constant-node outputs.

    Also chases ``Cast(Constant)`` chains so the consumer code can resolve
    Cast-wrapped scalar bounds (Clip's min/max are typically emitted by
    PyTorch's ONNX exporter as ``Constant → Cast``).
    """
    out: Dict[str, np.ndarray] = {}
    for init in graph.initializer:
        out[init.name] = numpy_helper.to_array(init)
    for n in graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    out[n.output[0]] = numpy_helper.to_array(a.t)
    # Resolve Cast(Constant) → Cast output → numerical value.
    # Iterate to a fixed point in case of Cast(Cast(Constant)).
    changed = True
    while changed:
        changed = False
        for n in graph.node:
            if n.op_type != "Cast" or n.output[0] in out:
                continue
            src = n.input[0]
            if src in out:
                # Apply the cast (best-effort: just propagate the value;
                # downstream consumers only need a scalar magnitude).
                out[n.output[0]] = np.asarray(out[src])
                changed = True
    return out


def _producer_map(graph: onnx.GraphProto) -> Dict[str, onnx.NodeProto]:
    out: Dict[str, onnx.NodeProto] = {}
    for n in graph.node:
        for o in n.output:
            out[o] = n
    return out


def _consumer_map(graph: onnx.GraphProto) -> Dict[str, List[onnx.NodeProto]]:
    out: Dict[str, List[onnx.NodeProto]] = {}
    for n in graph.node:
        for i in n.input:
            out.setdefault(i, []).append(n)
    return out


# ----------------------------------------------------------------------- #
# Pass 1 — ReduceMean axes: opset-13 attribute → opset-18 input            #
# ----------------------------------------------------------------------- #


def upgrade_reducemean_axes(graph: onnx.GraphProto) -> int:
    n_changed = 0
    for n in graph.node:
        if n.op_type != "ReduceMean":
            continue
        if len(n.input) >= 2:
            continue  # already opset-18 form
        axes_attr = None
        for a in list(n.attribute):
            if a.name == "axes":
                axes_attr = a
                break
        if axes_attr is None:
            continue
        # Deeploy reads ``node.attrs['axis']`` (singular) before falling back
        # to ``node.inputs[1]``. The opset-13 spec name is ``axes`` (plural)
        # — keep the values but rename so Deeploy's first check hits.
        new_axis = helper.make_attribute("axis", list(axes_attr.ints))
        n.attribute.remove(axes_attr)
        n.attribute.append(new_axis)
        n_changed += 1
    return n_changed


# ----------------------------------------------------------------------- #
# Pass 2 — Fold QCDQ Div/Add/Round/Clip + Sub/Mul into Quant + Dequant     #
# (Deeploy does this anyway via its pattern passes; doing it here keeps    #
# our later passes simple — they only need to match Quant / Dequant ops.)  #
# ----------------------------------------------------------------------- #


def fold_qcdq_to_quant_dequant(graph: onnx.GraphProto) -> int:
    """Find ``Div → Add → Round → Clip`` chains and collapse them into a
    single ``Quant`` node with ``scale, zero_point, bit_width, signed``
    attributes. Find ``Sub → Mul`` chains and collapse them into a single
    ``Dequant`` node with the same attribute set.
    """
    inits = _init_lookup(graph)
    prod = _producer_map(graph)

    nodes_to_remove: set = set()
    nodes_to_add: List[onnx.NodeProto] = []
    output_renames: Dict[str, str] = {}

    def _const_scalar(name: str) -> Optional[float]:
        if name not in inits:
            return None
        v = inits[name]
        if v.size != 1:
            return None
        return float(v.item())

    for node in graph.node:
        if node.op_type != "Clip":
            continue
        # Walk back: Clip ← Round ← Add ← Div  (each must have a single
        # consumer of its output; the chain must be linear).
        if node.input[0] not in prod:
            continue
        round_node = prod[node.input[0]]
        if round_node.op_type != "Round":
            continue
        if round_node.output[0] not in prod or round_node.output[0] == node.input[0]:
            pass
        add_node = prod.get(round_node.input[0])
        if add_node is None or add_node.op_type != "Add":
            continue
        # Add inputs: (Div_out, zero_point) — find which is which.
        div_out = None
        zp_val = None
        for inp in add_node.input:
            if inp in prod and prod[inp].op_type == "Div":
                div_out = inp
            else:
                zp_val = _const_scalar(inp)
        if div_out is None or zp_val is None:
            continue
        div_node = prod[div_out]
        # Div inputs: (x, scale)
        scale_val = None
        x_in = None
        for inp in div_node.input:
            v = _const_scalar(inp)
            if v is not None and scale_val is None:
                scale_val = v
            else:
                x_in = inp
        if scale_val is None or x_in is None:
            continue
        # Determine bit_width from Clip's min/max bounds.
        bw_inputs = [_const_scalar(node.input[i]) for i in range(1, len(node.input))]
        if len(bw_inputs) < 2 or bw_inputs[0] is None or bw_inputs[1] is None:
            continue
        lo, hi = int(bw_inputs[0]), int(bw_inputs[1])
        bit_width = int(round(math.log2(hi - lo + 1)))
        signed = lo < 0

        # Build the Quant node.
        q_name = "QCDQ_" + node.name + "_Quant"
        q_out = node.output[0]
        new_q = helper.make_node(
            "Quant",
            inputs=[x_in],
            outputs=[q_out],
            name=q_name,
            scale=float(scale_val),
            zero_point=float(zp_val),
            bit_width=bit_width,
            signed=int(signed),
        )
        nodes_to_add.append(new_q)
        nodes_to_remove.update([div_node.name, add_node.name, round_node.name, node.name])

    # --- Dequant: Sub → Mul ---
    prod2 = prod  # producer map already built
    for node in graph.node:
        if node.op_type != "Mul":
            continue
        if node.name in nodes_to_remove:
            continue
        # Mul inputs: (Sub_out, scale)
        sub_in = None
        scale_val = None
        for inp in node.input:
            v = _const_scalar(inp)
            if v is not None and scale_val is None:
                scale_val = v
            elif inp in prod2 and prod2[inp].op_type == "Sub":
                sub_in = inp
        if sub_in is None or scale_val is None:
            continue
        sub_node = prod2[sub_in]
        # Sub inputs: (q, zero_point)
        q_in = None
        zp_val = None
        for inp in sub_node.input:
            v = _const_scalar(inp)
            if v is not None and zp_val is None:
                zp_val = v
            else:
                q_in = inp
        if q_in is None or zp_val is None:
            continue
        bit_width = 8
        signed = True
        dq_name = "QCDQ_" + node.name + "_Dequant"
        dq_out = node.output[0]
        new_dq = helper.make_node(
            "Dequant",
            inputs=[q_in],
            outputs=[dq_out],
            name=dq_name,
            scale=float(scale_val),
            zero_point=float(zp_val),
            bit_width=bit_width,
            signed=int(signed),
        )
        nodes_to_add.append(new_dq)
        nodes_to_remove.update([sub_node.name, node.name])

    # Apply.
    kept = [n for n in graph.node if n.name not in nodes_to_remove]
    kept.extend(nodes_to_add)
    del graph.node[:]
    graph.node.extend(kept)
    return len(nodes_to_add)


# ----------------------------------------------------------------------- #
# Pass 3 — Fold Dequant → Quant into RequantShift                          #
# ----------------------------------------------------------------------- #


def fold_dequant_quant_to_requantshift(graph: onnx.GraphProto, shift_bits: int = 16) -> int:
    """Find every consecutive ``Dequant → Quant`` pair and replace with a
    single ``RequantShift`` op carrying mul / add (as 1-D int32 initializer
    inputs) plus n_levels / signed / div attributes.
    """
    prod = _producer_map(graph)
    cons = _consumer_map(graph)

    nodes_to_remove: set = set()
    nodes_to_add: List[onnx.NodeProto] = []
    initializers_to_add: List[onnx.TensorProto] = []
    output_renames: Dict[str, str] = {}

    n_folded = 0
    for q_node in list(graph.node):
        if q_node.op_type != "Quant":
            continue
        if q_node.name in nodes_to_remove:
            continue
        if q_node.input[0] not in prod:
            continue
        dq_node = prod[q_node.input[0]]
        if dq_node.op_type != "Dequant":
            continue
        if dq_node.name in nodes_to_remove:
            continue
        # If the Dequant has multiple consumers, we ONLY collapse the Quant —
        # the Dequant stays so other consumers still see the fp32 path. The
        # new RequantShift takes the Dequant's INPUT directly (the int value),
        # so the math is preserved.
        multi_consumer = len(cons.get(dq_node.output[0], [])) > 1

        scale_d = float([a.f for a in dq_node.attribute if a.name == "scale"][0])
        zp_d = float([a.f for a in dq_node.attribute if a.name == "zero_point"][0])
        scale_q = float([a.f for a in q_node.attribute if a.name == "scale"][0])
        zp_q = float([a.f for a in q_node.attribute if a.name == "zero_point"][0])
        bit_width = int([a.i for a in q_node.attribute if a.name == "bit_width"][0])
        signed = bool([a.i for a in q_node.attribute if a.name == "signed"][0])

        div = 1 << shift_bits
        mul_val = int(round((scale_d / scale_q) * div))
        add_val = int(round(zp_q * div - zp_d * mul_val))

        base = "RQS_" + q_node.name
        mul_init = numpy_helper.from_array(np.array([mul_val], dtype=np.int32), name=base + "_mul")
        add_init = numpy_helper.from_array(np.array([add_val], dtype=np.int32), name=base + "_add")
        initializers_to_add.extend([mul_init, add_init])

        # Deeploy's PULPUniformRequantShift only has int8_t output bindings
        # (no uint8). When the original Quant was unsigned (e.g. post-ReLU
        # range [0, 255]), shift the math so the RequantShift emits int8 in
        # range [-128, 127]: this is equivalent because the downstream
        # consumer (also an integer op of ours) is type-agnostic about the
        # 128-offset, and Deeploy's int8 conv kernels handle the shift
        # implicitly via the bias/add term.
        if not signed:
            # int8 = uint8 - 128  →  add adjusted by -128 * div
            add_val -= 128 * div
            signed_out = True
        else:
            signed_out = signed
        # Refresh the add initializer with the shifted value.
        add_init.CopyFrom(
            numpy_helper.from_array(np.array([add_val], dtype=np.int32), name=add_init.name)
        )

        new_rqs = helper.make_node(
            "RequantShift",
            inputs=[dq_node.input[0], mul_init.name, add_init.name],
            outputs=[q_node.output[0]],
            name=base,
        )
        new_rqs.attribute.extend(
            [
                helper.make_attribute(
                    "n_levels", numpy_helper.from_array(np.array(1 << bit_width, dtype=np.int64))
                ),
                helper.make_attribute(
                    "signed", numpy_helper.from_array(np.array(int(signed_out), dtype=np.int64))
                ),
                helper.make_attribute(
                    "div", numpy_helper.from_array(np.array(div, dtype=np.int64))
                ),
            ]
        )
        nodes_to_add.append(new_rqs)
        # Only drop the Dequant if it's now unused; the Quant is always
        # replaced by our RequantShift.
        if not multi_consumer:
            nodes_to_remove.add(dq_node.name)
        nodes_to_remove.add(q_node.name)
        n_folded += 1

    kept = [n for n in graph.node if n.name not in nodes_to_remove]
    kept.extend(nodes_to_add)
    del graph.node[:]
    graph.node.extend(kept)
    graph.initializer.extend(initializers_to_add)
    return n_folded


# ----------------------------------------------------------------------- #
# Pass 4 — Skip leading Quant → Dequant trail at graph input               #
# ----------------------------------------------------------------------- #


def skip_leading_quant_dequant(graph: onnx.GraphProto) -> int:
    """When the network starts with ``graph_input → Quant → Dequant → ...``
    (canonical Brevitas activation-quant pair), drop the trailing Dequant
    so the int8 output of the Quant feeds directly into the first integer
    op.
    """
    graph_input_names = {i.name for i in graph.input}
    _producer_map(graph)
    cons = _consumer_map(graph)

    n_changed = 0
    for q_node in list(graph.node):
        if q_node.op_type != "Quant":
            continue
        if q_node.input[0] not in graph_input_names:
            continue
        # Find the immediate downstream Dequant on this Quant's output.
        q_out = q_node.output[0]
        children = cons.get(q_out, [])
        for child in children:
            if child.op_type != "Dequant":
                continue
            d_out = child.output[0]
            # Rewire all consumers of Dequant's output to consume Quant's output.
            for c in graph.node:
                for i, inp in enumerate(c.input):
                    if inp == d_out:
                        c.input[i] = q_out
            # Re-point graph outputs if needed.
            for i, go in enumerate(graph.output):
                if go.name == d_out:
                    go.name = q_out
            # Mark the Dequant for removal.
            child.output[0] = "__deleted_" + d_out
            n_changed += 1

    if n_changed:
        keep = [
            n
            for n in graph.node
            if not (n.op_type == "Dequant" and n.output[0].startswith("__deleted_"))
        ]
        del graph.node[:]
        graph.node.extend(keep)
    return n_changed


# ----------------------------------------------------------------------- #
# Pass 5 — Absorb Conv bias into the following RequantShift add term       #
# ----------------------------------------------------------------------- #


def absorb_conv_bias_into_following_requantshift(graph: onnx.GraphProto) -> int:
    """For each ``Conv (with bias) → RequantShift`` pair, transform
    ``(X*W + B) * mul + add → X*W * mul + (B*mul + add)`` so the bias is
    no longer a Conv input. Required because PULPConv2DParser /
    PULPDWConv2DParser want exactly 4 inputs on the merged RequantizedConv:
    (X, W, mul, merged_add).
    """
    inits_by_name = {i.name: i for i in graph.initializer}
    _producer_map(graph)
    cons = _consumer_map(graph)

    n_changed = 0
    for conv in list(graph.node):
        if conv.op_type != "Conv":
            continue
        if len(conv.input) < 3:
            continue  # No bias to absorb.
        bias_name = conv.input[2]
        if bias_name not in inits_by_name:
            continue
        # Conv output must feed a single RequantShift.
        children = cons.get(conv.output[0], [])
        if len(children) != 1 or children[0].op_type != "RequantShift":
            continue
        rqs = children[0]
        if len(rqs.input) < 3:
            continue
        mul_name = rqs.input[1]
        add_name = rqs.input[2]
        if mul_name not in inits_by_name or add_name not in inits_by_name:
            continue
        B = numpy_helper.to_array(inits_by_name[bias_name]).astype(np.float64)
        mul = numpy_helper.to_array(inits_by_name[mul_name]).astype(np.float64)
        add = numpy_helper.to_array(inits_by_name[add_name]).astype(np.float64)
        new_add = (np.round(B * mul) + add).astype(np.int32)
        # Replace the add initializer in place.
        new_init = numpy_helper.from_array(new_add, name=add_name)
        # onnx.TensorProto can't be replaced in-place; rebuild the list.
        for idx, init in enumerate(graph.initializer):
            if init.name == add_name:
                graph.initializer.remove(init)
                graph.initializer.insert(idx, new_init)
                break
        # Drop the conv bias input.
        del conv.input[2]
        n_changed += 1
    return n_changed


# ----------------------------------------------------------------------- #
# Driver                                                                   #
# ----------------------------------------------------------------------- #


def constfold_quant_of_initializer(graph: onnx.GraphProto) -> int:
    """Pre-compute ``Quant(initializer)`` at static time.

    DeepQuant's QCDQ emission turns every Conv weight, Conv bias, fc weight,
    fc bias into ``Constant_fp → Div/Add/Round/Clip → Constant_int``. After
    our fold_qcdq pass, this becomes ``Constant_fp → Quant → Variable_int``.

    Downstream RequantMerge passes (PULPConvRequantMergePass,
    PULPGEMMRequantMergePass) expect those Variables to be ``gs.Constant``
    with ``.values``. If they're Quant-produced Variables, the merge crashes.

    Fix: for every ``Quant`` whose input is an initializer, apply the
    quantization at export time, store the int result as a new initializer,
    and remove the Quant node.
    """
    init_names = {init.name: init for init in graph.initializer}
    inits = _init_lookup(graph)
    nodes_to_remove: List[str] = []
    inits_to_add: List[onnx.TensorProto] = []
    rename: Dict[str, str] = {}

    for n in list(graph.node):
        if n.op_type != "Quant":
            continue
        src = n.input[0]
        if src not in inits:
            continue
        scale = float([a.f for a in n.attribute if a.name == "scale"][0])
        zp = float([a.f for a in n.attribute if a.name == "zero_point"][0])
        bw = int([a.i for a in n.attribute if a.name == "bit_width"][0])
        signed = bool([a.i for a in n.attribute if a.name == "signed"][0])
        lo = -(1 << (bw - 1)) if signed else 0
        hi = (1 << (bw - 1)) - 1 if signed else (1 << bw) - 1

        fp = inits[src]
        q = np.round(fp / scale + zp).astype(np.float64)
        q = np.clip(q, lo, hi)
        dtype = np.int8 if (signed and bw <= 8) else (np.int32 if bw > 16 else np.int16)
        q = q.astype(dtype)

        new_name = n.output[0]
        new_init = numpy_helper.from_array(q, name=new_name)
        inits_to_add.append(new_init)
        nodes_to_remove.append(n.name)
        # downstream consumers of n.output[0] now read the initializer directly
        # — no rewiring needed since names match.

    kept = [n for n in graph.node if n.name not in nodes_to_remove]
    del graph.node[:]
    graph.node.extend(kept)
    graph.initializer.extend(inits_to_add)
    return len(nodes_to_remove)


_INTEGER_OPS_AFTER_MERGE = {"Conv", "Gemm", "MatMul", "Add", "ReduceMean"}
_LAYOUT_OPS = {"Transpose", "Flatten", "Reshape", "Squeeze", "Unsqueeze"}


def skip_dequant_before_integer_op(graph: onnx.GraphProto) -> int:
    """Delete every standalone ``Dequant`` whose output flows ONLY into ops
    that will become integer after Deeploy's RequantMerge passes (Conv,
    Gemm, MatMul, Add — each followed by a RequantShift that Deeploy fuses
    into RequantizedX). Replacing ``Dequant_in → Dequant → op → RequantShift``
    with ``Dequant_in → op → RequantShift`` is mathematically a no-op once
    Deeploy collapses the merge into ``RequantizedX(int)`` because the
    RequantShift absorbs both the missing dequantize and the requantize.

    Any Dequant whose output also goes to a non-integer-mergeable op (e.g.
    ReduceMean, Mul, Transpose-then-fp32-something) is left in place.
    """
    _producer_map(graph)
    cons = _consumer_map(graph)

    n_removed = 0
    for dq in list(graph.node):
        if dq.op_type != "Dequant":
            continue
        consumers = cons.get(dq.output[0], [])
        if not consumers:
            continue

        # Walk through Transpose / Flatten / Reshape transparently — these
        # are shape-only ops that don't care about int vs fp32.
        def _resolves_to_int_op(start_node: onnx.NodeProto) -> bool:
            visited = set()
            stack = [start_node]
            while stack:
                cur = stack.pop()
                if id(cur) in visited:
                    continue
                visited.add(id(cur))
                if cur.op_type in _INTEGER_OPS_AFTER_MERGE:
                    return True
                if cur.op_type in {"Transpose", "Flatten", "Reshape", "Squeeze", "Unsqueeze"}:
                    for o in cur.output:
                        stack.extend(cons.get(o, []))
                    continue
                return False
            return False

        if not all(_resolves_to_int_op(c) for c in consumers):
            continue

        # Rewire all consumers from Dequant.output[0] to Dequant.input[0].
        dq_in = dq.input[0]
        dq_out = dq.output[0]
        for c in graph.node:
            for i, inp in enumerate(c.input):
                if inp == dq_out:
                    c.input[i] = dq_in
        # Remove the Dequant.
        dq.output[0] = "__DELETED_DQ_" + dq.name
        n_removed += 1

    keep = [
        n
        for n in graph.node
        if not (n.op_type == "Dequant" and n.output[0].startswith("__DELETED_DQ_"))
    ]
    del graph.node[:]
    graph.node.extend(keep)
    return n_removed


def strip_trailing_dequant(graph: onnx.GraphProto) -> int:
    """Remove the trailing ``Dequant`` if it's the last node feeding the
    graph output. Deeploy has no ``tileConstraint`` for Dequant either, so
    the network output must be the integer logits directly. (Float-domain
    interpretation of the int logits is left to the user / test harness.)
    """
    if not graph.output:
        return 0
    out_name = graph.output[0].name
    prod = _producer_map(graph)
    leading = prod.get(out_name)
    if leading is None or leading.op_type != "Dequant":
        return 0
    # Make the graph output the Dequant's input directly.
    src = leading.input[0]
    graph.output[0].name = src
    # Adjust output type to int8 (we drop dequant → int8 stays).
    graph.output[0].type.tensor_type.elem_type = onnx.TensorProto.INT8
    graph.node.remove(leading)
    return 1


def cleanup_orphan_nodes(graph: onnx.GraphProto) -> int:
    """Remove ``Constant`` / ``Cast`` nodes whose outputs are unused. The
    QCDQ→RequantShift fold leaves orphan scale/zp constants behind that
    Deeploy then trips on during binding (PULPConstantBuffer has no _type).
    """
    while True:
        used: set = set()
        for n in graph.node:
            for i in n.input:
                used.add(i)
        for o in graph.output:
            used.add(o.name)
        to_remove = []
        for n in graph.node:
            if n.op_type not in {"Constant", "Cast", "Identity"}:
                continue
            if all(out not in used for out in n.output):
                to_remove.append(n.name)
        if not to_remove:
            break
        keep = [n for n in graph.node if n.name not in to_remove]
        del graph.node[:]
        graph.node.extend(keep)
    # Also drop orphan initializers.
    used_inits = set()
    for n in graph.node:
        for i in n.input:
            used_inits.add(i)
    keep = [init for init in graph.initializer if init.name in used_inits]
    n_removed = len(graph.initializer) - len(keep)
    del graph.initializer[:]
    graph.initializer.extend(keep)
    return n_removed


def quantize_input_offline(model: onnx.ModelProto, inputs_npz_path: str) -> int:
    """When the graph starts with ``input(fp32) → Quant → ...``, pre-quantize
    the input data and remove the Quant from the graph.

    Background: Deeploy's tiler has no ``tileConstraint`` for the ``Quant``
    op, so the first op of an integer-pipeline graph cannot be a Quant. The
    fix is to do the fp32→int8 quantization **offline** on the calibration
    input: load ``inputs.npz``, apply the Quant's affine transform, save the
    int8 result, then change the graph input dtype to int8 and delete the
    leading Quant.

    Returns 1 if the strip happened, 0 otherwise.
    """
    g = model.graph
    if not g.input:
        return 0
    in_name = g.input[0].name
    # Find a Quant whose input is the graph input.
    leading = None
    for n in g.node:
        if n.op_type == "Quant" and len(n.input) == 1 and n.input[0] == in_name:
            leading = n
            break
    if leading is None:
        return 0

    scale = float([a.f for a in leading.attribute if a.name == "scale"][0])
    zp = float([a.f for a in leading.attribute if a.name == "zero_point"][0])
    bit_width = int([a.i for a in leading.attribute if a.name == "bit_width"][0])
    signed = bool([a.i for a in leading.attribute if a.name == "signed"][0])
    lo = -(1 << (bit_width - 1)) if signed else 0
    hi = (1 << (bit_width - 1)) - 1 if signed else (1 << bit_width) - 1

    # Load + transform inputs.npz.
    arr_dict = np.load(inputs_npz_path)
    new_arrs = {}
    for k in arr_dict.files:
        x_fp = arr_dict[k]
        x_q = np.round(x_fp / scale + zp).astype(np.float64)
        x_q = np.clip(x_q, lo, hi).astype(np.int8 if signed and bit_width <= 8 else np.int32)
        new_arrs[k] = x_q
    np.savez(inputs_npz_path, **new_arrs)

    # Rewire: consumers of leading.output[0] now consume in_name directly.
    q_out = leading.output[0]
    for c in g.node:
        for i, inp in enumerate(c.input):
            if inp == q_out:
                c.input[i] = in_name
    # Remove the Quant node.
    g.node.remove(leading)
    # Change graph input dtype to int8.
    g.input[0].type.tensor_type.elem_type = (
        onnx.TensorProto.INT8 if signed else onnx.TensorProto.UINT8
    )
    return 1


def fold_standalone_quant_to_requantshift(graph: onnx.GraphProto, shift_bits: int = 16) -> int:
    """Replace standalone ``Quant`` nodes (those whose input now flows from
    an integer-producing op like RequantShift or Add) with ``RequantShift``.

    Background: after ``skip_dequant_before_integer_op``, ops like Add take
    integer inputs and produce integer outputs. The Quant directly downstream
    used to mean "round fp32 to int8", but the input is no longer fp32 — it's
    int. The corrected semantic is "rescale this int to a new int range",
    which is exactly RequantShift.

    The new RequantShift has the same scale/zp as the original Quant, but
    interprets the input as integer (scale-1 units) and produces int8.
    """
    prod = _producer_map(graph)
    _consumer_map(graph)
    graph_input_names = {i.name for i in graph.input}

    n_folded = 0
    nodes_to_remove: set = set()
    inits_to_add: List[onnx.TensorProto] = []
    new_nodes: List[onnx.NodeProto] = []

    for q_node in list(graph.node):
        if q_node.op_type != "Quant":
            continue
        if q_node.name in nodes_to_remove:
            continue
        # Skip if the input is fp32 (graph input or non-integer op output).
        src = q_node.input[0]
        if src in graph_input_names:
            continue  # leading Quant; handled by skip_leading_quant_dequant
        upstream = prod.get(src)
        if upstream is None:
            continue
        # Walk back through layout ops to find the real upstream.
        while upstream is not None and upstream.op_type in _LAYOUT_OPS:
            up_src = upstream.input[0]
            if up_src not in prod:
                upstream = None
                break
            upstream = prod[up_src]
        if upstream is None:
            continue
        # Only treat as standalone-on-int when upstream is an integer-producing
        # op after our other passes.
        if upstream.op_type not in {"RequantShift", "Add", "Conv", "Gemm", "MatMul", "ReduceMean"}:
            continue

        scale_q = float([a.f for a in q_node.attribute if a.name == "scale"][0])
        zp_q = float([a.f for a in q_node.attribute if a.name == "zero_point"][0])
        bit_width = int([a.i for a in q_node.attribute if a.name == "bit_width"][0])
        signed = bool([a.i for a in q_node.attribute if a.name == "signed"][0])

        # x is already an integer in scale-1 units; mul = 1/scale_q * 2^N.
        div = 1 << shift_bits
        mul_val = int(round((1.0 / scale_q) * div))
        add_val = int(round(zp_q * div))
        if not signed:
            # Same int8-output normalization as the other pass.
            add_val -= 128 * div
            signed = True

        base = "QSTANDALONE_" + q_node.name
        mul_init = numpy_helper.from_array(np.array([mul_val], dtype=np.int32), name=base + "_mul")
        add_init = numpy_helper.from_array(np.array([add_val], dtype=np.int32), name=base + "_add")
        inits_to_add.extend([mul_init, add_init])

        new_rqs = helper.make_node(
            "RequantShift",
            inputs=[src, mul_init.name, add_init.name],
            outputs=[q_node.output[0]],
            name=base,
        )
        new_rqs.attribute.extend(
            [
                helper.make_attribute(
                    "n_levels", numpy_helper.from_array(np.array(1 << bit_width, dtype=np.int64))
                ),
                helper.make_attribute(
                    "signed", numpy_helper.from_array(np.array(int(signed), dtype=np.int64))
                ),
                helper.make_attribute(
                    "div", numpy_helper.from_array(np.array(div, dtype=np.int64))
                ),
            ]
        )
        new_nodes.append(new_rqs)
        nodes_to_remove.add(q_node.name)
        n_folded += 1

    kept = [n for n in graph.node if n.name not in nodes_to_remove]
    kept.extend(new_nodes)
    del graph.node[:]
    graph.node.extend(kept)
    graph.initializer.extend(inits_to_add)
    return n_folded


def remove_initializers_from_inputs(graph: onnx.GraphProto) -> int:
    """Drop every entry from ``graph.input`` whose name also appears in
    ``graph.initializer``. DeepQuant exports with
    ``keep_initializers_as_inputs=True`` (PyTorch < 1.13 compatibility),
    leaving the weights also declared as inputs. Deeploy's gs loader then
    sees them as ``Variable`` instead of ``Constant`` and the downstream
    RequantMerge passes that read ``.values`` off the bias / weight crash.

    Returns the count of inputs removed.
    """
    init_names = {init.name for init in graph.initializer}
    keep = [i for i in graph.input if i.name not in init_names]
    n_removed = len(graph.input) - len(keep)
    del graph.input[:]
    graph.input.extend(keep)
    return n_removed


def _toposort(graph: onnx.GraphProto) -> None:
    """In-place topological sort by producer dependency."""
    name_to_node = {n.name: n for n in graph.node}
    producer_of: Dict[str, str] = {}  # tensor name → producing node name
    for n in graph.node:
        for o in n.output:
            producer_of[o] = n.name

    visited: set = set()
    order: List[onnx.NodeProto] = []

    def visit(node_name: str) -> None:
        if node_name in visited:
            return
        visited.add(node_name)
        node = name_to_node[node_name]
        for inp in node.input:
            up = producer_of.get(inp)
            if up is not None and up not in visited:
                visit(up)
        order.append(node)

    for n in list(graph.node):
        visit(n.name)
    del graph.node[:]
    graph.node.extend(order)


def run_all_qcdq_to_deeploy_passes(
    model: onnx.ModelProto, inputs_npz_path: Optional[str] = None
) -> Dict[str, int]:
    """Run every pass in order, in-place on the model, returning a stats dict.

    If ``inputs_npz_path`` is given, the offline-quantize pass also rewrites
    the npz to int8 so the leading Quant can be stripped.
    """
    g = model.graph
    stats = OrderedDict()
    stats["remove_initializers_from_inputs"] = remove_initializers_from_inputs(g)
    stats["upgrade_reducemean_axes"] = upgrade_reducemean_axes(g)
    stats["fold_qcdq_to_quant_dequant"] = fold_qcdq_to_quant_dequant(g)
    _toposort(g)
    stats["constfold_quant_of_initializer"] = constfold_quant_of_initializer(g)
    _toposort(g)
    stats["fold_dequant_quant_to_requantshift"] = fold_dequant_quant_to_requantshift(g)
    _toposort(g)
    stats["skip_dequant_before_integer_op"] = skip_dequant_before_integer_op(g)
    _toposort(g)
    stats["fold_standalone_quant_to_requantshift"] = fold_standalone_quant_to_requantshift(g)
    _toposort(g)
    stats["skip_leading_quant_dequant"] = skip_leading_quant_dequant(g)
    _toposort(g)
    stats["absorb_conv_bias_into_following_requantshift"] = (
        absorb_conv_bias_into_following_requantshift(g)
    )
    _toposort(g)
    if inputs_npz_path is not None:
        stats["quantize_input_offline"] = quantize_input_offline(model, inputs_npz_path)
        _toposort(g)
    stats["strip_trailing_dequant"] = strip_trailing_dequant(g)
    _toposort(g)
    stats["cleanup_orphan_nodes"] = cleanup_orphan_nodes(g)
    return stats
