#!/usr/bin/env python3
"""
qdq_to_deeploy.py

Convert ONNX QDQ/QCDQ graphs (QuantizeLinear/DequantizeLinear) into Deeploy-style
Quant/Dequant nodes, and optionally insert RequantShift nodes for known
scale transitions.

Dependencies:
  pip install onnx onnx-graphsurgeon numpy

Usage:
  python qdq_to_deeploy.py --in model_qdq.onnx --out model_deeploy.onnx
  python qdq_to_deeploy.py --in model_qdq.onnx --out model_deeploy.onnx --rqs rqs_map.json
"""

import argparse
import json
from typing import Any, Dict, Tuple, Optional

import numpy as np
import onnx
import onnx_graphsurgeon as gs


# ---------------------------
# Helpers: initializers / dtypes
# ---------------------------

def _as_numpy(t: gs.Constant) -> np.ndarray:
    return np.asarray(t.values)

def _const_scalar(name: str, value: Any, dtype=np.float32) -> gs.Constant:
    return gs.Constant(name=name, values=np.array(value, dtype=dtype))

def _get_initializer_value(graph: gs.Graph, name: str) -> Optional[np.ndarray]:
    """Return initializer values as numpy array if present."""
    for t in graph.initializers:
        if t.name == name:
            if isinstance(t, gs.Constant):
                return _as_numpy(t)
            # gs may store initializers as Constants; this is fallback
    # Also check graph.tensors() for Constant
    tensors = graph.tensors()
    if name in tensors and isinstance(tensors[name], gs.Constant):
        return _as_numpy(tensors[name])
    return None

def _infer_signed_from_zero_point(zp_arr: Optional[np.ndarray]) -> bool:
    """
    Deeploy Quant node expects 'signed' attribute. We infer it from zero_point dtype
    when possible (int8 -> signed, uint8 -> unsigned). If zp missing, default signed.
    """
    if zp_arr is None:
        return True
    dt = zp_arr.dtype
    if dt == np.int8 or dt == np.int32 or dt == np.int16:
        return True
    if dt == np.uint8 or dt == np.uint16:
        return False
    # fallback
    return True


# ---------------------------
# RequantShift parameterization
# ---------------------------

def float_to_rqs_params(scale_ratio: np.ndarray, max_shift: int = 30) -> Tuple[np.ndarray, int, int]:
    """
    Convert scale_ratio to (mul, add, div_pow2) such that:
      out ≈ ((in * mul) + add) >> shift     where div = 2^shift

    Deeploy RQS formula uses:
      out = clip(((in * mul) + add) >> log2(div), ...)  [2](https://deepwiki.com/pulp-platform/Deeploy/8.1-quantization-and-training-support)

    We pick a single shift (power-of-two div), and per-element mul if scale_ratio is vector.
    add implements rounding-to-nearest: add = 1<<(shift-1).
    """
    # Ensure array
    r = np.asarray(scale_ratio, dtype=np.float64)

    # Choose largest shift that keeps mul within int32 range.
    # mul = round(r * 2^shift)
    # For safety use signed int32.
    best_shift = None
    for shift in range(max_shift, -1, -1):
        mul = np.round(r * (2.0 ** shift))
        if np.all(np.abs(mul) <= (2**31 - 1)):
            best_shift = shift
            break
    if best_shift is None:
        raise RuntimeError("Could not find valid shift for scale_ratio")

    mul = np.round(r * (2.0 ** best_shift)).astype(np.int32)
    add = (1 << (best_shift - 1)) if best_shift > 0 else 0
    div = 1 << best_shift
    return mul, add, div


# ---------------------------
# Core transforms: QDQ -> Deeploy
# ---------------------------

def replace_qdq_with_deeploy(graph: gs.Graph) -> None:
    """
    Replace QuantizeLinear and DequantizeLinear nodes with Deeploy Quant and Dequant nodes.

    Deeploy Quant op parameters: scale, zero_point, bit_width, signed. [2](https://deepwiki.com/pulp-platform/Deeploy/8.1-quantization-and-training-support)
    Deeploy Dequant op parameters: scale, zero_point; supports int8/int32 inputs. [2](https://deepwiki.com/pulp-platform/Deeploy/8.1-quantization-and-training-support)
    """
    new_nodes = []
    for node in graph.nodes:
        if node.op == "QuantizeLinear":
            # Inputs: x, y_scale, y_zero_point (zp optional per ONNX)
            x = node.inputs[0]
            scale_in = node.inputs[1] if len(node.inputs) > 1 else None
            zp_in = node.inputs[2] if len(node.inputs) > 2 else None

            scale = _as_numpy(scale_in) if isinstance(scale_in, gs.Constant) else _get_initializer_value(graph, scale_in.name)
            zp = None
            if zp_in is not None:
                zp = _as_numpy(zp_in) if isinstance(zp_in, gs.Constant) else _get_initializer_value(graph, zp_in.name)

            if scale is None:
                raise RuntimeError(f"QuantizeLinear node {node.name} has non-constant scale; provide constant initializer.")

            signed = _infer_signed_from_zero_point(zp)
            bit_width = 8  # Deeploy QuantParser expects bit_width attribute; typical is 8. [2](https://deepwiki.com/pulp-platform/Deeploy/8.1-quantization-and-training-support)

            # Create Deeploy Quant node
            q = gs.Node(
                op="Quant",
                name=(node.name or "Quant") + "_Deeploy",
                inputs=[x],
                outputs=node.outputs,
                attrs={
                    "scale": float(scale.reshape(-1)[0]) if scale.size == 1 else scale.astype(np.float32),
                    "zero_point": int(zp.reshape(-1)[0]) if (zp is not None and zp.size == 1) else (zp.astype(np.int32) if zp is not None else 0),
                    "bit_width": bit_width,
                    "signed": int(signed),
                }
            )
            new_nodes.append(q)
            continue

        if node.op == "DequantizeLinear":
            # Inputs: x, x_scale, x_zero_point (zp optional)
            xq = node.inputs[0]
            scale_in = node.inputs[1] if len(node.inputs) > 1 else None
            zp_in = node.inputs[2] if len(node.inputs) > 2 else None

            scale = _as_numpy(scale_in) if isinstance(scale_in, gs.Constant) else _get_initializer_value(graph, scale_in.name)
            zp = None
            if zp_in is not None:
                zp = _as_numpy(zp_in) if isinstance(zp_in, gs.Constant) else _get_initializer_value(graph, zp_in.name)

            if scale is None:
                raise RuntimeError(f"DequantizeLinear node {node.name} has non-constant scale; provide constant initializer.")

            dq = gs.Node(
                op="Dequant",
                name=(node.name or "Dequant") + "_Deeploy",
                inputs=[xq],
                outputs=node.outputs,
                attrs={
                    "scale": float(scale.reshape(-1)[0]) if scale.size == 1 else scale.astype(np.float32),
                    "zero_point": int(zp.reshape(-1)[0]) if (zp is not None and zp.size == 1) else (zp.astype(np.int32) if zp is not None else 0),
                }
            )
            new_nodes.append(dq)
            continue

        # passthrough
        new_nodes.append(node)

    graph.nodes = new_nodes


def insert_rqs_from_map(graph: gs.Graph, rqs_map: Dict[str, Any]) -> None:
    """
    Insert Deeploy RequantShift nodes according to a user-supplied mapping.

    Deeploy RequantShift formula: out = clip(((in * mul) + add) >> log2(div), ...) [2](https://deepwiki.com/pulp-platform/Deeploy/8.1-quantization-and-training-support)

    rqs_map format:
    {
      "edges": [
        {
          "src_tensor": "tensorA_q",
          "dst_tensor": "tensorB_q",
          "src_scale": 0.0078125,
          "dst_scale": 0.00390625
        }
      ]
    }

    This will insert RequantShift between producer(src_tensor) and consumers(dst_tensor).
    """
    edges = rqs_map.get("edges", [])
    if not edges:
        return

    tensor_dict = graph.tensors()

    for e in edges:
        src = e["src_tensor"]
        dst = e["dst_tensor"]
        s_src = np.array(e["src_scale"], dtype=np.float64)
        s_dst = np.array(e["dst_scale"], dtype=np.float64)

        if src not in tensor_dict:
            raise KeyError(f"src_tensor '{src}' not found in graph tensors.")
        if dst not in tensor_dict:
            raise KeyError(f"dst_tensor '{dst}' not found in graph tensors.")

        src_tensor = tensor_dict[src]
        dst_tensor = tensor_dict[dst]

        # Compute ratio and integer params
        ratio = s_src / s_dst
        mul, add, div = float_to_rqs_params(ratio)

        # Create constants for mul/add/div
        mul_const = gs.Constant(name=f"{src}_to_{dst}_mul", values=np.array(mul, dtype=np.int32))
        add_const = gs.Constant(name=f"{src}_to_{dst}_add", values=np.array(add, dtype=np.int32))
        div_const = gs.Constant(name=f"{src}_to_{dst}_div", values=np.array(div, dtype=np.int32))

        # New intermediate tensor
        rqs_out = gs.Variable(name=f"{src}_rqs_to_{dst}", dtype=src_tensor.dtype, shape=src_tensor.shape)

        rqs_node = gs.Node(
            op="RequantShift",
            name=f"RQS_{src}_to_{dst}",
            inputs=[src_tensor, mul_const, add_const, div_const],
            outputs=[rqs_out],
            attrs={}
        )

        # Rewire: every consumer of src_tensor that was expecting dst_tensor gets rqs_out
        # Safer approach: replace uses of src_tensor in dst's producer chain is complex;
        # instead, we replace occurrences of src_tensor in inputs of nodes where it appears as dst_tensor input.
        for n in graph.nodes:
            for i, inp in enumerate(n.inputs):
                if inp is dst_tensor:
                    n.inputs[i] = rqs_out

        graph.nodes.append(rqs_node)

    graph.cleanup().toposort()


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input ONNX (Brevitas QDQ/QCDQ)")
    ap.add_argument("--out", dest="out", required=True, help="Output ONNX (Deeploy Quant/Dequant/RQS)")
    ap.add_argument("--rqs", dest="rqs", default=None, help="Optional JSON map to insert RequantShift nodes")
    args = ap.parse_args()

    model = onnx.load(args.inp)
    graph = gs.import_onnx(model)

    replace_qdq_with_deeploy(graph)

    if args.rqs is not None:
        with open(args.rqs, "r") as f:
            rqs_map = json.load(f)
        insert_rqs_from_map(graph, rqs_map)

    graph.cleanup().toposort()
    out_model = gs.export_onnx(graph)
    onnx.save(out_model, args.out)
    print(f"[OK] Wrote Deeploy-style ONNX to: {args.out}")

if __name__ == "__main__":
    main()