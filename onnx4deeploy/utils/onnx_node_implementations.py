# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Pure-Python ONNX graph executor.

Executes ONNX graphs without relying on onnxruntime.InferenceSession, so that
graphs containing custom ops (Deeploy Quant/Dequant/RequantShift and MeZO
perturbation ops) can be run natively.

Supported op domains:
  - ""  (standard ONNX ops)
  - "ai.onnx.contrib"  → Quant, Dequant, RequantShift
  - "mezo"             → PerturbNormal, PerturbUniform, PerturbRademacher,
                         PerturbTriangle, PerturbEggroll,
                         RQSPerturbRademacher, RQSPerturbUniform
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

# ---------------------------------------------------------------------------
# RNG reference matching Deeploy C++ implementation
# ---------------------------------------------------------------------------

NUM_CORES: int = 8


def _scramble(seed: int) -> np.uint32:
    """seed * 1664525 + 1013904223  (mod 2**32, 32-bit LCG scramble)."""
    return np.uint32(np.uint32(seed) * np.uint32(1664525) + np.uint32(1013904223))


def _xorshift32(state: np.uint32) -> np.uint32:
    """One Xorshift32 step."""
    state = np.uint32(state)
    state ^= np.uint32(state << np.uint32(13))
    state ^= np.uint32(state >> np.uint32(17))
    state ^= np.uint32(state << np.uint32(5))
    return state


def _uint32_to_signed_float(u: np.uint32) -> float:
    """Map uint32 uniformly to (-1, 1)."""
    return float(u) / float(np.iinfo(np.uint32).max) * 2.0 - 1.0


def _perturb_uniform(
    data: np.ndarray,
    global_seed: int,
    node_id: int,
    eps: float,
    sign: int = 1,
) -> np.ndarray:
    """
    Uniform perturbation matching Deeploy PerturbUniform kernel.

    seed per core = scramble(global_seed + NUM_CORES*node_id + core_id)
    rand in (-1, 1) scaled by eps  (the ONNX node attribute already encodes
    the full scale, e.g. epsilon*2*sqrt(3)).
    """
    flat = data.flatten().astype(np.float32)
    size = flat.size
    log2core = int(math.log2(NUM_CORES))

    for core_id in range(NUM_CORES):
        chunk = (size >> log2core) + (1 if (size & (NUM_CORES - 1)) else 0)
        chunk_start = min(chunk * core_id, size)
        chunk_stop = min(chunk_start + chunk, size)

        seed = _scramble(global_seed + NUM_CORES * node_id + core_id)
        for i in range(chunk_start, chunk_stop):
            seed = _xorshift32(seed)
            flat[i] += np.float32(sign * (float(seed) / float(0xFFFFFFFF) - 0.5) * eps)

    return flat.reshape(data.shape)


def _perturb_triangle(
    data: np.ndarray,
    global_seed: int,
    node_id: int,
    eps: float,
    sign: int = 1,
) -> np.ndarray:
    """Triangle perturbation: (u1 - u2) * epsilon per element.

    Matches C kernel which draws two uniform values per element.
    """
    flat = data.flatten().astype(np.float32)
    size = flat.size
    log2core = int(math.log2(NUM_CORES))

    for core_id in range(NUM_CORES):
        chunk = (size >> log2core) + (1 if (size & (NUM_CORES - 1)) else 0)
        chunk_start = min(chunk * core_id, size)
        chunk_stop = min(chunk_start + chunk, size)

        rng_state = _scramble(global_seed + NUM_CORES * node_id + core_id)
        for i in range(chunk_start, chunk_stop):
            rng_state = _xorshift32(rng_state)
            u1 = float(rng_state) / float(0xFFFFFFFF)
            rng_state = _xorshift32(rng_state)
            u2 = float(rng_state) / float(0xFFFFFFFF)
            flat[i] += np.float32(sign * (u1 - u2) * eps)

    return flat.reshape(data.shape)


def _perturb_rademacher(
    data: np.ndarray,
    global_seed: int,
    node_id: int,
    eps: float,
    sign: int = 1,
) -> np.ndarray:
    """Rademacher perturbation: each element offset by ±eps.

    Matches the C kernel which extracts 32 Rademacher samples per xorshift call.
    """
    flat = data.flatten().astype(np.float32)
    size = flat.size
    log2core = int(math.log2(NUM_CORES))

    for core_id in range(NUM_CORES):
        chunk = (size >> log2core) + (1 if (size & (NUM_CORES - 1)) else 0)
        chunk_start = min(chunk * core_id, size)
        chunk_stop = min(chunk_start + chunk, size)
        local_size = chunk_stop - chunk_start

        rng_state = _scramble(global_seed + NUM_CORES * node_id + core_id)

        n_full_batches = local_size // 32
        leftover = local_size % 32
        i = chunk_start

        for _ in range(n_full_batches):
            rng_state = _xorshift32(rng_state)
            bits = int(rng_state)
            for _ in range(32):
                rad = np.float32(1.0) if (bits & 1) else np.float32(-1.0)
                flat[i] += np.float32(sign) * rad * np.float32(eps)
                bits >>= 1
                i += 1

        if leftover > 0:
            rng_state = _xorshift32(rng_state)
            bits = int(rng_state)
            for _ in range(leftover):
                rad = np.float32(1.0) if (bits & 1) else np.float32(-1.0)
                flat[i] += np.float32(sign) * rad * np.float32(eps)
                bits >>= 1
                i += 1

    return flat.reshape(data.shape)


def _perturb_normal(
    data: np.ndarray,
    global_seed: int,
    node_id: int,
    eps: float,
    sign: int = 1,
) -> np.ndarray:
    """Gaussian perturbation via Box-Muller on Xorshift32 draws.

    Matches the C kernel which processes elements in pairs (cos, sin)
    from each Box-Muller transform, consuming 2 xorshift calls per pair.
    """
    flat = data.flatten().astype(np.float32)
    size = flat.size
    log2core = int(math.log2(NUM_CORES))

    for core_id in range(NUM_CORES):
        chunk = (size >> log2core) + (1 if (size & (NUM_CORES - 1)) else 0)
        chunk_start = min(chunk * core_id, size)
        chunk_stop = min(chunk_start + chunk, size)
        local_size = chunk_stop - chunk_start

        rng_state = _scramble(global_seed + NUM_CORES * node_id + core_id)

        # Process pairs (matching C kernel's 2-at-a-time Box-Muller)
        n_pairs = local_size // 2
        i = chunk_start
        for _ in range(n_pairs):
            rng_state = _xorshift32(rng_state)
            u1 = max(float(rng_state) / float(0xFFFFFFFF), 1e-10)
            rng_state = _xorshift32(rng_state)
            u2 = float(rng_state) / float(0xFFFFFFFF)
            mag = math.sqrt(-2.0 * math.log(u1))
            angle = 2.0 * math.pi * u2
            flat[i]     += np.float32(sign * mag * math.cos(angle) * eps)
            flat[i + 1] += np.float32(sign * mag * math.sin(angle) * eps)
            i += 2

        # Remainder (odd element) — C kernel also does single-element Box-Muller
        if local_size % 2 != 0:
            rng_state = _xorshift32(rng_state)
            u1 = max(float(rng_state) / float(0xFFFFFFFF), 1e-10)
            rng_state = _xorshift32(rng_state)
            u2 = float(rng_state) / float(0xFFFFFFFF)
            mag = math.sqrt(-2.0 * math.log(u1))
            angle = 2.0 * math.pi * u2
            flat[i] += np.float32(sign * mag * math.cos(angle) * eps)

    return flat.reshape(data.shape)


def _perturb_rqs_rademacher(
    data: np.ndarray,
    mul: np.ndarray,
    global_seed: int,
    node_id: int,
    div: int,
    n_levels: int,
    signed: int,
    sign: int = 1,
) -> np.ndarray:
    """RQS Rademacher perturbation for integer-quantised tensors."""
    flat = data.astype(np.int64).flatten()
    size = flat.size
    log2core = int(math.log2(NUM_CORES))

    mul_flat = mul.flatten().astype(np.int64)
    num_out = mul_flat.size
    # Tile mul to match C kernel: M[i % channel_width]
    reps = (size + num_out - 1) // num_out if num_out > 0 else 1
    mul_per_elem = np.tile(mul_flat, reps)[:size]

    S = int(math.log2(div))
    rounding = np.int64(1 << (S - 1)) if S > 0 else np.int64(0)

    for core_id in range(NUM_CORES):
        chunk = (size >> log2core) + (1 if (size & (NUM_CORES - 1)) else 0)
        chunk_start = min(chunk * core_id, size)
        chunk_stop = min(chunk_start + chunk, size)
        local_size = chunk_stop - chunk_start

        seed = _scramble(global_seed + NUM_CORES * node_id + core_id)
        n_full = local_size // 32
        leftover = local_size % 32
        idx = chunk_start

        # Process full batches of 32 (1 xorshift per 32 elements)
        for _ in range(n_full):
            seed = _xorshift32(seed)
            bits = int(seed)
            for _ in range(32):
                rad = np.int64(1) if (bits & 1) else np.int64(-1)
                m_val = np.int64(mul_per_elem[idx])
                noise_q = (rad * m_val + rounding) >> S
                flat[idx] += sign * noise_q
                bits >>= 1
                idx += 1

        # Process leftover elements
        if leftover > 0:
            seed = _xorshift32(seed)
            bits = int(seed)
            for _ in range(leftover):
                rad = np.int64(1) if (bits & 1) else np.int64(-1)
                m_val = np.int64(mul_per_elem[idx])
                noise_q = (rad * m_val + rounding) >> S
                flat[idx] += sign * noise_q
                bits >>= 1
                idx += 1

    lo = -(n_levels // 2) + 1 if signed else 0
    hi = (n_levels // 2) - 1 if signed else n_levels - 1
    flat = np.clip(flat, lo, hi)
    return flat.reshape(data.shape).astype(data.dtype)


def _perturb_rqs_uniform(
    data: np.ndarray,
    mul: np.ndarray,
    global_seed: int,
    node_id: int,
    div: int,
    n_levels: int,
    signed: int,
    sign: int = 1,
) -> np.ndarray:
    """RQS Uniform perturbation for integer-quantised tensors."""
    flat = data.astype(np.int64).flatten()
    size = flat.size
    log2core = int(math.log2(NUM_CORES))

    mul_flat = mul.flatten().astype(np.int64)
    num_out = mul_flat.size
    # Tile mul to match C kernel: M[i % channel_width]
    reps = (size + num_out - 1) // num_out if num_out > 0 else 1
    mul_per_elem = np.tile(mul_flat, reps)[:size]

    for core_id in range(NUM_CORES):
        chunk = (size >> log2core) + (1 if (size & (NUM_CORES - 1)) else 0)
        chunk_start = min(chunk * core_id, size)
        chunk_stop = min(chunk_start + chunk, size)

        S = int(math.log2(div))
        rounding = np.int64(1 << (S - 1)) if S > 0 else np.int64(0)
        threshold = 255

        seed = _scramble(global_seed + NUM_CORES * node_id + core_id)
        n_full = (chunk_stop - chunk_start) // 4
        leftover_count = (chunk_stop - chunk_start) % 4
        idx = chunk_start

        # Process batches of 4 (rejection-sampled ternary noise)
        for _ in range(n_full):
            while True:
                seed = _xorshift32(seed)
                rw = int(seed)
                b0 = (rw >> 0) & 0xFF
                b1 = (rw >> 8) & 0xFF
                b2 = (rw >> 16) & 0xFF
                b3 = (rw >> 24) & 0xFF
                if b0 < threshold and b1 < threshold and b2 < threshold and b3 < threshold:
                    break
            n_int = [(b0 % 3) - 1, (b1 % 3) - 1, (b2 % 3) - 1, (b3 % 3) - 1]
            for j in range(4):
                m_val = np.int64(mul_per_elem[idx + j])
                noise_q = (np.int64(n_int[j]) * m_val + rounding) >> S
                flat[idx + j] += sign * noise_q
            idx += 4

        # Leftover
        for _ in range(leftover_count):
            while True:
                seed = _xorshift32(seed)
                u = int(seed) & 0xFF
                if u < threshold:
                    break
            n_int_val = (u % 3) - 1
            m_val = np.int64(mul_per_elem[idx])
            noise_q = (np.int64(n_int_val) * m_val + rounding) >> S
            flat[idx] += sign * noise_q
            idx += 1

    lo = -(n_levels // 2) + 1 if signed else 0
    hi = (n_levels // 2) - 1 if signed else n_levels - 1
    flat = np.clip(flat, lo, hi)
    return flat.reshape(data.shape).astype(data.dtype)


# ---------------------------------------------------------------------------
# Attribute extraction helper
# ---------------------------------------------------------------------------

def _node_attrs(node: onnx.NodeProto) -> Dict[str, Any]:
    """Return node attributes as a plain Python dict."""
    attrs: Dict[str, Any] = {}
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.FLOAT:
            attrs[attr.name] = attr.f
        elif attr.type == onnx.AttributeProto.INT:
            attrs[attr.name] = attr.i
        elif attr.type == onnx.AttributeProto.STRING:
            attrs[attr.name] = attr.s.decode("utf-8") if attr.s else ""
        elif attr.type == onnx.AttributeProto.TENSOR:
            attrs[attr.name] = numpy_helper.to_array(attr.t)
        elif attr.type == onnx.AttributeProto.INTS:
            attrs[attr.name] = list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOATS:
            attrs[attr.name] = list(attr.floats)
        else:
            attrs[attr.name] = attr
    return attrs


# ---------------------------------------------------------------------------
# Standard ONNX op dispatcher
# ---------------------------------------------------------------------------

def _exec_standard(op: str, inputs: List, attrs: Dict[str, Any]) -> List[np.ndarray]:  # noqa: C901
    if op == "Add":
        return [inputs[0] + inputs[1]]
    if op == "Sub":
        return [inputs[0] - inputs[1]]
    if op == "Mul":
        return [inputs[0] * inputs[1]]
    if op == "Div":
        return [inputs[0] / inputs[1]]
    if op == "Neg":
        return [-inputs[0]]
    if op == "Abs":
        return [np.abs(inputs[0])]
    if op == "Sqrt":
        return [np.sqrt(inputs[0])]
    if op == "Exp":
        return [np.exp(inputs[0])]
    if op == "Log":
        return [np.log(inputs[0])]
    if op == "Pow":
        return [np.power(inputs[0], inputs[1])]
    if op == "Erf":
        return [np.vectorize(math.erf)(inputs[0]).astype(inputs[0].dtype)]
    if op == "Ceil":
        return [np.ceil(inputs[0])]
    if op == "Floor":
        return [np.floor(inputs[0])]
    if op == "Round":
        return [np.round(inputs[0])]
    if op == "Sign":
        return [np.sign(inputs[0]).astype(inputs[0].dtype)]

    if op == "Clip":
        lo = inputs[1] if len(inputs) > 1 and inputs[1] is not None else attrs.get("min", None)
        hi = inputs[2] if len(inputs) > 2 and inputs[2] is not None else attrs.get("max", None)
        return [np.clip(inputs[0], lo, hi)]

    if op in ("ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin"):
        x = inputs[0]
        axes = attrs.get("axes", None)
        keepdims = bool(attrs.get("keepdims", 1))
        noop_empty = bool(attrs.get("noop_with_empty_axes", 0))
        if len(inputs) > 1 and inputs[1] is not None:
            axes = tuple(int(a) for a in inputs[1].flatten())
        elif axes is not None:
            axes = tuple(axes)
        if noop_empty and (axes is None or len(axes) == 0):
            return [x]
        fn = {"ReduceSum": np.sum, "ReduceMean": np.mean,
              "ReduceMax": np.max, "ReduceMin": np.min}[op]
        return [fn(x, axis=axes, keepdims=keepdims)]

    if op == "Relu":
        return [np.maximum(inputs[0], 0)]
    if op == "Sigmoid":
        return [1.0 / (1.0 + np.exp(-inputs[0].astype(np.float64))).astype(np.float32)]
    if op == "Tanh":
        return [np.tanh(inputs[0])]
    if op == "LeakyRelu":
        alpha = float(attrs.get("alpha", 0.01))
        x = inputs[0]
        return [np.where(x >= 0, x, alpha * x).astype(x.dtype)]
    if op in ("Gelu", "FastGelu"):
        x = inputs[0]
        return [(x * 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2)))).astype(x.dtype)]
    if op == "Softmax":
        axis = int(attrs.get("axis", -1))
        x = inputs[0]
        x_max = np.max(x, axis=axis, keepdims=True)
        ex = np.exp(x - x_max)
        return [(ex / np.sum(ex, axis=axis, keepdims=True)).astype(x.dtype)]

    if op == "MatMul":
        return [np.matmul(inputs[0], inputs[1])]

    if op == "Gemm":
        A, B = inputs[0], inputs[1]
        C = inputs[2] if len(inputs) > 2 and inputs[2] is not None else 0.0
        alpha = float(attrs.get("alpha", 1.0))
        beta = float(attrs.get("beta", 1.0))
        if int(attrs.get("transA", 0)):
            A = A.T
        if int(attrs.get("transB", 0)):
            B = B.T
        if np.issubdtype(A.dtype, np.integer):
            # Integer path: accumulate in int32 (matches C kernel behavior)
            result = np.matmul(A.astype(np.int32), B.astype(np.int32))
            if not isinstance(C, float):
                result = result + C.astype(np.int32)
            return [result]
        else:
            return [(alpha * np.matmul(A, B) + beta * C).astype(A.dtype)]

    if op == "Conv":
        x, w = inputs[0], inputs[1]
        b = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
        groups = int(attrs.get("group", 1))
        # Promote 1-D conv (NCL) to 2-D (NC1L) so the same kernel handles both.
        squeezed = x.ndim == 3
        if squeezed:
            x = x[:, :, np.newaxis, :]   # (N, C, 1, L)
            w = w[:, :, np.newaxis, :]   # (C_out, C_in_pg, 1, kL)
            raw_pads = list(attrs.get("pads", [0, 0]))
            raw_strides = list(attrs.get("strides", [1]))
            raw_dils = list(attrs.get("dilations", [1]))
            attrs = dict(attrs,
                         pads=[0, raw_pads[0], 0, raw_pads[1]],
                         strides=[1, raw_strides[0]],
                         dilations=[1, raw_dils[0]])
        dilations = list(attrs.get("dilations", [1, 1]))
        strides = list(attrs.get("strides", [1, 1]))
        pads = list(attrs.get("pads", [0, 0, 0, 0]))
        N, C_in, H, W = x.shape
        C_out, C_in_pg, kH, kW = w.shape
        sH, sW = strides[0], strides[1]
        dH, dW = dilations[0], dilations[1]
        pH_t, pW_l = pads[0], pads[1]
        pH_b, pW_r = pads[2], pads[3]
        H_out = (H + pH_t + pH_b - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + pW_l + pW_r - dW * (kW - 1) - 1) // sW + 1
        # Use int32 accumulation for integer inputs, float32 for float
        if np.issubdtype(x.dtype, np.integer):
            acc_dtype = np.int32
        else:
            acc_dtype = np.float32
        x_pad = np.pad(x, ((0,0),(0,0),(pH_t,pH_b),(pW_l,pW_r)))
        out = np.zeros((N, C_out, H_out, W_out), dtype=acc_dtype)
        cpp = C_out // groups
        for g in range(groups):
            xi = x_pad[:, g*C_in_pg:(g+1)*C_in_pg].astype(acc_dtype)
            wg = w[g*cpp:(g+1)*cpp].astype(acc_dtype)
            for oc in range(cpp):
                for oh in range(H_out):
                    for ow in range(W_out):
                        patch = xi[:, :,
                                   oh*sH:oh*sH+kH*dH:dH,
                                   ow*sW:ow*sW+kW*dW:dW]
                        out[:, g*cpp+oc, oh, ow] = np.sum(
                            patch * wg[oc], axis=(1, 2, 3))
        if b is not None:
            out += b[np.newaxis, :, np.newaxis, np.newaxis].astype(acc_dtype)
        if squeezed:
            out = out[:, :, 0, :]   # (N, C_out, 1, L_out) → (N, C_out, L_out)
        return [out]

    if op == "BatchNormalization":
        x, scale, bias, mean, var = inputs[:5]
        eps = float(attrs.get("epsilon", 1e-5))
        return [(scale * (x - mean) / np.sqrt(var + eps) + bias).astype(x.dtype)]

    if op == "LayerNormalization":
        x = inputs[0]
        scale = inputs[1] if len(inputs) > 1 and inputs[1] is not None else np.ones(1, dtype=x.dtype)
        b = inputs[2] if len(inputs) > 2 and inputs[2] is not None else np.zeros(1, dtype=x.dtype)
        axis = int(attrs.get("axis", -1))
        eps = float(attrs.get("epsilon", 1e-5))
        mean = np.mean(x, axis=axis, keepdims=True)
        var = np.var(x, axis=axis, keepdims=True)
        out = ((x - mean) / np.sqrt(var + eps)) * scale + b
        return [out.astype(x.dtype)]

    if op == "GroupNormalization":
        x, scale, bias = inputs[:3]
        num_groups = int(attrs.get("num_groups", 1))
        eps = float(attrs.get("epsilon", 1e-5))
        N, C = x.shape[:2]
        spatial = x.shape[2:]
        xr = x.reshape(N, num_groups, C // num_groups, *spatial)
        axes = tuple(range(2, xr.ndim))
        mean = np.mean(xr, axis=axes, keepdims=True)
        var = np.var(xr, axis=axes, keepdims=True)
        xn = ((xr - mean) / np.sqrt(var + eps)).reshape(N, C, *spatial)
        return [(xn * scale.reshape(1, C, *([1]*len(spatial))) +
                 bias.reshape(1, C, *([1]*len(spatial)))).astype(x.dtype)]

    if op == "MaxPool":
        x = inputs[0]
        kernel = list(attrs.get("kernel_shape", [2, 2]))
        strides = list(attrs.get("strides", [1, 1]))
        pads = list(attrs.get("pads", [0, 0, 0, 0]))
        N, C, H, W = x.shape
        kH, kW = kernel[0], kernel[1]
        sH, sW = strides[0], strides[1]
        pH_t, pW_l, pH_b, pW_r = pads[0], pads[1], pads[2], pads[3]
        fill = np.finfo(x.dtype).min if np.issubdtype(x.dtype, np.floating) else np.iinfo(x.dtype).min
        xp = np.pad(x, ((0,0),(0,0),(pH_t,pH_b),(pW_l,pW_r)), constant_values=fill)
        H_out = (H + pH_t + pH_b - kH) // sH + 1
        W_out = (W + pW_l + pW_r - kW) // sW + 1
        out = np.stack([[
            np.max(xp[:, :, oh*sH:oh*sH+kH, ow*sW:ow*sW+kW], axis=(2, 3))
            for ow in range(W_out)] for oh in range(H_out)], axis=0)
        return [out.transpose(2, 3, 0, 1)]  # (N,C,H_out,W_out)

    if op == "AveragePool":
        x = inputs[0]
        kernel = list(attrs.get("kernel_shape", [2, 2]))
        strides = list(attrs.get("strides", [1, 1]))
        pads = list(attrs.get("pads", [0, 0, 0, 0]))
        N, C, H, W = x.shape
        kH, kW = kernel[0], kernel[1]
        sH, sW = strides[0], strides[1]
        pH_t, pW_l, pH_b, pW_r = pads[0], pads[1], pads[2], pads[3]
        xp = np.pad(x, ((0,0),(0,0),(pH_t,pH_b),(pW_l,pW_r)))
        H_out = (H + pH_t + pH_b - kH) // sH + 1
        W_out = (W + pW_l + pW_r - kW) // sW + 1
        out = np.stack([[
            np.mean(xp[:, :, oh*sH:oh*sH+kH, ow*sW:ow*sW+kW], axis=(2, 3))
            for ow in range(W_out)] for oh in range(H_out)], axis=0)
        return [out.transpose(2, 3, 0, 1)]

    if op == "GlobalAveragePool":
        x = inputs[0]
        return [np.mean(x, axis=tuple(range(2, x.ndim)), keepdims=True)]
    if op == "GlobalMaxPool":
        x = inputs[0]
        return [np.max(x, axis=tuple(range(2, x.ndim)), keepdims=True)]

    if op == "Reshape":
        shape = [int(d) for d in inputs[1].flatten()]
        src = inputs[0].shape
        new_shape = [int(src[i]) if d == 0 else d for i, d in enumerate(shape)]
        return [inputs[0].reshape(new_shape)]
    if op == "Flatten":
        axis = int(attrs.get("axis", 1))
        x = inputs[0]
        pre = int(np.prod(x.shape[:axis])) if axis > 0 else 1
        post = int(np.prod(x.shape[axis:]))
        return [x.reshape(pre, post)]
    if op == "Transpose":
        perm = attrs.get("perm", None)
        return [np.transpose(inputs[0], axes=perm)]

    if op == "Squeeze":
        x = inputs[0]
        if len(inputs) > 1 and inputs[1] is not None:
            axes = tuple(int(a) for a in inputs[1].flatten())
        else:
            raw = attrs.get("axes", None)
            if raw is None:
                axes = None
            elif isinstance(raw, (int, np.integer)):
                axes = (int(raw),)
            else:
                axes = tuple(int(a) for a in raw) or None
        return [np.squeeze(x, axis=axes)]
    if op == "Unsqueeze":
        x = inputs[0]
        if len(inputs) > 1 and inputs[1] is not None:
            axes = sorted(int(a) for a in inputs[1].flatten())
        else:
            raw = attrs.get("axes", [])
            axes = sorted([raw] if isinstance(raw, int) else raw)
        for ax in axes:
            x = np.expand_dims(x, axis=ax)
        return [x]
    if op == "Expand":
        return [np.broadcast_to(inputs[0], inputs[1].tolist()).copy()]
    if op == "Concat":
        axis = int(attrs.get("axis", 0))
        valid = [t for t in inputs if t is not None]
        return [np.concatenate(valid, axis=axis)]
    if op == "Split":
        axis = int(attrs.get("axis", 0))
        split = list(attrs.get("split", []))
        if len(inputs) > 1 and inputs[1] is not None:
            split = inputs[1].flatten().tolist()
        if not split:
            num_out = int(attrs.get("num_outputs", 2))
            split = [inputs[0].shape[axis] // num_out] * num_out
        secs = np.cumsum([int(s) for s in split[:-1]]).tolist()
        return list(np.split(inputs[0], secs, axis=axis))
    if op == "Slice":
        data, starts, ends = inputs[0], inputs[1].flatten(), inputs[2].flatten()
        axes = inputs[3].flatten() if len(inputs) > 3 and inputs[3] is not None else np.arange(len(starts))
        steps = inputs[4].flatten() if len(inputs) > 4 and inputs[4] is not None else np.ones(len(starts), dtype=np.int64)
        idx = [slice(None)] * data.ndim
        for ax, s, e, st in zip(axes, starts, ends, steps):
            idx[int(ax)] = slice(int(s), int(e), int(st))
        return [data[tuple(idx)]]
    if op == "Gather":
        axis = int(attrs.get("axis", 0))
        return [np.take(inputs[0], inputs[1].astype(np.int64), axis=axis)]
    if op == "GatherElements":
        axis = int(attrs.get("axis", 0))
        return [np.take_along_axis(inputs[0], inputs[1].astype(np.int64), axis=axis)]
    if op == "Shape":
        start = int(attrs.get("start", 0))
        end = attrs.get("end", None)
        sh = np.array(inputs[0].shape, dtype=np.int64)
        return [sh[start:end]]
    if op == "Size":
        return [np.array(inputs[0].size, dtype=np.int64)]
    if op == "Cast":
        to_type = int(attrs.get("to", TensorProto.FLOAT))
        _DT = {
            TensorProto.FLOAT: np.float32,
            TensorProto.DOUBLE: np.float64,
            TensorProto.INT32: np.int32,
            TensorProto.INT64: np.int64,
            TensorProto.INT8: np.int8,
            TensorProto.UINT8: np.uint8,
            TensorProto.BOOL: bool,
            TensorProto.FLOAT16: np.float16,
        }
        return [inputs[0].astype(_DT.get(to_type, np.float32))]
    if op == "Identity":
        return [inputs[0].copy()]
    if op == "Pad":
        x = inputs[0]
        pads_arr = inputs[1].flatten().tolist() if len(inputs) > 1 and inputs[1] is not None else list(attrs.get("pads", []))
        val = float(inputs[2].flat[0]) if len(inputs) > 2 and inputs[2] is not None else float(attrs.get("value", 0.0))
        n = x.ndim
        pw = [(int(pads_arr[i]), int(pads_arr[i+n])) for i in range(n)]
        return [np.pad(x, pw, constant_values=val)]
    if op == "Tile":
        return [np.tile(inputs[0], inputs[1].flatten().tolist())]
    if op == "Where":
        return [np.where(inputs[0], inputs[1], inputs[2])]
    if op in ("Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual"):
        fn = {"Equal": np.equal, "Less": np.less, "Greater": np.greater,
              "LessOrEqual": np.less_equal, "GreaterOrEqual": np.greater_equal}[op]
        return [fn(inputs[0], inputs[1])]
    if op == "Not":
        return [~inputs[0]]
    if op == "Min":
        r = inputs[0]
        for t in inputs[1:]:
            r = np.minimum(r, t)
        return [r]
    if op == "Max":
        r = inputs[0]
        for t in inputs[1:]:
            r = np.maximum(r, t)
        return [r]
    if op == "Sum":
        r = inputs[0].copy()
        for t in inputs[1:]:
            r = r + t
        return [r]
    if op == "Mean":
        return [sum(inputs) / len(inputs)]
    if op == "Einsum":
        return [np.einsum(attrs["equation"], *inputs)]
    if op == "ArgMax":
        axis = int(attrs.get("axis", 0))
        keepdims = bool(attrs.get("keepdims", 1))
        return [np.argmax(inputs[0], axis=axis, keepdims=keepdims).astype(np.int64)]
    if op == "ArgMin":
        axis = int(attrs.get("axis", 0))
        keepdims = bool(attrs.get("keepdims", 1))
        return [np.argmin(inputs[0], axis=axis, keepdims=keepdims).astype(np.int64)]
    if op == "TopK":
        k = int(inputs[1].flat[0]) if len(inputs) > 1 else int(attrs.get("k", 1))
        axis = int(attrs.get("axis", -1))
        largest = bool(attrs.get("largest", 1))
        idx = np.argsort(inputs[0], axis=axis)
        if largest:
            idx = np.flip(idx, axis=axis)
        idx = np.take(idx, np.arange(k), axis=axis)
        vals = np.take_along_axis(inputs[0], idx, axis=axis)
        return [vals, idx.astype(np.int64)]
    if op == "ConstantOfShape":
        shape = inputs[0].flatten().tolist()
        val_t = attrs.get("value", None)
        fill = float(val_t.flat[0]) if val_t is not None else 0.0
        return [np.full([int(d) for d in shape], fill, dtype=np.float32)]
    if op == "Constant":
        val_t = attrs.get("value", None)
        if val_t is not None:
            return [val_t.copy() if isinstance(val_t, np.ndarray) else np.array(val_t)]
        vf = attrs.get("value_float", None)
        if vf is not None:
            return [np.array(vf, dtype=np.float32)]
        vi = attrs.get("value_int", None)
        if vi is not None:
            return [np.array(vi, dtype=np.int64)]
        raise ValueError("Constant node has no supported value attribute")
    if op == "Range":
        s, lim, d = inputs
        return [np.arange(float(s), float(lim), float(d)).astype(s.dtype)]
    if op == "NonZero":
        return [np.array(np.nonzero(inputs[0]), dtype=np.int64)]
    if op == "Dropout":
        return [inputs[0].copy(), np.ones_like(inputs[0], dtype=bool)]
    if op == "SoftmaxCrossEntropyLoss":
        logits = inputs[0]
        labels = inputs[1]
        xm = np.max(logits, axis=-1, keepdims=True)
        lp = logits - xm - np.log(np.sum(np.exp(logits - xm), axis=-1, keepdims=True))
        if labels.dtype in (np.int32, np.int64):
            nll = -lp[np.arange(logits.shape[0]), labels.flatten()]
        else:
            nll = -np.sum(lp * labels, axis=-1)
        red = attrs.get("reduction", "mean")
        loss = np.mean(nll) if red == "mean" else (np.sum(nll) if red == "sum" else nll)
        # Return log_prob as first output (matches Deeploy C code which outputs log-softmax)
        return [lp.astype(np.float32), np.array(loss, dtype=np.float32)]

    raise NotImplementedError(f"Standard ONNX op '{op}' is not implemented in run_onnx_graph")


# ---------------------------------------------------------------------------
# Deeploy custom ops  (ai.onnx.contrib)
# ---------------------------------------------------------------------------

def _exec_deeploy(op: str, inputs: List, attrs: Dict[str, Any], add_is_initializer: bool = True) -> List[np.ndarray]:
    if op == "Quant":
        x = inputs[0].astype(np.float64)
        scale = inputs[1].astype(np.float64)
        zp = inputs[2].astype(np.float64)
        bits = int(attrs.get("bits", 8))
        signed = bool(attrs.get("signed", True))
        qmin = -(2 ** (bits-1)) if signed else 0
        qmax = (2 ** (bits-1)) - 1 if signed else (2**bits) - 1
        return [np.clip(np.round(x / scale) + zp, qmin, qmax).astype(np.int8 if signed else np.uint8)]

    if op == "Dequant":
        q = inputs[0].astype(np.float64)
        scale = inputs[1].astype(np.float64)
        zp = inputs[2].astype(np.float64)
        return [((q - zp) * scale).astype(np.float32)]

    if op == "RequantShift":
        x = inputs[0].astype(np.int64)
        mul = inputs[1].astype(np.int64)
        add = inputs[2].astype(np.int64)
        div_attr = attrs.get("div", None)
        if div_attr is None:
            raise ValueError("RequantShift missing 'div' tensor attribute")
        div = int(div_attr.flat[0]) if hasattr(div_attr, "flat") else int(div_attr)
        log2D = int(np.round(np.log2(div)))
        bits = int(attrs.get("out_bits", 8))
        signed = bool(attrs.get("signed", 1))
        n_levels_attr = attrs.get("n_levels", None)
        if n_levels_attr is not None:
            n_levels = int(n_levels_attr.flat[0]) if hasattr(n_levels_attr, "flat") else int(n_levels_attr)
        else:
            n_levels = 2**bits
        if signed:
            qmin = -(n_levels // 2)
            qmax = (n_levels // 2) - 1
        else:
            qmin = 0
            qmax = n_levels - 1
        # mul/add may be per-channel vectors of shape (C,); reshape for N-D broadcast
        # NHWC: channel is last dim; NCHW: channel is dim 1
        if x.ndim > 1 and mul.ndim == 1:
            if mul.shape[0] == x.shape[-1]:
                # NHWC layout
                reshape = (1,) * (x.ndim - 1) + (-1,)
            elif mul.shape[0] == x.shape[1]:
                # NCHW layout
                reshape = (1, -1) + (1,) * (x.ndim - 2)
            else:
                reshape = None
            if reshape is not None:
                mul = mul.reshape(reshape)
                add = add.reshape(reshape)
        # Match C fused kernel (pulp_nn_bn_quant_i8) which uses int32 arithmetic:
        #   int32_t val = (k * phi) + lambda;   // int32 with wrapping overflow
        #   result = val >> d;
        # Only add rounding when add is a constant initializer.
        # For RQSRad, add comes from a perturbation node (not an initializer),
        # so the C merge pass cannot bake rounding in -> kernel just truncates.
        if add_is_initializer:
            rounding = np.int32(1 << (log2D - 1)) if log2D > 0 else np.int32(0)
        else:
            rounding = np.int32(0)
        x32 = x.astype(np.int32)
        mul32 = mul.astype(np.int32)
        add32 = add.astype(np.int32)
        rounding32 = np.int32(rounding)
        intermediate = x32 * mul32 + add32 + rounding32  # int32 wrapping overflow matches C
        result = intermediate >> np.int32(log2D)
        return [np.clip(result, qmin, qmax).astype(np.int8 if signed else np.uint8)]

    raise NotImplementedError(f"Deeploy op '{op}' not implemented")


# ---------------------------------------------------------------------------
# MeZO perturbation ops  (mezo / com.microsoft)
# ---------------------------------------------------------------------------

def _exec_mezo(op: str, inputs: List, attrs: Dict[str, Any]) -> List[np.ndarray]:
    seed = int(attrs.get("seed", 0))
    node_id = int(attrs.get("idx", 0))
    eps = float(attrs.get("eps", 0.01))
    sign = 1  # forward perturbation sign

    if op == "PerturbUniform":
        return [_perturb_uniform(inputs[0].astype(np.float32), seed, node_id, eps, sign)]
    if op == "PerturbRademacher":
        return [_perturb_rademacher(inputs[0].astype(np.float32), seed, node_id, eps, sign)]
    if op == "PerturbNormal":
        return [_perturb_normal(inputs[0].astype(np.float32), seed, node_id, eps, sign)]
    if op == "PerturbTriangle":
        return [_perturb_triangle(inputs[0].astype(np.float32), seed, node_id, eps, sign)]
    if op == "PerturbEggroll":
        # Generates a Rademacher column vector of the requested shape
        shape = [int(d) for d in inputs[0].flatten()]
        x = np.zeros(shape, dtype=np.float32)
        return [_perturb_rademacher(x, seed, node_id, 1.0, 1)]
    if op == "RQSPerturbRademacher":
        mul = inputs[1] if len(inputs) > 1 and inputs[1] is not None else np.ones(inputs[0].shape[0], dtype=np.int64)
        div = int(attrs.get("div", 2**15))
        n_levels = int(attrs.get("n_levels", 256))
        signed_flag = int(attrs.get("signed", 1))
        return [_perturb_rqs_rademacher(inputs[0], mul, seed, node_id, div, n_levels, signed_flag, sign)]
    if op == "RQSPerturbUniform":
        mul = inputs[1] if len(inputs) > 1 and inputs[1] is not None else np.ones(inputs[0].shape[0], dtype=np.int64)
        div = int(attrs.get("div", 2**15))
        n_levels = int(attrs.get("n_levels", 256))
        signed_flag = int(attrs.get("signed", 1))
        return [_perturb_rqs_uniform(inputs[0], mul, seed, node_id, div, n_levels, signed_flag, sign)]

    raise NotImplementedError(f"MeZO op '{op}' not implemented")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_onnx_graph(
    onnx_path: str,
    inputs: Dict[str, np.ndarray],
    output_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Execute an ONNX graph in pure Python — no onnxruntime required.

    Supports standard ONNX ops, Deeploy custom ops (ai.onnx.contrib domain:
    Quant, Dequant, RequantShift), and MeZO perturbation ops (mezo domain:
    PerturbUniform, PerturbRademacher, PerturbNormal, PerturbTriangle,
    PerturbEggroll, RQSPerturbRademacher, RQSPerturbUniform).

    Args:
        onnx_path:    Path to the ``.onnx`` model file.
        inputs:       ``{name: ndarray}`` for each graph input.
        output_names: If given, return those specific outputs.
                      Otherwise the first graph output is returned.

    Returns:
        First (or requested) graph output as a numpy array.
    """
    model = onnx.load(onnx_path)
    graph = model.graph

    # value store: name → numpy array
    values: Dict[str, np.ndarray] = {}
    values.update(inputs)

    _initializer_names = set()
    for init in graph.initializer:
        _initializer_names.add(init.name)
        if init.name not in values:
            values[init.name] = numpy_helper.to_array(init)

    # ONNX graph is topologically sorted by spec
    for node in graph.node:
        op = node.op_type
        domain = (node.domain or "").strip()
        attrs = _node_attrs(node)

        # Gather inputs; absent optional inputs (empty string) become None
        node_inputs: List = []
        for name in node.input:
            if name == "":
                node_inputs.append(None)
            elif name in values:
                node_inputs.append(values[name])
            else:
                raise KeyError(
                    f"Node '{node.name}' (op='{op}') needs input '{name}' "
                    f"which is missing from value map."
                )

        # com.microsoft is used for both ORT contrib ops (e.g. Gelu,
        # Attention) and MeZO perturbation ops. Route by op name.
        _MEZO_OPS = {
            "PerturbUniform", "PerturbRademacher", "PerturbNormal",
            "PerturbTriangle", "PerturbEggroll",
            "RQSPerturbRademacher", "RQSPerturbUniform",
        }

        try:
            if domain in ("", "ai.onnx"):
                outs = _exec_standard(op, node_inputs, attrs)
            elif domain == "ai.onnx.contrib":
                # For RequantShift, check if the 'add' input (3rd) is an initializer
                _add_is_init = True
                if op == "RequantShift" and len(node.input) >= 3:
                    _add_is_init = node.input[2] in _initializer_names
                outs = _exec_deeploy(op, node_inputs, attrs, add_is_initializer=_add_is_init)
            elif domain == "mezo" or (domain == "com.microsoft" and op in _MEZO_OPS):
                outs = _exec_mezo(op, node_inputs, attrs)
            elif domain == "com.microsoft":
                # ORT contrib ops that map to standard implementations
                outs = _exec_standard(op, node_inputs, attrs)
            else:
                raise NotImplementedError(
                    f"Unsupported domain '{domain}' for op '{op}'"
                )
        except NotImplementedError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Error in node '{node.name}' op='{op}' domain='{domain}': {exc}"
            ) from exc

        for out_name, out_val in zip(node.output, outs):
            if out_name:
                values[out_name] = out_val

    graph_outputs = [o.name for o in graph.output]
    if output_names is not None:
        return [values[n] for n in output_names]
    if graph_outputs:
        return values[graph_outputs[0]]
    raise RuntimeError("ONNX graph has no outputs")
