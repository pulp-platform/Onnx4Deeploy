# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""
Pure Python / NumPy / PyTorch implementations of ONNX nodes used in
Onnx4Deeploy exports, including custom Deeploy and MeZO operator domains.

Organised in three sections:
  1. RNG utilities  – Xorshift32 and Ziggurat, matching the C/hardware RNGs
                      used by the MeZO perturbation operators.
  2. Custom ops     – Deeploy (Quant, Dequant, RequantShift) and MeZO
                      (PerturbNormal, PerturbUniform, PerturbTriangle,
                       PerturbRademacher, PerturbEggroll,
                       RQSPerturbRademacher, RQSPerturbUniform).
  3. Standard ops   – Conv, MaxPool, Gemm, Relu, Flatten, Reshape, Clip,
                      QuantizeLinear, DequantizeLinear, and friends.
  4. Graph executor – ``run_onnx_graph`` dispatches every node in an ONNX
                      model to one of the implementations above and returns
                      the final graph output as a float32 NumPy array.

Usage::

    from tests.models.onnx_node_implementations import run_onnx_graph

    output = run_onnx_graph("model.onnx", {"input": my_input_array})
"""

import math
from typing import Any, Dict

import numpy as np
import onnx
from onnx import numpy_helper, AttributeProto
import torch
import torch.nn.functional as F


# ===========================================================================
# Section 1 – RNG utilities
# ===========================================================================


class Xorshift32:
    """32-bit xorshift PRNG, matching the hardware implementation."""

    def __init__(self, seed: int):
        self.state = int(seed) if int(seed) != 0 else 1  # zero state is invalid

    def next(self) -> int:
        s = self.state
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= (s >> 17) & 0xFFFFFFFF
        s ^= (s << 5) & 0xFFFFFFFF
        self.state = s & 0xFFFFFFFF
        return self.state


class Ziggurat:
    """
    Ziggurat normal sampler backed by Xorshift32, matching the hardware RNG
    used by PerturbNormal.

    The table initialisation and sampling loop are intentionally identical to
    the reference Python implementation in
    ``onnx4deeploy/operators/perturbnormal.py`` so that test outputs agree
    with what the hardware operator produces.
    """

    N = 256
    R = 3.442619855899

    def __init__(self, seed: int):
        self.rng = Xorshift32(seed)
        self.x = np.zeros(self.N + 1)
        self.y = np.zeros(self.N)
        self.x[0] = self.R
        self.x[self.N] = 0.0
        for i in range(1, self.N):
            self.x[i] = np.sqrt(-2.0 * np.log(np.exp(-0.5 * self.x[i - 1] ** 2)))
        for i in range(self.N):
            self.y[i] = np.exp(-0.5 * self.x[i] ** 2)

    def next(self) -> float:
        while True:
            k = self.rng.next() % self.N
            u = self.rng.next() / 0xFFFFFFFF
            x = u * (self.x[k] - self.x[k + 1]) + self.x[k + 1]
            if u < self.y[k] / self.y[k + 1]:
                return x
            if x < self.R:
                y = math.exp(-0.5 * x * x)
                if u * (self.y[k + 1] - self.y[k]) < (y - self.y[k]):
                    return x


def _ziggurat_array(seed: int, idx: int, numel: int, epsilon: float) -> np.ndarray:
    """
    Generate ``numel`` Ziggurat-normal samples scaled by ``epsilon``.

    The effective seed is ``seed + idx``, matching the unique-seed convention
    in ``zo_transform.py`` (``unique_seed = base_seed + perturbation_counter``).
    """
    rng = Ziggurat(int(seed) + int(idx))
    return np.array([rng.next() * epsilon for _ in range(numel)], dtype=np.float32)


def _xorshift_rademacher(seed: int, idx: int, numel: int) -> np.ndarray:
    """
    Generate ``numel`` Rademacher samples (±1) using Xorshift32.

    Odd raw value → +1, even raw value → -1 (least-significant bit test).
    """
    rng = Xorshift32(int(seed) + int(idx))
    return np.array([1 if rng.next() & 1 else -1 for _ in range(numel)],
                    dtype=np.float32)


def _xorshift_uniform(seed: int, idx: int, numel: int, low: float, high: float) -> np.ndarray:
    """Generate ``numel`` uniform samples in [low, high] using Xorshift32."""
    rng = Xorshift32(int(seed) + int(idx))
    span = high - low
    return np.array([rng.next() / 0xFFFFFFFF * span + low for _ in range(numel)],
                    dtype=np.float32)


# ===========================================================================
# Section 2a – Deeploy custom ops
# ===========================================================================


def exec_quant(x: np.ndarray, attrs: Dict[str, Any]) -> np.ndarray:
    """
    Deeploy ``Quant`` node.

    Formula::

        out = clip(round(x / scale) + zero_point, q_min, q_max)

    where ``[q_min, q_max]`` is ``[-128, 127]`` for signed int8 or
    ``[0, 255]`` for unsigned uint8.
    """
    scale = float(attrs["scale"])
    zero_point = int(attrs.get("zero_point", 0))
    bit_width = int(attrs.get("bit_width", 8))
    signed = bool(attrs.get("signed", 1))

    q_min = -(2 ** (bit_width - 1)) if signed else 0
    q_max = (2 ** (bit_width - 1)) - 1 if signed else (2 ** bit_width) - 1

    out = np.clip(np.round(x / scale).astype(np.int32) + zero_point, q_min, q_max)
    return out.astype(np.int8 if signed else np.uint8)


def exec_dequant(x: np.ndarray, attrs: Dict[str, Any]) -> np.ndarray:
    """
    Deeploy ``Dequant`` node.

    Formula::

        out = (x - zero_point) * scale
    """
    scale = float(attrs["scale"])
    zero_point = int(attrs.get("zero_point", 0))
    return ((x.astype(np.float32) - zero_point) * scale).astype(np.float32)


def exec_requant_shift(
    x: np.ndarray,
    mul: np.ndarray,
    add: np.ndarray,
    div: np.ndarray,
) -> np.ndarray:
    """
    Deeploy ``RequantShift`` node.

    Formula::

        out = clip(((x * mul) + add) >> log2(div), -128, 127)

    ``div`` is a power-of-two scalar; ``mul`` may be a per-channel vector.
    """
    div_val = int(np.asarray(div).flat[0])
    shift = int(round(math.log2(div_val))) if div_val > 1 else 0

    x64 = x.astype(np.int64)
    mul64 = np.asarray(mul, dtype=np.int64)
    add_val = int(np.asarray(add).flat[0])

    # Broadcast per-channel mul across spatial dims
    if mul64.ndim > 0 and mul64.size > 1:
        shape = [1] * x64.ndim
        shape[1] = mul64.size  # channel axis = 1
        mul64 = mul64.reshape(shape)

    out = ((x64 * mul64) + add_val) >> shift
    return np.clip(out, -128, 127).astype(np.int8)


# ===========================================================================
# Section 2b – MeZO perturbation ops
# ===========================================================================


def exec_perturb_normal(x: np.ndarray, attrs: Dict[str, Any]) -> np.ndarray:
    """
    MeZO ``PerturbNormal`` node (domain: ``mezo``).

    Formula::

        y = x + eps * ziggurat_normal(seed + idx)

    The noise array has the same shape as ``x``; the RNG is initialised with
    ``seed + idx`` for uniqueness across layers.

    .. note::
        The reference implementation in ``perturbnormal.py`` applies the RNG
        in a loop and uses only the **last** sample as a scalar perturbation.
        This implementation follows that same behaviour for fidelity.
    """
    seed = attrs.get("seed", 42)
    idx = attrs.get("idx", 0)
    eps = float(attrs.get("eps", 0.01))

    rng = Ziggurat(int(seed) + int(idx))
    noise = 0.0
    for _ in range(x.size):
        noise = rng.next() * eps  # scalar: last sample only (matches reference)
    return (x.astype(np.float32) + noise).astype(x.dtype)


def exec_perturb_uniform(x: np.ndarray, attrs: Dict[str, Any]) -> np.ndarray:
    """
    MeZO ``PerturbUniform`` node (domain: ``mezo``).

    Formula::

        y = x + eps * uniform(low, high, seed + idx)

    Per-element uniform noise with range ``[low, high]``, scaled by ``eps``.
    """
    seed = attrs.get("seed", 42)
    idx = attrs.get("idx", 0)
    eps = float(attrs.get("eps", 0.01))
    low = float(attrs.get("low", -math.sqrt(3)))
    high = float(attrs.get("high", math.sqrt(3)))

    noise = _xorshift_uniform(int(seed), int(idx), x.size, low, high)
    noise = noise.reshape(x.shape) * eps
    return (x.astype(np.float32) + noise).astype(x.dtype)


def exec_perturb_triangle(x: np.ndarray, attrs: Dict[str, Any]) -> np.ndarray:
    """
    MeZO ``PerturbTriangle`` node (domain: ``mezo``).

    A triangle-distribution perturbation is approximated as the sum of two
    independent uniform samples on ``[low/2, high/2]``, scaled by ``eps``.
    """
    seed = attrs.get("seed", 42)
    idx = attrs.get("idx", 0)
    eps = float(attrs.get("eps", 0.01))
    low = float(attrs.get("low", -math.sqrt(6)))
    high = float(attrs.get("high", math.sqrt(6)))

    u1 = _xorshift_uniform(int(seed), int(idx), x.size, low / 2, high / 2)
    u2 = _xorshift_uniform(int(seed) + 1, int(idx), x.size, low / 2, high / 2)
    noise = ((u1 + u2) * eps).reshape(x.shape)
    return (x.astype(np.float32) + noise).astype(x.dtype)


def exec_perturb_rademacher(x: np.ndarray, attrs: Dict[str, Any]) -> np.ndarray:
    """
    MeZO ``PerturbRademacher`` node (domain: ``mezo``).

    Formula::

        y = x + eps * rademacher(seed + idx)

    Each element of the noise is independently ±1.
    """
    seed = attrs.get("seed", 42)
    idx = attrs.get("idx", 0)
    eps = float(attrs.get("eps", 0.01))

    noise = _xorshift_rademacher(int(seed), int(idx), x.size).reshape(x.shape) * eps
    return (x.astype(np.float32) + noise).astype(x.dtype)


def exec_perturb_eggroll(shape_input: np.ndarray, attrs: Dict[str, Any]) -> np.ndarray:
    """
    MeZO ``PerturbEggroll`` node (domain: ``com.microsoft``).

    Generates a Rademacher vector whose length is given by ``shape_input``
    (a 1-D INT64 tensor encoding the desired output shape).

    In the eggroll factored perturbation, two such vectors ``a`` and ``b``
    are combined as ``noise = eps * (a @ b.T)`` by a downstream Gemm node;
    this function only generates one vector per call.
    """
    seed = attrs.get("seed", 42)
    idx = attrs.get("idx", 0)
    eps = float(attrs.get("eps", 0.01))

    out_shape = tuple(int(d) for d in shape_input.flat)
    numel = int(np.prod(out_shape))
    return _xorshift_rademacher(int(seed), int(idx), numel).reshape(out_shape)


def exec_rqs_perturb_rademacher(
    x: np.ndarray, mul: np.ndarray, attrs: Dict[str, Any]
) -> np.ndarray:
    """
    MeZO ``RQSPerturbRademacher`` node (domain: ``mezo``).

    Quantised Rademacher perturbation::

        noise = (rademacher(seed + idx) * mul) >> log2(div)
        y = clip(x + noise, -(2^(n_bits-1)), 2^(n_bits-1)-1)

    ``mul`` is a per-output-channel scaling factor (int32).
    ``div`` is a power-of-two shift (matches the weight quantisation scale).
    """
    seed = attrs.get("seed", 42)
    idx = attrs.get("idx", 0)
    div = int(attrs.get("div", 2 ** 15))
    n_levels = int(attrs.get("n_levels", 256))
    signed = bool(attrs.get("signed", 1))

    shift = int(round(math.log2(div))) if div > 1 else 0
    q_min = -(n_levels // 2) if signed else 0
    q_max = (n_levels // 2) - 1 if signed else n_levels - 1

    rad = _xorshift_rademacher(int(seed), int(idx), x.size).reshape(x.shape).astype(np.int32)
    mul32 = np.asarray(mul, dtype=np.int32)

    # Broadcast per-channel mul
    if mul32.ndim > 0 and mul32.size > 1:
        shape = [1] * x.ndim
        shape[0] = mul32.size  # first axis = output channels for weights
        mul32 = mul32.reshape(shape)

    noise = (rad * mul32) >> shift
    return np.clip(x.astype(np.int32) + noise, q_min, q_max).astype(x.dtype)


def exec_rqs_perturb_uniform(
    x: np.ndarray, mul: np.ndarray, attrs: Dict[str, Any]
) -> np.ndarray:
    """
    MeZO ``RQSPerturbUniform`` node (domain: ``mezo``).

    Quantised uniform perturbation::

        noise = (uniform(-1, 1, seed + idx) * mul) >> log2(div)
        y = clip(x + noise, -(2^(n_bits-1)), 2^(n_bits-1)-1)
    """
    seed = attrs.get("seed", 42)
    idx = attrs.get("idx", 0)
    div = int(attrs.get("div", 2 ** 15))
    n_levels = int(attrs.get("n_levels", 256))
    signed = bool(attrs.get("signed", 1))

    shift = int(round(math.log2(div))) if div > 1 else 0
    q_min = -(n_levels // 2) if signed else 0
    q_max = (n_levels // 2) - 1 if signed else n_levels - 1

    uni = _xorshift_uniform(int(seed), int(idx), x.size, -1.0, 1.0).reshape(x.shape)
    mul32 = np.asarray(mul, dtype=np.float32)

    if mul32.ndim > 0 and mul32.size > 1:
        shape = [1] * x.ndim
        shape[0] = mul32.size
        mul32 = mul32.reshape(shape)

    noise = (uni * mul32).astype(np.int32) >> shift
    return np.clip(x.astype(np.int32) + noise, q_min, q_max).astype(x.dtype)


# ===========================================================================
# Section 3 – Standard ONNX op implementations
# ===========================================================================


def exec_quantize_linear(
    x: np.ndarray, scale: np.ndarray, zero_point: np.ndarray, axis: int = 1
) -> np.ndarray:
    """Standard ONNX ``QuantizeLinear``."""
    s = scale.astype(np.float32)
    zp = zero_point
    q_min, q_max = (-128, 127) if zp.dtype == np.int8 else (0, 255)

    if s.ndim > 0 and s.size > 1:  # per-channel: broadcast along the given axis
        shape = [1] * x.ndim
        shape[axis] = s.size
        s = s.reshape(shape)
        zp = zp.reshape(shape)

    out = np.clip(np.round(x / s) + zp.astype(np.float32), q_min, q_max)
    return out.astype(zp.dtype)


def exec_dequantize_linear(
    x: np.ndarray, scale: np.ndarray, zero_point: np.ndarray, axis: int = 1
) -> np.ndarray:
    """Standard ONNX ``DequantizeLinear``."""
    s = scale.astype(np.float32)
    if s.ndim > 0 and s.size > 1:  # per-channel: broadcast along the given axis
        shape = [1] * x.ndim
        shape[axis] = s.size
        s = s.reshape(shape)
        zero_point = zero_point.reshape(shape)
    return ((x.astype(np.float32) - zero_point.astype(np.float32)) * s).astype(np.float32)


def exec_conv(node_inputs: list, attrs: Dict[str, Any]) -> np.ndarray:
    """Standard ONNX ``Conv`` via ``torch.nn.functional.conv2d``.

    Handles asymmetric padding by pre-padding with ``F.pad`` before the
    convolution (PyTorch ``padding=`` only supports symmetric values).
    ONNX pads format: [H_begin, W_begin, H_end, W_end].
    """
    x = torch.from_numpy(node_inputs[0].astype(np.float32))
    w = torch.from_numpy(node_inputs[1].astype(np.float32))
    b = torch.from_numpy(node_inputs[2].astype(np.float32)) if len(node_inputs) > 2 else None

    strides = tuple(attrs.get("strides", [1, 1]))
    pads = attrs.get("pads", [0, 0, 0, 0])  # [H_begin, W_begin, H_end, W_end]
    dilations = tuple(attrs.get("dilations", [1, 1]))
    group = int(attrs.get("group", 1))

    # F.pad order is (W_begin, W_end, H_begin, H_end) — innermost dim first
    x = F.pad(x, (pads[1], pads[3], pads[0], pads[2]))

    return F.conv2d(x, w, bias=b, stride=strides, padding=0,
                    dilation=dilations, groups=group).numpy()


def exec_maxpool(x: np.ndarray, attrs: Dict[str, Any]) -> np.ndarray:
    """Standard ONNX ``MaxPool`` via ``torch.nn.functional.max_pool2d``.

    Handles asymmetric padding via ``F.pad`` pre-padding.
    ONNX pads format: [H_begin, W_begin, H_end, W_end].
    """
    t = torch.from_numpy(x.astype(np.float32))
    kernel = tuple(attrs["kernel_shape"])
    strides = tuple(attrs.get("strides", kernel))
    pads = attrs.get("pads", [0] * (2 * len(kernel)))

    # F.pad order: (W_begin, W_end, H_begin, H_end)
    t = F.pad(t, (pads[1], pads[3], pads[0], pads[2]), value=-float("inf"))

    return F.max_pool2d(t, kernel_size=kernel, stride=strides, padding=0).numpy()


def exec_gemm(node_inputs: list, attrs: Dict[str, Any]) -> np.ndarray:
    """Standard ONNX ``Gemm``: ``Y = alpha * A @ B + beta * C``."""
    A, B = node_inputs[0].astype(np.float32), node_inputs[1].astype(np.float32)
    C = node_inputs[2].astype(np.float32) if len(node_inputs) > 2 else None
    alpha = float(attrs.get("alpha", 1.0))
    beta = float(attrs.get("beta", 1.0))
    if attrs.get("transA", 0):
        A = A.T
    if attrs.get("transB", 0):
        B = B.T
    out = alpha * (A @ B)
    if C is not None:
        out += beta * C
    return out


def exec_relu(x: np.ndarray) -> np.ndarray:
    """Standard ONNX ``Relu``."""
    return np.maximum(x, 0).astype(x.dtype)


def exec_flatten(x: np.ndarray, attrs: Dict[str, Any]) -> np.ndarray:
    """Standard ONNX ``Flatten``."""
    axis = int(attrs.get("axis", 1))
    outer = int(np.prod(x.shape[:axis]))
    inner = int(np.prod(x.shape[axis:]))
    return x.reshape(outer, inner)


def exec_reshape(node_inputs: list, attrs: Dict[str, Any]) -> np.ndarray:
    """Standard ONNX ``Reshape``."""
    x = node_inputs[0]
    if len(node_inputs) > 1:
        shape = [int(s) for s in node_inputs[1].flat]
    else:
        shape = list(attrs.get("shape", x.shape))
    # Replace 0 with corresponding input dim; -1 is inferred by numpy
    resolved = [x.shape[i] if s == 0 and i < x.ndim else s for i, s in enumerate(shape)]
    return x.reshape(resolved)


def exec_clip(node_inputs: list, attrs: Dict[str, Any]) -> np.ndarray:
    """Standard ONNX ``Clip``."""
    x = node_inputs[0].astype(np.float32)
    lo = float(node_inputs[1]) if len(node_inputs) > 1 and node_inputs[1] is not None \
        else float(attrs.get("min", -np.inf))
    hi = float(node_inputs[2]) if len(node_inputs) > 2 and node_inputs[2] is not None \
        else float(attrs.get("max", np.inf))
    return np.clip(x, lo, hi)


def exec_softmax_cross_entropy_loss(
    node_inputs: list, attrs: Dict[str, Any]
) -> np.ndarray:
    """
    Standard ONNX ``SoftmaxCrossEntropyLoss`` (appended by ``append_cross_entropy_loss``).

    Returns the mean cross-entropy loss over the batch.
    """
    logits = node_inputs[0].astype(np.float32)   # (N, C)
    labels = node_inputs[1].astype(np.int64).flatten()  # (N,)
    log_softmax = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))
    loss = -log_softmax[np.arange(len(labels)), labels]
    reduction = attrs.get("reduction", "mean")
    return np.array(loss.mean() if reduction == "mean" else loss.sum(), dtype=np.float32)


# ===========================================================================
# Section 4 – ONNX attribute reader helper
# ===========================================================================


def _read_attr(attr) -> Any:
    """Return the Python value of an ``onnx.AttributeProto``."""
    t = attr.type
    if t == AttributeProto.FLOAT:
        return attr.f
    if t == AttributeProto.INT:
        return attr.i
    if t == AttributeProto.STRING:
        return attr.s.decode()
    if t == AttributeProto.TENSOR:
        return numpy_helper.to_array(attr.t)
    if t == AttributeProto.FLOATS:
        return list(attr.floats)
    if t == AttributeProto.INTS:
        return list(attr.ints)
    return None


def _node_attrs(node) -> Dict[str, Any]:
    return {a.name: _read_attr(a) for a in node.attribute}


# ===========================================================================
# Section 5 – ONNX graph executor
# ===========================================================================


def run_onnx_graph(
    onnx_path: str,
    inputs_dict: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Execute an ONNX graph with pure Python / NumPy / PyTorch.

    Supports:
    * Standard ops – Conv, Relu, Clip, MaxPool, GlobalAveragePool, Gemm,
      Reshape, Flatten, Transpose, MatMul, Add, Sub, Mul, Div, Round, Cast,
      Identity, Dropout, QuantizeLinear, DequantizeLinear,
      SoftmaxCrossEntropyLoss.
    * Deeploy custom ops – Quant, Dequant, RequantShift.
    * MeZO perturbation ops – PerturbNormal, PerturbUniform, PerturbTriangle,
      PerturbRademacher, PerturbEggroll, RQSPerturbRademacher,
      RQSPerturbUniform.

    Args:
        onnx_path:   Path to the ONNX model file.
        inputs_dict: Mapping from graph-input name to numpy array.

    Returns:
        The first graph output as a float32 NumPy array.

    Raises:
        NotImplementedError: If the graph contains an op not listed above.
    """
    model = onnx.load(onnx_path)
    graph = model.graph

    # Build tensor registry: initializers + supplied inputs
    tensors: Dict[str, np.ndarray] = {}
    for init in graph.initializer:
        tensors[init.name] = numpy_helper.to_array(init)
    tensors.update(inputs_dict)

    for node in graph.node:
        op = node.op_type
        attrs = _node_attrs(node)

        # Collect inputs, allowing empty optional inputs to stay None
        node_inputs = [
            tensors.get(n, None) for n in node.input
        ]
        # Compact list (drop trailing Nones, keep internal Nones for optional args)
        while node_inputs and node_inputs[-1] is None:
            node_inputs.pop()

        # ------------------------------------------------------------------
        # Deeploy custom ops
        # ------------------------------------------------------------------
        if op == "Quant":
            out = exec_quant(node_inputs[0], attrs)

        elif op == "Dequant":
            out = exec_dequant(node_inputs[0], attrs)

        elif op == "RequantShift":
            x, mul, add, div = node_inputs
            out = exec_requant_shift(x, mul, add, div)

        # ------------------------------------------------------------------
        # MeZO perturbation ops
        # ------------------------------------------------------------------
        elif op == "PerturbNormal":
            out = exec_perturb_normal(node_inputs[0], attrs)

        elif op == "PerturbUniform":
            out = exec_perturb_uniform(node_inputs[0], attrs)

        elif op == "PerturbTriangle":
            out = exec_perturb_triangle(node_inputs[0], attrs)

        elif op == "PerturbRademacher":
            out = exec_perturb_rademacher(node_inputs[0], attrs)

        elif op == "PerturbEggroll":
            # Input is a shape tensor (INT64); output is a Rademacher vector.
            out = exec_perturb_eggroll(node_inputs[0], attrs)

        elif op == "RQSPerturbRademacher":
            x, mul = node_inputs[0], node_inputs[1]
            out = exec_rqs_perturb_rademacher(x, mul, attrs)

        elif op == "RQSPerturbUniform":
            x, mul = node_inputs[0], node_inputs[1]
            out = exec_rqs_perturb_uniform(x, mul, attrs)

        # ------------------------------------------------------------------
        # Standard ONNX QDQ ops
        # ------------------------------------------------------------------
        elif op == "QuantizeLinear":
            x = node_inputs[0]
            scale = node_inputs[1] if len(node_inputs) > 1 else np.array(1.0, np.float32)
            zp = node_inputs[2] if len(node_inputs) > 2 else np.array(0, np.int8)
            out = exec_quantize_linear(x, scale, zp, axis=int(attrs.get("axis", 1)))

        elif op == "DequantizeLinear":
            x = node_inputs[0]
            scale = node_inputs[1] if len(node_inputs) > 1 else np.array(1.0, np.float32)
            zp = node_inputs[2] if len(node_inputs) > 2 else np.array(0, np.int8)
            out = exec_dequantize_linear(x, scale, zp, axis=int(attrs.get("axis", 1)))

        # ------------------------------------------------------------------
        # Standard ONNX compute ops
        # ------------------------------------------------------------------
        elif op == "Conv":
            out = exec_conv(node_inputs, attrs)

        elif op == "Relu":
            out = exec_relu(node_inputs[0])

        elif op == "Clip":
            out = exec_clip(node_inputs, attrs)

        elif op == "MaxPool":
            out = exec_maxpool(node_inputs[0], attrs)

        elif op == "GlobalAveragePool":
            out = node_inputs[0].astype(np.float32).mean(axis=(2, 3), keepdims=True)

        elif op == "Gemm":
            out = exec_gemm(node_inputs, attrs)

        elif op == "MatMul":
            out = (node_inputs[0].astype(np.float32)
                   @ node_inputs[1].astype(np.float32))

        elif op == "Add":
            out = (node_inputs[0].astype(np.float32)
                   + node_inputs[1].astype(np.float32))

        elif op == "Sub":
            out = (node_inputs[0].astype(np.float32)
                   - node_inputs[1].astype(np.float32))

        elif op == "Mul":
            out = (node_inputs[0].astype(np.float32)
                   * node_inputs[1].astype(np.float32))

        elif op == "Div":
            out = (node_inputs[0].astype(np.float32)
                   / node_inputs[1].astype(np.float32))

        elif op == "Round":
            out = np.round(node_inputs[0].astype(np.float32))

        elif op == "Cast":
            dtype_map = {
                1: np.float32, 2: np.uint8, 3: np.int8, 5: np.int16,
                6: np.int32, 7: np.int64, 9: bool, 10: np.float16, 11: np.float64,
            }
            to = int(attrs.get("to", 1))
            out = node_inputs[0].astype(dtype_map[to])

        elif op == "Flatten":
            out = exec_flatten(node_inputs[0], attrs)

        elif op == "Reshape":
            out = exec_reshape(node_inputs, attrs)

        elif op == "Transpose":
            perm = attrs.get("perm", None)
            out = np.transpose(node_inputs[0], perm)

        elif op in ("Identity", "Dropout"):
            out = node_inputs[0]

        elif op == "SoftmaxCrossEntropyLoss":
            out = exec_softmax_cross_entropy_loss(node_inputs, attrs)

        else:
            raise NotImplementedError(
                f"Unsupported ONNX op '{op}' (node '{node.name}'). "
                "Add an implementation to onnx_node_implementations.py."
            )

        # Store each named output
        for out_name, out_val in zip(node.output, [out]):
            if out_name:
                tensors[out_name] = out_val

    output_name = graph.output[0].name
    return tensors[output_name].astype(np.float32)
