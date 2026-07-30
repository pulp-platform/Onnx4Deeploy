# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT
"""Requantized integer Conv2D test generator for the GAP9 NE16 accelerator.

NE16 has exactly three native convolution modes (see AI_AGENT/NE16/
ne16_architecture.md section 6):

    ``dense``  CONV3x3      3x3, group == 1
    ``dw``     CONV3x3_DW   3x3, group == C_in == C_out
    ``pw``     CONV1x1      1x1, group == 1

This generator emits one single-operator Deeploy test per mode, sized so the
accelerator is actually the bottleneck rather than the dispatch overhead.

Why this does not reuse ``Conv2DOperatorTest``
----------------------------------------------
Deeploy's integer path expects ``Conv -> RequantShift``, where ``RequantShift``
is a Deeploy-private op that ONNX Runtime cannot execute.  The base class runs
ORT to produce the golden outputs, so ``run_inference`` is overridden with an
integer NumPy reference.  That reference is bit-exact against the three
pre-existing NE16 fixtures (Dense_2D_RQ, DW_2D_RQ, PW_2D_RQ/Regular_RQ).

Shape rules that keep NE16 utilisation high
-------------------------------------------
* ``C_in`` a multiple of 16 (``TP_IN``) and ``C_out`` a multiple of 32
  (``TP_OUT``); for depthwise the output tile is 16, so 16 is enough.
  Anything else wastes ``subtile_rem_*`` columns.
* 8-bit weights (``qw = 8``).
* Inputs are unsigned.  The NE16 HAL has no signed-input flag in CONFIG0, so a
  signed input silently prevents the NE16 offload.

Constraints that make the node parse at all (Deeploy/Targets/NE16/Parsers.py)
-----------------------------------------------------------------------------
* The Conv must have **no bias**.  After the requant merge the node inputs are
  positionally mapped to ``[data_in, weight, mul, add]``, so a third Conv input
  shifts ``mul``/``add`` and the parse fails.
* ``dilations == [1, 1]`` and ``strides == [1, 1]`` (strides need
  ``--enableStrides``).
* ``dense`` and ``dw`` additionally need ``--enable-3x3`` on the runner;
  without it ``NE16Engine.canExecute`` only accepts pointwise.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .base_operator import BaseOperatorTest

# NE16 hardware granularity, from archi/ne16 + ne16_architecture.md
TP_IN = 16
TP_OUT = 32

MODES = ("dense", "dw", "pw")


def requant_shift(
    acc: np.ndarray, mul: np.ndarray, add: np.ndarray, log2d: int, signed: bool
) -> np.ndarray:
    """Deeploy's RequantShift, integer-exact.

    Mirrors ``TargetLibraries/Generic/src/RequantShift_s8.c`` with rounding
    enabled, which is what the shipped fixtures were generated with (verified
    bit-exact against all three of them).
    """
    out = acc.astype(np.int64) * mul.astype(np.int64) + add.astype(np.int64)
    if log2d > 0:
        out = out + (1 << (log2d - 1))
    out = out >> log2d
    lo, hi = (-128, 127) if signed else (0, 255)
    return np.clip(out, lo, hi)


def conv2d_int(
    x: np.ndarray, w: np.ndarray, pads: Tuple[int, int, int, int], group: int
) -> np.ndarray:
    """Stride-1 NCHW integer convolution, exact in int64. No bias."""
    _, c_in, h, w_in = x.shape
    c_out, c_in_per_group, kh, kw = w.shape
    pt, pl, pb, pr = pads
    xp = np.pad(x.astype(np.int64), ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    ho, wo = h + pt + pb - kh + 1, w_in + pl + pr - kw + 1
    out = np.zeros((1, c_out, ho, wo), dtype=np.int64)
    out_per_group = c_out // group
    for co in range(c_out):
        g = co // out_per_group
        for ci in range(c_in_per_group):
            src = xp[0, g * (c_in // group) + ci]
            for i in range(kh):
                for j in range(kw):
                    out[0, co] += int(w[co, ci, i, j]) * src[i : i + ho, j : j + wo]
    return out


class NE16RQSConv2DTest(BaseOperatorTest):
    """Generate one ``Conv -> RequantShift`` Deeploy test for an NE16 mode."""

    def __init__(
        self,
        mode: str = "dense",
        config_path: Optional[str] = None,
        save_path: Optional[str] = None,
        seed: int = 0,
    ):
        super().__init__(config_path, save_path)
        assert mode in MODES, f"mode must be one of {MODES}, got {mode!r}"
        self.mode = mode
        self.seed = seed
        self.params: Dict[str, Any] = {}

    # -- BaseOperatorTest interface ------------------------------------------------

    def get_operator_name(self) -> str:
        return "Conv"

    def load_config(self) -> Dict[str, Any]:
        config = super().load_config()
        section = config["ne16_rqs_conv2d"]
        self.params = dict(section["defaults"])
        self.params.update(section["modes"][self.mode])
        self._check_shapes()
        return config

    def _check_shapes(self) -> None:
        p = self.params
        c_in, c_out = p["c_in"], p["c_out"]
        if self.mode == "dw":
            assert c_in == c_out, f"depthwise needs c_in == c_out, got {c_in} vs {c_out}"
            if c_in % TP_IN:
                print(
                    f"⚠️  c_in={c_in} is not a multiple of {TP_IN}; NE16 will waste input subtile columns"
                )
        else:
            if c_in % TP_IN:
                print(
                    f"⚠️  c_in={c_in} is not a multiple of {TP_IN}; NE16 will waste input subtile columns"
                )
            if c_out % TP_OUT:
                print(
                    f"⚠️  c_out={c_out} is not a multiple of {TP_OUT}; NE16 will waste output subtile columns"
                )

    @property
    def _kernel_shape(self) -> Tuple[int, int]:
        return (1, 1) if self.mode == "pw" else (3, 3)

    @property
    def _group(self) -> int:
        return self.params["c_in"] if self.mode == "dw" else 1

    @property
    def _pads(self) -> Tuple[int, int, int, int]:
        return (0, 0, 0, 0) if self.mode == "pw" else (1, 1, 1, 1)

    def generate_inputs(self) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        p = self.params
        # Unsigned activations: NE16's CONFIG0 has no signed-input flag.
        x = rng.integers(0, 256, size=(1, p["c_in"], p["height"], p["width"]), dtype=np.int64)
        return {"input": x}

    def _make_weight(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 1)
        p = self.params
        kh, kw = self._kernel_shape
        c_in_per_group = 1 if self.mode == "dw" else p["c_in"]
        # 8-bit signed weights (qw = 8) -- the NE16 sweet spot.
        return rng.integers(-128, 128, size=(p["c_out"], c_in_per_group, kh, kw), dtype=np.int64)

    def _fit_requant(self, acc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Derive per-channel mul/add from the real accumulator distribution.

        A fixture whose outputs are mostly clipped is a weak test: saturation
        hides arithmetic errors.  Pick the affine map that puts roughly +-3
        sigma of each channel's accumulator inside the int8 range, the way a
        real quantiser would, so only a few percent of outputs saturate.
        """
        p = self.params
        target_sigma = self.params.get("target_sigma", 40.0)
        scale = float(2 ** p["log2d"])
        c_out = acc.shape[1]
        mul = np.empty((c_out, 1, 1), dtype=np.int64)
        add = np.empty((c_out, 1, 1), dtype=np.int64)
        for co in range(c_out):
            chan = acc[0, co]
            sigma = float(chan.std())
            mean = float(chan.mean())
            m = 1 if sigma == 0.0 else max(1, int(round(target_sigma * scale / sigma)))
            mul[co, 0, 0] = m
            # Centre on 0 for a signed output, on the middle of [0,255] otherwise.
            centre = 0.0 if p["signed"] else 128.0
            add[co, 0, 0] = int(round(centre * scale - mean * m))
        return mul, add

    def create_onnx_graph(self, inputs: Dict[str, np.ndarray]) -> onnx.GraphProto:
        p = self.params
        kh, kw = self._kernel_shape
        pads = self._pads

        x = inputs["input"]
        weight = self._make_weight()
        acc = conv2d_int(x, weight, pads, self._group)
        mul, add = self._fit_requant(acc)
        consts = {"weight": weight, "mul": mul, "add": add}
        self._consts = consts  # reused by run_inference
        out = requant_shift(acc, mul[None], add[None], p["log2d"], bool(p["signed"]))
        self._golden = out
        sat = float(
            ((out == (-128 if p["signed"] else 0)) | (out == (127 if p["signed"] else 255))).mean()
        )
        print(f"   golden: {out.min()}..{out.max()}, saturated {sat * 100:.1f}%")
        if sat > 0.15:
            print("⚠️  more than 15% of outputs are clipped -- lower target_sigma in config.yaml")

        def const(name: str, arr: np.ndarray) -> onnx.TensorProto:
            # Deeploy's integer fixtures carry integer values in FLOAT tensors.
            return numpy_helper.from_array(arr.astype(np.float32), name=name)

        conv = helper.make_node(
            "Conv",
            inputs=["input", "weight"],  # no bias: see module docstring
            outputs=["conv_out"],
            dilations=[1, 1],
            group=self._group,
            kernel_shape=[kh, kw],
            pads=list(pads),
            strides=[1, 1],
        )
        rqs = helper.make_node(
            "RequantShift",
            inputs=["conv_out", "mul", "add"],
            outputs=["output"],
        )
        # These three are TENSOR-typed floats in every shipped fixture; the
        # parser reads them through NodeParser._unpack_const.
        rqs.attribute.extend(
            [
                helper.make_attribute(
                    "div", numpy_helper.from_array(np.array(2.0 ** p["log2d"], dtype=np.float32))
                ),
                helper.make_attribute(
                    "n_levels_out", numpy_helper.from_array(np.array([256.0], dtype=np.float32))
                ),
                helper.make_attribute(
                    "signed",
                    numpy_helper.from_array(np.array([float(p["signed"])], dtype=np.float32)),
                ),
            ]
        )

        graph = helper.make_graph(
            nodes=[conv, rqs],
            name=f"ne16_{self.mode}_rqs_conv",
            inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, list(x.shape))],
            outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, list(out.shape))],
            initializer=[const(n, consts[n]) for n in ("weight", "mul", "add")],
        )
        return graph

    def create_model(self, graph: onnx.GraphProto, opset_version: int = 13) -> onnx.ModelProto:
        # Deeploy's integer fixtures are opset 13; RequantShift is not a
        # standard op so the model is intentionally not checked.
        return helper.make_model(
            graph,
            producer_name=f"ne16_{self.mode}_rqs_conv",
            opset_imports=[helper.make_opsetid("", opset_version)],
        )

    def run_inference(self, onnx_file: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # ORT cannot execute RequantShift; the golden was computed in NumPy.
        _ = onnx_file, inputs
        return {"output": self._golden}

    def save_data(self, inputs, outputs, input_file, output_file):
        # Deeploy reads these with int64 semantics, matching the shipped fixtures.
        np.savez(input_file, **{k: v.astype(np.int64) for k, v in inputs.items()})
        np.savez(output_file, **{k: v.astype(np.int64) for k, v in outputs.items()})
        print(f"✅ Saved {input_file} and {output_file}")

    # -- reporting -----------------------------------------------------------------

    def describe(self) -> str:
        p = self.params
        kh, kw = self._kernel_shape
        ho = p["height"] + self._pads[0] + self._pads[2] - kh + 1
        wo = p["width"] + self._pads[1] + self._pads[3] - kw + 1
        c_in_per_group = 1 if self.mode == "dw" else p["c_in"]
        macs = p["c_out"] * c_in_per_group * kh * kw * ho * wo
        return (
            f"{self.mode:5} {kh}x{kw} group={self._group:<4} "
            f"in=[1,{p['c_in']},{p['height']},{p['width']}] out=[1,{p['c_out']},{ho},{wo}] "
            f"{macs / 1e6:.2f} MMAC"
        )
