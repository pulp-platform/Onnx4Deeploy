#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT
"""Emit the three NE16 benchmark kernel tests into a Deeploy checkout.

    python3 scripts/gen_ne16_bench_tests.py --deeploy ~/Deeploy-ne16

Writes, for each of NE16's native modes:

    DeeployTest/Tests/Kernels/Integer/Conv/NE16Bench_{Dense,DW,PW}_RQ/
        network.onnx  inputs.npz  outputs.npz
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from onnx4deeploy.operators.ne16_rqs_conv2d import MODES, NE16RQSConv2DTest  # noqa: E402

DIRNAME = {"dense": "NE16Bench_Dense_RQ", "dw": "NE16Bench_DW_RQ", "pw": "NE16Bench_PW_RQ"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deeploy", required=True, help="path to the Deeploy checkout")
    ap.add_argument("--modes", nargs="*", default=list(MODES), choices=list(MODES))
    args = ap.parse_args()

    root = os.path.join(
        os.path.expanduser(args.deeploy), "DeeployTest", "Tests", "Kernels", "Integer", "Conv"
    )
    if not os.path.isdir(root):
        print(f"❌ not a Deeploy checkout: {root}")
        return 1

    for mode in args.modes:
        out_dir = os.path.join(root, DIRNAME[mode])
        test = NE16RQSConv2DTest(mode=mode, save_path=out_dir)
        test.generate()
        print(f"   {test.describe()}  ->  {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
