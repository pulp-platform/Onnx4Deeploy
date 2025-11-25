#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从日志文件中统计 L3 节点的 DMA 周期：
- 仅统计“节点名以 _L3 结尾”的行
- 仅统计 "Input DMA took X cycles" / "Output DMA took X cycles"
- 输出总计与按 node+tile 的明细

用法:
    python sum_dma_l3.py /path/to/your_log.txt
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

# 例： 
# [node_0_classifier_blocks_1_self_attn_k_proj_Transpose__0_sgd_L3][DB][32768 ops][Tile 0] Input DMA took 92043 cycles
# [node_0_classifier_blocks_1_self_attn_k_proj_Transpose__0_sgd_L3][DB][32768 ops][Tile 1] Output DMA took 44948 cycles

# 正则说明：
# - 捕获以 _L3 结尾的 node 名（不含右方括号）
# - 可选地捕获 Tile 索引（如无 Tile 也不报错）
# - 捕获 "Input DMA" 或 "Output DMA"
# - 捕获 cycles 数字
LINE_RE = re.compile(
    r"""
    \[
        (?P<node>[^\]]*?_L3)      # node 名以 _L3 结尾
    \]
    [^\n]*?                       # 中间任意
    (?:\[Tile\s+(?P<tile>\d+)\])? # 可选 Tile
    [^\n]*?
    (?P<kind>Input\ DMA|Output\ DMA)
    \s+took\s+
    (?P<cycles>\d+)
    \s+cycles
    """,
    re.IGNORECASE | re.VERBOSE
)

def parse_file(path: Path):
    input_total = 0
    output_total = 0
    # 明细： key=(node, tile)  -> {'input': x, 'output': y}
    breakdown = defaultdict(lambda: {'input': 0, 'output': 0})

    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for ln, line in enumerate(f, 1):
            m = LINE_RE.search(line)
            if not m:
                continue
            node = m.group('node')
            tile = m.group('tile') or 'NA'
            kind = m.group('kind').lower()  # 'input dma' / 'output dma'
            cycles = int(m.group('cycles'))

            key = (node, tile)
            if 'input dma' in kind:
                input_total += cycles
                breakdown[key]['input'] += cycles
            else:
                output_total += cycles
                breakdown[key]['output'] += cycles

    return input_total, output_total, breakdown

def print_report(input_total, output_total, breakdown):
    grand = input_total + output_total
    print("=== L3 DMA Cycles Summary ===")
    print(f"L3_Input_DMA_Total  : {input_total:,} cycles")
    print(f"L3_Output_DMA_Total : {output_total:,} cycles")
    print(f"L3_DMA_Grand_Total  : {grand:,} cycles")

    # if breakdown:
    #     print("\n--- Per Node (with Tile) Breakdown ---")
    #     # 排序：按 node 名，再按 tile
    #     for (node, tile) in sorted(breakdown.keys(), key=lambda x: (x[0], int(x[1]) if str(x[1]).isdigit() else -1)):
    #         b = breakdown[(node, tile)]
    #         subtotal = b['input'] + b['output']
    #         print(f"{node} | Tile {tile}:  Input={b['input']:,}  Output={b['output']:,}  Total={subtotal:,}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python sum_dma_l3.py /path/to/log.txt")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    input_total, output_total, breakdown = parse_file(path)
    print_report(input_total, output_total, breakdown)

if __name__ == "__main__":
    main()
