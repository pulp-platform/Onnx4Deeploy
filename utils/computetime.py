#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从日志文件中统计 L2 节点的 compute（Kernel）周期：
- 仅统计“节点名以 _L2 结尾”的行
- 仅统计 "Kernel took X cycles"（把 Kernel 视为 compute）
- 输出明细，并在最后打印总计

用法:
    python sum_compute_l2.py /path/to/your_log.txt
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

LINE_RE = re.compile(
    r"""
    \[
      (?P<node>[^\]]*?_L2)     
    \]
    [^\n]*?                     
    (?:\[Tile\s+(?P<tile>\d+)\])?  
    [^\n]*?
    (?P<kind>Kernel)           
    \s+took\s+
    (?P<cycles>\d+)            
    \s+cycles
    """,
    re.IGNORECASE | re.VERBOSE
)

def parse_file(path: Path):
    compute_total = 0
    breakdown = defaultdict(lambda: {'compute': 0})

    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for ln, line in enumerate(f, 1):
            m = LINE_RE.search(line)
            if not m:
                continue

            node = m.group('node')
            tile = m.group('tile') or 'NA'
            cycles = int(m.group('cycles'))

            key = (node, tile)
            compute_total += cycles
            breakdown[key]['compute'] += cycles

    return compute_total, breakdown

def print_report(compute_total, breakdown):
    print("=== L2 Compute (Kernel) Cycles Summary ===")

    if breakdown:
        print("\n--- Per Node (with Tile) Breakdown ---")
        for (node, tile) in sorted(
            breakdown.keys(),
            key=lambda x: (x[0], int(x[1]) if str(x[1]).isdigit() else -1)
        ):
            c = breakdown[(node, tile)]['compute']
            print(f"{node} | Tile {tile}: Compute={c:,}")


    print("\n--- Total ---")
    print(f"L2_Compute_Total : {compute_total:,} cycles")

def main():
    if len(sys.argv) < 2:
        print("Usage: python sum_compute_l2.py /path/to/log.txt")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    compute_total, breakdown = parse_file(path)
    print_report(compute_total, breakdown)

if __name__ == "__main__":
    main()
