# -*- coding: utf-8 -*-
"""打印 notebook 里指定格子的代码（供本地复核）。

用法：python show_cell.py <关键词>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NB = Path(r"D:\A\zfy-v5\zfy_v5_fullft.ipynb")
kw = sys.argv[1] if len(sys.argv) > 1 else "tmp_smoke"

nb = json.loads(NB.read_text(encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    s = "".join(c["source"])
    if kw in s:
        print(f"{'='*72}\ncell {i}  ({c['cell_type']})\n{'='*72}")
        print(s)
        print()
