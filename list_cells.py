# -*- coding: utf-8 -*-
"""列出 notebook 的格子编号与用途（供用户对照执行）。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NB = Path(r"D:\A\zfy-v5\zfy_v5_fullft.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

ROLE = [
    ("tmp_smoke", "冒烟配置 + 训练 + 验收判定"),
    ("tmp_full.json", "【第8节·上】全量训练配置 -> 生成 tmp_full.json"),
    ("nohup", "【第8节·下】启动全量训练（nohup 后台）"),
]

for i, c in enumerate(nb["cells"]):
    s = "".join(c["source"])
    if c["cell_type"] == "markdown":
        head = s.strip().split("\n")[0].strip()
        head = re.sub(r"[^\u4e00-\u9fff\w\s\.\-—（）()：:#]", "", head)[:56]
        print(f"  cell {i:2d}  [说明] {head}")
        continue
    first = s.strip().split("\n")[0][:50]
    mark = ""
    for kw, role in ROLE:
        if kw in s:
            mark = f"   <<< {role}"
            break
    if not mark and "train_full.log" in s:
        mark = "   <<< 【第8节·补充】监控进度"
    print(f"  cell {i:2d}  [代码] {first}{mark}")
