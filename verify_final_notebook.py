# -*- coding: utf-8 -*-
"""最终 Notebook 静态验证。"""
import ast
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

p = Path(__file__).with_name("zfy_v5_fullft_final.ipynb")
d = json.loads(p.read_text(encoding="utf-8"))
assert d["nbformat"] == 4
assert len(d["cells"]) == 25
for i, c in enumerate(d["cells"]):
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]), filename=f"cell-{i}")
s = "\n".join("".join(c.get("source", [])) for c in d["cells"])
for needle in (
    "apt-get", "git-lfs", '["git", "lfs", "version"]',
    '["git", "lfs", "install", "--skip-repo"]',
    "避免只得到 LFS 指针", "enc_p + ssl_proj",
    "SoVITS_weights_v5_full_e32_final",
):
    assert needle in s, needle
print(f"✓ Notebook {len(d['cells'])} cells：JSON/AST/git-lfs检测安装初始化全部通过")
