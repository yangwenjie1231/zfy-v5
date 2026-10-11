# -*- coding: utf-8 -*-
"""抽出 shell 脚本里 heredoc 包着的 Python，逐个做语法检查。

因为本地 WSL 没有 python，不能整体干跑，所以退一步：
把每个 `python - <<'PYEOF' ... PYEOF` 块抽出来用 Windows 的 python 编译。
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")
SCRIPTS = ["fix_deps_patch.sh", "install_all_deps.sh", "recover_instance.sh",
           "install_infer_deps.sh", "install_ffmpeg.sh", "install_deps_v3.sh",
           "setup_venv.sh", "install_deps_minimal.sh", "run_sovits_e32.sh"]

BLOCK = re.compile(r"<<'?(\w+)'?\s*\n(.*?)\n\1", re.S)

bad = 0
for name in SCRIPTS:
    p = V5 / name
    if not p.exists():
        print(f"  --   {name} 不存在")
        continue
    txt = p.read_text(encoding="utf-8", errors="replace")
    blocks = BLOCK.findall(txt)
    if not blocks:
        print(f"  --   {name} 无 heredoc")
        continue
    n_ok = 0
    for i, (tag, body) in enumerate(blocks, 1):
        if tag.lower() not in ("pyEOF".lower(), "python", "py"):
            continue
        try:
            ast.parse(body)
            n_ok += 1
        except SyntaxError as e:
            bad += 1
            print(f"  FAIL {name} 第{i}块 ({tag}) 行{e.lineno}: {e.msg}")
            lines = body.split("\n")
            lo = max(0, e.lineno - 2)
            for l in lines[lo:lo + 3]:
                print(f"        {l}")
    print(f"  OK   {name}  内嵌 python 块 {n_ok} 个语法正确"
          if n_ok else f"  --   {name} 无可检块")

print()
print("结论:", "✓ 全部内嵌 Python 语法正确" if bad == 0
      else f"✗ {bad} 个块有语法错误")
sys.exit(1 if bad else 0)
