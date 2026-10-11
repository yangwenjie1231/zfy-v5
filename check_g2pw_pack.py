# -*- coding: utf-8 -*-
"""检查 G2PW 模型文件是否真的在打包清单里（云端 onnxruntime 报错的关联项）。

## 背景
云端合成报：
    ModuleNotFoundError: No module named 'onnxruntime'
来源：chinese2.py:29-31 无条件 import text.g2pw
      -> g2pw/onnx_api.py:11 模块级 import onnxruntime

而 G2PW 是**中文多音字消歧**，用 620MB 的 g2pW.onnx。
它需要：onnxruntime + G2PWModel/ 里的模型文件。

所以要么装 onnxruntime（+620MB 模型），要么关掉 G2PW。
本脚本先确认模型文件在不在包里。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, r"D:\A\zfy-v5")
import pack_for_cloud as P  # noqa: E402

rels = {rel.replace("\\", "/") for _, rel in P.collect()}

print("=" * 78)
print("1. G2PWModel 目录：本地 vs 打包清单")
print("=" * 78)
GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
d = GSV / "GPT_SoVITS" / "text" / "G2PWModel"
local = sorted(f for f in d.rglob("*")
               if f.is_file() and "__pycache__" not in f.parts)
print(f"  本地 {len(local)} 个文件:")
for f in local:
    rel = str(f.relative_to(GSV)).replace("\\", "/")
    inpack = rel in rels
    sz = f.stat().st_size
    print(f"    {'OK  ' if inpack else '**缺**'} {f.name:42s} "
          f"{sz/1e6:9.1f} MB")

print()
print("=" * 78)
print("2. 为什么被排除？逐条测试 EXCLUDE_PATTERNS")
print("=" * 78)
targets = ["GPT_SoVITS/text/G2PWModel/g2pW.onnx",
           "GPT_SoVITS/text/G2PWModel/char_bopomofo_dict.json",
           "GPT_SoVITS/text/G2PWModel/version"]
for t in targets:
    hits = [pat for pat in P.EXCLUDE_PATTERNS if pat in t]
    print(f"  {t}")
    print(f"     命中排除规则: {hits if hits else '（无）'}")
    print(f"     _excluded() = {P._excluded(t)}")

print()
print("=" * 78)
print("3. 关键：G2PW 是必需的吗")
print("=" * 78)
c2 = GSV / "GPT_SoVITS" / "text" / "chinese2.py"
src = c2.read_text(encoding="utf-8", errors="replace")
lines = src.split("\n")
for i, l in enumerate(lines, 1):
    if "is_g2pw" in l:
        print(f"  chinese2.py:{i}  {l.strip()[:88]}")

print()
print("=" * 78)
print("4. 结论")
print("=" * 78)
print("""
  chinese2.py:28  `is_g2pw = True` 是**硬编码**的（上面两行是注释掉的
  环境变量读取），所以 G2PW 一定会被 import。

  两条路：
    A. 装 onnxruntime + 把 G2PWModel 模型文件补进包（+620MB）
    B. 把 is_g2pw 改成读环境变量，默认关闭（省 620MB + 一个依赖）
       —— 但会损失多音字消歧精度（如「长」「重」「行」）

  注意：G2PW 用 onnxruntime 跑 620MB 模型，**每次 import 都要加载**。
  对纯中文短句合成，多音字错误的概率不高，但要实测。
""")
