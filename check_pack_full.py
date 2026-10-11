# -*- coding: utf-8 -*-
"""列出 GPT_SoVITS/ 下所有顶层 .py，标出哪些没打包。

上次只检查了目录，漏掉了 `GPT_SoVITS/sv.py` 这种**顶层单文件**。
这次一并查全，避免再补一轮。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, r"D:\A\zfy-v5")
import pack_for_cloud as P  # noqa: E402

rels = set(rel.replace("\\", "/") for _, rel in P.collect())
GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")

print("=" * 70)
print("GPT_SoVITS/ 顶层 .py")
print("=" * 70)
missing_files = []
for f in sorted((GSV / "GPT_SoVITS").glob("*.py")):
    rel = f"GPT_SoVITS/{f.name}"
    ok = rel in rels
    if not ok:
        missing_files.append(rel)
    print(f"  {'OK  ' if ok else '缺失'} {rel}")

print()
print("=" * 70)
print("GPT_SoVITS/ 顶层目录")
print("=" * 70)
missing_dirs = []
for d in sorted((GSV / "GPT_SoVITS").iterdir()):
    if not d.is_dir() or d.name == "__pycache__":
        continue
    rel = f"GPT_SoVITS/{d.name}"
    n = sum(1 for r in rels if r.startswith(rel + "/"))
    if n == 0:
        missing_dirs.append(rel)
    sz = sum(x.stat().st_size for x in d.rglob("*") if x.is_file())
    print(f"  {'OK  ' if n else '缺失'} {rel:34s} {n:5d} 文件  {sz/1e6:8.2f} MB")

print()
print("=" * 70)
print("tools/ 顶层 .py 与目录")
print("=" * 70)
tm = []
for p in sorted((GSV / "tools").iterdir()):
    if p.name == "__pycache__":
        continue
    if p.is_dir():
        rel = f"tools/{p.name}"
        n = sum(1 for r in rels if r.startswith(rel + "/"))
    else:
        rel = f"tools/{p.name}"
        n = 1 if rel in rels else 0
    if n == 0:
        tm.append(rel)
    print(f"  {'OK  ' if n else '缺失'} {rel}")

print()
print("=" * 70)
print("需要补进 pack 的条目")
print("=" * 70)
print("  CODE_DIRS 追加:")
for d in missing_dirs:
    print(f'      "{d}",')
print("  CODE_FILES 追加:")
for f in missing_files:
    print(f'      "{f}",')
print("  tools 缺失（如非必需可忽略）:", tm or "无")
