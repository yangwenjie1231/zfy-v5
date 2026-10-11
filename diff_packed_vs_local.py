# -*- coding: utf-8 -*-
"""系统性比对：本地 GPT-SoVITS 检出 vs 已上传的远端仓库。

## 背景
云端跑合成报：
    ModuleNotFoundError: No module named 'BigVGAN'

原因：`pack_for_cloud.py` 的 CODE_DIRS 是为了**训练**裁剪的，
没有包含 `GPT_SoVITS/BigVGAN`（只有推理才需要）。

与其一个个补，不如**整体 diff 一次**，把所有缺失目录/文件找出来。

同时检查推理链路真正 import 了哪些东西。
"""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"

sys.path.insert(0, str(V5))
import pack_for_cloud as P  # noqa: E402

os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"

print("=" * 78)
print("1. 本地检出根下的顶层条目")
print("=" * 78)
local = []
for p in sorted(GSV.iterdir()):
    if p.name in (".git", "__pycache__"):
        continue
    local.append((p.name, p.is_dir(), _sz := (
        sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        if p.is_dir() else p.stat().st_size)))
for name, isdir, sz in local:
    print(f"  {'dir ' if isdir else 'file'} {name:40s} {sz/1e6:9.2f} MB")

print()
print("=" * 78)
print("2. GPT_SoVITS/ 下的子目录（推理/训练都用这里）")
print("=" * 78)
gsub = [p for p in sorted((GSV / "GPT_SoVITS").iterdir())
        if p.is_dir() and p.name != "__pycache__"]
print(f"  共 {len(gsub)} 个:")
for p in gsub:
    sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    print(f"    {p.name:28s} {sz/1e6:9.2f} MB")

print()
print("=" * 78)
print("3. 哪些**没有**被打包")
print("=" * 78)
items = P.collect()
packed = set(rel.replace("\\", "/") for _, rel in items)
# 从打包清单推断已覆盖的顶层目录
covered = set()
for r in packed:
    covered.add(r.split("/")[0])
    parts = r.split("/")
    if len(parts) >= 2:
        covered.add("/".join(parts[:2]))

missing = []
for name, isdir, sz in local:
    key = name
    if key not in covered:
        missing.append((name, isdir, sz, "顶层条目未覆盖"))
for p in gsub:
    key = f"GPT_SoVITS/{p.name}"
    if key not in covered:
        sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        missing.append((key, True, sz, "GPT_SoVITS 子目录未覆盖"))

if missing:
    for name, isdir, sz, why in missing:
        print(f"  [缺] {name:44s} {sz/1e6:8.2f} MB  ({why})")
else:
    print("  无缺失")

print()
print("=" * 78)
print("4. 推理链路真正 import 的第三方/本地模块")
print("=" * 78)


def imports_of(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module)
    return out


ENTRY = GSV / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py"
seen, queue, localmods, third = set(), [ENTRY], set(), set()
ROOTS = [GSV, GSV / "GPT_SoVITS", GSV / "tools",
         GSV / "GPT_SoVITS" / "module", GSV / "GPT_SoVITS" / "AR"]
STD = {"os", "sys", "re", "json", "math", "time", "random", "logging",
       "warnings", "traceback", "typing", "pathlib", "collections",
       "functools", "itertools", "subprocess", "shutil", "glob", "copy",
       "hashlib", "io", "struct", "argparse", "platform", "datetime",
       "contextlib", "dataclasses", "enum", "abc", "inspect", "textwrap",
       "unicodedata", "ctypes", "threading", "queue", "tempfile", "uuid",
       "base64", "urllib", "socket", "string", "operator", "pickle",
       "importlib", "gc", "signal", "stat", "errno", "csv", "zipfile"}


def resolve(mod: str) -> Path | None:
    parts = mod.split(".")
    for root in ROOTS:
        p = root.joinpath(*parts)
        for c in (p.with_suffix(".py"), p / "__init__.py"):
            if c.exists():
                return c
    if len(parts) == 1:
        for root in ROOTS:
            c = root / (parts[0] + ".py")
            if c.exists():
                return c
    return None


while queue:
    f = queue.pop()
    if f in seen or not f.exists():
        continue
    seen.add(f)
    for mod in imports_of(f):
        top = mod.split(".")[0]
        if top in STD:
            continue
        rp = resolve(mod)
        if rp is not None:
            localmods.add(top)
            queue.append(rp)
        else:
            third.add(top)

print(f"  追踪了 {len(seen)} 个本地文件")
print(f"  本地模块: {sorted(localmods)}")
print(f"  第三方  : {sorted(third)}")

print()
print("=" * 78)
print("5. 建议加入 CODE_DIRS 的条目")
print("=" * 78)
need = [m for m, isd, sz, why in missing if isd]
for m in need:
    print(f'    "{m}",')
