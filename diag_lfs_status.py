#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断云端那 2500+ 项 `M` 是真改动还是 git-lfs 缺失的**假象**。

## 现象
`resolve_git_conflict.py --apply` 后 `git status` 仍列出 2500+ 项 `M`，
包括：
    GPT_SoVITS/configs/s2.json
    GPT_SoVITS/text/G2PWModel/g2pW.onnx
    GPT_SoVITS/text/engdict_cache.pickle
    logs/zfy_v5/3-bert/*.pt
    logs/zfy_v5/2-name2text.txt

## 根因假设
ModelScope 自动生成的 `.gitattributes`（89 条规则）**把 .json/.txt/.pickle
也划进了 LFS**（实测 `git check-attr filter` 返回 `lfs`）。

云端实例若**没装 git-lfs**（或没 `git lfs install`），
`git add`/`git status` 无法把工作区文件"清洗"成 LFS 指针，
于是 git 认为每个 LFS 文件都"被改过" -> 全列成 M。
**内容其实没变**，是工具链缺失造成的假象。

## 本脚本
  1. 检查 git-lfs 是否可用
  2. 抽样比对「工作区文件」与「HEAD 里的内容」是否真的不同
     （用 git hash-object 算工作区 blob，与 HEAD 的 blob 比）
  3. 对 LFS 文件，比较**指针文本**而非内容
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path.cwd()
for c in (ROOT, Path("/mnt/workspace/zfy-v5-fullft-pkg")):
    if (c / ".git").is_dir():
        ROOT = c
        break


def run(args, timeout=300):
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=timeout)


print("=" * 78)
print(f"仓库: {ROOT}")
print("=" * 78)

print()
print("1) git-lfs 是否可用")
r = run(["lfs", "version"])
print(f"  git lfs version -> exit {r.returncode}")
print(f"    {(r.stdout or r.stderr or '').strip()[:120]}")
r2 = subprocess.run(["git", "lfs", "env"], cwd=str(ROOT),
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace")
print(f"  git lfs env -> exit {r2.returncode}")
for l in (r2.stdout or "").split("\n")[:6]:
    if l.strip():
        print(f"    {l.strip()[:100]}")

print()
print("2) .gitattributes 里哪些常见扩展名走了 LFS")
attr = ROOT / ".gitattributes"
if attr.exists():
    txt = attr.read_text(encoding="utf-8", errors="replace")
    lines = [l.strip() for l in txt.split("\n") if l.strip()
             and not l.startswith("#")]
    print(f"  规则数: {len(lines)}")
    interesting = [l for l in lines
                   if any(e in l for e in (".json", ".txt", ".pickle", ".py",
                                           ".yaml", ".onnx", ".pt", ".pth"))]
    for l in interesting[:12]:
        print(f"    {l[:88]}")
else:
    print("  无 .gitattributes")

print()
print("3) 抽样：工作区 vs HEAD 是否**真的**不同")
# 取 git status 里的一些 M 文件
st = run(["status", "--porcelain"])
mods = [l[3:].strip().strip('"') for l in (st.stdout or "").split("\n")
        if l.startswith(" M ") or l.startswith("M ")]
print(f"  status 报告 {len(mods)} 个修改")

SAMPLES = [
    "GPT_SoVITS/configs/s2.json",
    "GPT_SoVITS/text/G2PWModel/g2pW.onnx",
    "GPT_SoVITS/text/engdict_cache.pickle",
    "logs/zfy_v5/2-name2text.txt",
]
for rel in SAMPLES:
    p = ROOT / rel
    if not p.exists():
        print(f"  {rel}: 工作区不存在")
        continue
    # 工作区文件的 git 对象 hash（会经过 clean filter）
    h_work = run(["hash-object", "--", rel]).stdout.strip()
    # HEAD 里的对象 hash
    h_head = run(["rev-parse", f"HEAD:{rel}"]).stdout.strip()
    same = h_work == h_head
    sz = p.stat().st_size
    # 判断工作区文件是"实体"还是"指针"
    head2 = p.read_bytes()[:40] if sz < 400 else b""
    is_pointer = head2.startswith(b"version https://git-lfs")
    print(f"  {rel}")
    print(f"    大小 {sz/1e6:.3f} MB   "
          f"{'**LFS 指针**' if is_pointer else '实体文件'}")
    print(f"    工作区 blob {h_work[:12]}   HEAD blob {h_head[:12]}   "
          f"{'相同' if same else '**不同**'}")

print()
print("=" * 78)
print("4) 结论与处理")
print("=" * 78)
print("""
  若 `git lfs version` 失败 / exit != 0：
    **云端没装 git-lfs** -> 那 2500+ 项 M 全是假象。
    处理（二选一）：

    A. 装 git-lfs（推荐，一劳永逸）
         apt-get update -qq && apt-get install -y -qq git-lfs
         git lfs install --skip-smudge
      装完 `git status` 应立刻变干净（因为能正确 clean 了）。

    B. 临时忽略（只想赶紧跑合成）
         git update-index --assume-unchanged <文件...>
       或直接无视 —— **合成和训练都不受影响**，
       只要不 commit/push 就行。

  ⚠️ 千万不要 `git add -A && git commit` ——
     在没装 lfs 的机器上会把 LFS 指针替换成**实体文件**提交，
     仓库会瞬间膨胀且可能被 ModelScope 拒收。
""")
