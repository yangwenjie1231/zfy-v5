#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：装了 git-lfs 后 `git status` 仍有 2561 项改动，是真是假？

## 背景
用户装好 git-lfs 3.0.2 后，`git status --porcelain | wc -l` 仍是 **2561**。
预期：若是 LFS 假象，装完 lfs 应该大幅减少。

## 两种可能
  A. **真改动** —— 那 2500+ 文件内容确实和 HEAD 不同
     （补丁改了它们 / LFS 拉取时内容变了 / 时间戳无关）
  B. **仍是假象** —— git-lfs 装了但没生效（比如 lfs.fetchexclude、
     filter 配置、或 smudge 未跑）

## 关键判据（本脚本）
对抽样文件同时算三个 hash：
  · 工作区文件的 git blob（经 clean filter）
  · 索引里的 blob（`git ls-files -s`）
  · HEAD 里的 blob（`git rev-parse HEAD:<path>`）
三者关系能直接区分 A / B。

另外对 LFS 文件，直接看**文件头**：
  · 以 `version https://git-lfs` 开头 = 指针（内容丢了）
  · 否则 = 实体文件
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path("/mnt/workspace/zfy-v5-fullft-pkg")
if not (ROOT / ".git").is_dir():
    ROOT = Path.cwd()


def git(*a, timeout=300):
    return subprocess.run(["git", *a], cwd=str(ROOT), capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=timeout)


print("=" * 80)
print(f"仓库: {ROOT}")
print("=" * 80)

print()
print("1) git-lfs 与 filter 配置")
for k in ("filter.lfs.clean", "filter.lfs.smudge", "filter.lfs.process",
          "filter.lfs.required", "lfs.fetchexclude"):
    r = git("config", "--get", k)
    print(f"  {k:24s} = {(r.stdout or '').strip()!r}")

print()
print("2) 改动总数与分类")
r = git("status", "--porcelain")
lines = [l for l in (r.stdout or "").split("\n") if l.strip()]
print(f"  总改动: {len(lines)}")
from collections import Counter
codes = Counter(l[:2] for l in lines)
print(f"  状态码分布: {dict(codes)}")

# 按顶层目录分组
grp = Counter()
for l in lines:
    p = l[3:].strip().strip('"')
    grp[p.split("/")[0] if "/" in p else p] += 1
print(f"  目录分布: {dict(grp.most_common(10))}")

print()
print("3) 抽样：三个 hash 对比（区分真改动 vs 假象）")
SAMPLES = [
    "GPT_SoVITS/s1_train.py",
    "GPT_SoVITS/configs/s2.json",
    "logs/zfy_v5/2-name2text.txt",
    "GPT_SoVITS/text/G2PWModel/g2pW.onnx",
    "_tools/train_gpt.py",
]
for rel in SAMPLES:
    p = ROOT / rel
    if not p.exists():
        print(f"  {rel}: 工作区不存在")
        continue
    sz = p.stat().st_size
    head = p.read_bytes()[:40]
    is_ptr = head.startswith(b"version https://git-lfs")
    h_work = git("hash-object", "--", rel).stdout.strip()
    h_head = git("rev-parse", f"HEAD:{rel}").stdout.strip()
    idx = git("ls-files", "-s", "--", rel).stdout.strip()
    h_idx = idx.split()[1] if idx.split() else "?"
    attr = git("check-attr", "filter", "--", rel).stdout.strip()
    lfs = "lfs" in attr
    print(f"  {rel}")
    print(f"    大小 {sz:>10d}  {'**LFS 指针**' if is_ptr else '实体文件'}  "
          f"{'[LFS]' if lfs else '[普通]'}")
    print(f"    work={h_work[:10]}  idx={h_idx[:10]}  HEAD={h_head[:10]}")
    verdict = "相同" if h_work == h_head else "**不同**"
    print(f"    work vs HEAD: {verdict}")

print()
print("=" * 80)
print("4) 结论")
print("=" * 80)
print("""
  判读方法：
  · 若抽样里 work == HEAD（相同）-> 那些 M 是**假象**（时间戳/stat 缓存问题）
      解决：git update-index --refresh  或  git add --renormalize .
  · 若抽样里 work != HEAD（不同）-> 是**真改动**
      解决：这些文件确实被改过（补丁 / LFS 内容变化）
            直接 git restore --source=HEAD --staged --worktree <非LFS文件>
            或干脆**无视**（不影响训练/合成，只要不 commit）
""")
