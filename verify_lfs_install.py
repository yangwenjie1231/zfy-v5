#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`git lfs install` 之后的验收：确认状态变干净、且**不会**误下 3.5GB。

## 背景
云端跑了 `git lfs install`（输出 `Git LFS initialized.`），
说明 git-lfs 二进制已就位。但要确认三件事：

  1. 之前那 2500+ 项 `M` 是否消失（假象应消失）
  2. **smudge 是否被关掉** —— 否则以后任何 checkout/pull 都会去下载
     3.5 GB 的 LFS 内容（这个仓库里 .pth/.ckpt/.pt/.onnx/.wav 全是 LFS）
  3. 工作区里的实体文件是否**还完好**（没被指针覆盖）

## 关于 smudge
`git lfs install` 默认会同时设置 **clean（提交时转指针）** 和
**smudge（检出时下载内容）**。我们只需要 clean。
关掉 smudge 的两种方式：
  · 进程级：export GIT_LFS_SKIP_SMUDGE=1
  · 仓库级：git config --local lfs.fetchexclude '*'   （或 filter.lfs.smudge=）
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
print("1) git-lfs 版本与仓库绑定")
r = run(["lfs", "version"])
print(f"  git lfs version -> exit {r.returncode}  "
      f"{(r.stdout or '').strip()[:60]}")

print()
print("2) filter 配置（关键：smudge 是否还开着）")
for key in ("filter.lfs.clean", "filter.lfs.smudge", "filter.lfs.process",
            "filter.lfs.required"):
    for scope, flag in (("local", "--local"), ("global", "--global")):
        rr = run(["config", flag, "--get", key])
        v = (rr.stdout or "").strip()
        if v or rr.returncode == 0:
            print(f"  [{scope:6s}] {key:24s} = {v!r}")

print()
print("3) status 是否变干净（之前 2500+ 项 M）")
st = run(["status", "--porcelain"])
lines = [l for l in (st.stdout or "").split("\n") if l.strip()]
print(f"  剩余改动: {len(lines)} 项")
for l in lines[:15]:
    print(f"    {l[:100]}")
if len(lines) > 15:
    print(f"    … 还有 {len(lines)-15} 项")

# 统计按目录分组
from collections import Counter
grp = Counter()
for l in lines:
    p = l[3:].strip().strip('"')
    grp[p.split("/")[0] if "/" in p else p] += 1
if grp:
    print(f"  分组: {dict(grp.most_common(8))}")

print()
print("4) 工作区实体文件是否完好（抽查几个大文件）")
CHECK = [
    "GPT_SoVITS/text/G2PWModel/g2pW.onnx",
    "ref/zfh-e4.ckpt",
    "ref/v2pro_sovits.pth",
    "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth",
    "ref/default.wav",
]
for rel in CHECK:
    p = ROOT / rel
    if not p.exists():
        print(f"  {rel}: **不存在**")
        continue
    sz = p.stat().st_size
    head = p.read_bytes()[:40]
    is_ptr = head.startswith(b"version https://git-lfs")
    if is_ptr:
        print(f"  {rel}: **LFS 指针**（{sz} 字节）<- 内容被覆盖了！")
    else:
        print(f"  {rel}: 实体文件 {sz/1e6:.1f} MB  OK")

print()
print("5) lfs.fetchexclude / skip-smudge 现状")
for key in ("lfs.fetchexclude", "lfs.skipdownloaderrors"):
    rr = run(["config", "--local", "--get", key])
    print(f"  {key} = {(rr.stdout or '').strip()!r}")

print()
print("=" * 78)
print("结论与建议")
print("=" * 78)
n_mod = len(lines)
print(f"""
  剩余改动 {n_mod} 项。
""")
if n_mod == 0:
    print("  ✓ 状态干净 —— 之前的 M 确实全是 lfs 假象，装完就消失了。")
else:
    print("  ⚠️ 仍有改动。若这些是 .json/.txt/.pickle 等 LFS 文件，")
    print("     可能是 index 里存的还是实体 blob（pull 时没经过 clean）。")
    print("     无害 —— 只要不 commit 就行。想清掉：")
    print("       git add --renormalize .        # 重新按 .gitattributes 规范化")
    print("       git status --short             # 应显著变少")
    print("     ⚠️ 但 add --renormalize 会把 LFS 文件改成指针，")
    print("        若之后 commit 就会提交指针（这是**正确**行为）")

print("""
  ### ⚠️ 必须关掉 smudge，否则会误下 3.5 GB
  仓库里 .pth/.ckpt/.pt/.onnx/.wav 全是 LFS，总计约 3.5 GB。
  `git lfs install` 默认**开着** smudge -> 以后任何 checkout/pull 都会下载。

  建议（在云端执行一次，仓库级）：
      git config --local lfs.fetchexclude '*'
      git config --local filter.lfs.smudge "git-lfs smudge --skip -- %f"
      git config --local filter.lfs.process "git-lfs filter-process --skip"

  或者每次操作前：
      export GIT_LFS_SKIP_SMUDGE=1

  ### 现在可以直接跑合成了
      python _tools/cloud_synth_sweep.py \\
          --weights-dir SoVITS_weights_v5_full_e32 --epochs 16 --no-v2pro
""")
