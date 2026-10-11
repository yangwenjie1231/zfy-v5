# -*- coding: utf-8 -*-
"""诊断：云上 pretrained_models 里的 .pth 是真文件还是 LFS 指针。

症状（用户实测）：
    RuntimeError: PytorchStreamReader failed reading zip archive:
    failed finding central directory

这是 torch.load 读到**文本指针**（而非二进制权重）的典型报错。

## 怀疑

我的仓库**没有 .gitattributes**（已核实：`git show HEAD:.gitattributes` 报不存在），
所以推送时 .pth 未按 LFS 处理。但魔搭平台可能自己生成了 .gitattributes，
导致 clone 端把大文件当 LFS 对象、却拿不到内容 -> 留下指针文本。

## 本脚本

不猜测，直接查远程仓库：
  1. 远程是否有 .gitattributes，内容是什么
  2. 远程的 .pth 是 LFS 对象还是普通 blob（看 git ls-tree 的 size）
  3. 本地文件与远程 blob 的 sha256 是否一致
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
TARGETS = [
    "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth",
    "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth",
    "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin",
]

tok = (Path.home() / ".modelscope" / "credentials" / "git_token") \
    .read_text(encoding="utf-8").strip()
url = ("https://oauth2:" + tok + "@www.modelscope.cn/datasets/"
       "yangwenjie1231/zfy-v5-fullft-pkg.git")

tmp = Path(tempfile.mkdtemp(prefix="ms_lfs_"))
try:
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    r = subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", url, "r"],
        cwd=str(tmp), capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print("clone 失败:", r.stderr[-400:])
        sys.exit(1)
    repo = tmp / "r"

    def git(*a):
        return subprocess.run(["git", *a], cwd=str(repo), capture_output=True,
                              text=True, encoding="utf-8",
                              errors="replace").stdout

    print("=" * 74)
    print("1. 远程是否有 .gitattributes（决定 .pth 是否走 LFS）")
    print("=" * 74)
    ga = repo / ".gitattributes"
    if ga.exists():
        print("  存在，内容：")
        for line in ga.read_text(encoding="utf-8", errors="replace").split("\n"):
            if line.strip():
                print(f"    {line}")
    else:
        print("  **不存在**")

    print()
    print("=" * 74)
    print("2. 远程 .pth 是 LFS 对象还是普通 blob")
    print("=" * 74)
    out = git("ls-tree", "-r", "-l", "HEAD")
    info = {}
    for line in out.split("\n"):
        parts = line.split()
        if len(parts) >= 5:
            size, path = parts[3], parts[4]
            info[path] = size
    for t in TARGETS:
        sz = info.get(t, "?")
        try:
            n = int(sz)
            # LFS 指针文件约 130 字节
            kind = ("**LFS 指针**（约130字节）" if n < 300
                    else f"实际内容 {n/1e6:.1f} MB")
        except ValueError:
            kind = "?"
        print(f"  {Path(t).name:34s} {sz:>12s} 字节  {kind}")

    print()
    print("=" * 74)
    print("3. 本地文件 sha256（对照用）")
    print("=" * 74)
    for t in TARGETS:
        p = GSV / t.replace("GPT_SoVITS/", "GPT_SoVITS/", 1)
        if not p.exists():
            print(f"  {Path(t).name:34s} 本地不存在")
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        print(f"  {Path(t).name:34s} {p.stat().st_size/1e6:8.1f} MB  {h}")

    print()
    print("=" * 74)
    print("4. 结论")
    print("=" * 74)
    lfs_ptrs = [t for t in TARGETS
                if info.get(t, "999999").isdigit()
                and int(info.get(t, "999999")) < 300]
    if lfs_ptrs:
        print(f"  {len(lfs_ptrs)} 个模型文件在远程是 **LFS 指针**：")
        for t in lfs_ptrs:
            print(f"    {t}")
        print()
        print("  => clone 时若没拉 LFS，torch.load 会读到指针文本并报")
        print("     'failed finding central directory'")
        print()
        print("  修复（云上执行）：")
        print("    cd /mnt/workspace/zfy-v5-fullft-pkg")
        print("    git lfs install --local")
        print("    git lfs pull")
        print("    # 验证体积")
        print("    ls -la GPT_SoVITS/pretrained_models/gsv-v5-pretrained/")
    else:
        print("  未发现 LFS 指针，模型文件是真实内容。")
        print("  那么报错可能是别的原因（传输损坏），建议重新 clone。")
finally:
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
