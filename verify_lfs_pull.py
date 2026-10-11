# -*- coding: utf-8 -*-
"""验证：LFS 指针问题能否通过 git lfs pull 修复？

## 事故根因（已确认）

远程 4 个模型文件都是 **134 字节的 LFS 指针**，不是真实内容。
原因：我推送时用的临时 clone 带了 `GIT_LFS_SKIP_SMUDGE=1`
（本意是跳过旧的 2.77GB zip），但这个变量**同时让 git-lfs 不接管新文件**，
于是 .pth 被当作普通小文本提交。

而魔搭平台自己生成了 .gitattributes（含 *.pth / *.ckpt / *.bin 等规则），
所以 clone 端会把这些路径**当 LFS 对象**去拉 —— 但远程 LFS 存储里
根本没有对应的对象（因为推送时没上传），于是留下指针。

## 本脚本验证

  1. 远程 LFS 存储里到底有没有这些对象（能否 pull 到）
  2. 若没有，则需要**重新推送**（这次启用 LFS）
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
BIG = "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"

tok = (Path.home() / ".modelscope" / "credentials" / "git_token") \
    .read_text(encoding="utf-8").strip()
url = ("https://oauth2:" + tok + "@www.modelscope.cn/datasets/"
       "yangwenjie1231/zfy-v5-fullft-pkg.git")

tmp = Path(tempfile.mkdtemp(prefix="ms_pull_"))
try:
    print("=" * 74)
    print("1. 不带 SKIP_SMUDGE 克隆（让 LFS 正常接管）")
    print("=" * 74)
    env = os.environ.copy()
    env.pop("GIT_LFS_SKIP_SMUDGE", None)     # 关键：这次不跳过
    r = subprocess.run(
        ["git", "clone", "--depth", "1", url, "r"],
        cwd=str(tmp), capture_output=True, text=True, env=env, timeout=900)
    print(f"  clone 退出码: {r.returncode}")
    if r.returncode != 0:
        print(f"  输出: {(r.stdout + r.stderr)[-800:]}")

    repo = tmp / "r"
    if repo.exists():
        p = repo / BIG
        if p.exists():
            sz = p.stat().st_size
            head = p.read_bytes()[:80]
            is_ptr = head.startswith(b"version https://git-lfs")
            print(f"\n  {Path(BIG).name}: {sz/1e6:.1f} MB")
            print(f"  是 LFS 指针: {is_ptr}")
            if is_ptr:
                print(f"  指针内容: {head.decode('utf-8','replace')[:120]}")
                print()
                print("  => clone 后仍是指针，说明**远程 LFS 存储里没有对象**")
            else:
                print("  => 拿到了真实内容")

        print()
        print("=" * 74)
        print("2. 尝试 git lfs pull")
        print("=" * 74)
        r2 = subprocess.run(["git", "lfs", "pull"],
                            cwd=str(repo), capture_output=True, text=True,
                            env=env, timeout=900)
        print(f"  退出码: {r2.returncode}")
        print((r2.stdout + r2.stderr)[-1200:])
        if p.exists():
            print(f"\n  pull 后 {Path(BIG).name}: {p.stat().st_size/1e6:.1f} MB")

        print()
        print("=" * 74)
        print("3. 远程 LFS 对象清单（判断对象是否存在）")
        print("=" * 74)
        r3 = subprocess.run(["git", "lfs", "ls-files", "-l"],
                            cwd=str(repo), capture_output=True, text=True,
                            env=env)
        lines = [l for l in r3.stdout.split("\n") if l.strip()]
        print(f"  远程登记的 LFS 文件数: {len(lines)}")
        for l in lines[:8]:
            print(f"    {l[:100]}")
        missing = [l for l in lines if "*" in l or "?" in l]
        if missing:
            print(f"\n  !! 有 {len(missing)} 个对象缺失（标记 * 或 ?）")
        else:
            print("\n  所有 LFS 对象状态正常")

finally:
    keep = os.environ.get("KEEP_TMP")
    if keep:
        print(f"\n保留临时目录: {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
