# -*- coding: utf-8 -*-
"""确认 LFS 指针的确切内容与最小修复步骤。

## 已确认

  · 远程 LFS **存储里有真实对象**（不带 SKIP_SMUDGE 克隆能拿到 766.4 MB）
  · 但仓库里的 blob 是 **134 字节的指针**
  · 说明：推送时 git-lfs **没有上传内容**（因为 SKIP_SMUDGE 阻止了接管），
    但魔搭平台随后把文件作为 LFS 对象处理并上传了？
    —— 或者：魔搭在服务端做了转换。

  无论哪种，结果是：**用普通 clone 就能拿到真实内容**。

## 本脚本

  1. 取出指针文件的确切文本（给用户看证据）
  2. 验证"最小修复"：在已有的（指针）clone 里跑 `git lfs pull` 能否修复
  3. 给出确切命令
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
REL = "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"
LOCAL = GSV / REL
LOCAL_SIZE = LOCAL.stat().st_size if LOCAL.exists() else 0

tok = (Path.home() / ".modelscope" / "credentials" / "git_token") \
    .read_text(encoding="utf-8").strip()
url = ("https://oauth2:" + tok + "@www.modelscope.cn/datasets/"
       "yangwenjie1231/zfy-v5-fullft-pkg.git")


def run(cmd, cwd, env=None, timeout=900):
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, cwd=str(cwd), env=e, timeout=timeout,
                       capture_output=True)
    out = r.stdout.decode("utf-8", "replace") if r.stdout else ""
    err = r.stderr.decode("utf-8", "replace") if r.stderr else ""
    return r.returncode, out, err


tmp = Path(tempfile.mkdtemp(prefix="ms_ptr_"))
try:
    print("=" * 74)
    print("1. 模拟「跳过 LFS」的 clone（复现用户遇到的情况）")
    print("=" * 74)
    code, out, err = run(
        ["git", "clone", "--depth", "1", url, "skip"],
        tmp, env={"GIT_LFS_SKIP_SMUDGE": "1"})
    print(f"  退出码: {code}")
    p_skip = tmp / "skip" / REL
    if p_skip.exists():
        sz = p_skip.stat().st_size
        raw = p_skip.read_bytes()
        print(f"  体积: {sz} 字节")
        print("  内容:")
        for line in raw.decode("utf-8", "replace").split("\n")[:4]:
            if line.strip():
                print(f"    {line}")

    print()
    print("=" * 74)
    print("2. 在指针 clone 里跑 git lfs pull（最小修复）")
    print("=" * 74)
    code, out, err = run(["git", "lfs", "install", "--local"], tmp / "skip")
    print(f"  lfs install --local: 退出码 {code}")
    code, out, err = run(["git", "lfs", "pull"], tmp / "skip")
    print(f"  lfs pull: 退出码 {code}")
    if err.strip():
        print(f"  stderr: {err.strip()[-600:]}")
    if p_skip.exists():
        sz2 = p_skip.stat().st_size
        print(f"  pull 后体积: {sz2/1e6:.1f} MB")
        if sz2 > 1e6:
            print(f"  ✓ 修复成功（本地真实文件 {LOCAL_SIZE/1e6:.1f} MB）")
        else:
            print("  ✗ 仍是指针")

    print()
    print("=" * 74)
    print("3. 结论与修复命令")
    print("=" * 74)
    print(f"""
  远程 LFS 存储**有**真实对象（普通 clone 能拿到 766.4 MB）。
  你之前遇到的指针，是因为 clone 时跳过了 LFS 内容
  （可能用了 GIT_LFS_SKIP_SMUDGE=1，或环境未装 git-lfs）。

  云上修复（在已有的仓库目录里执行）：
      cd /mnt/workspace/zfy-v5-fullft-pkg
      git lfs install --local
      git lfs pull

  验证（体积应从 134 字节变成真实大小）：
      ls -la GPT_SoVITS/pretrained_models/gsv-v5-pretrained/
      # s2Gv5dev.pth 应为 ~766 MB（不是 134 字节）
      du -sh GPT_SoVITS/pretrained_models/

  若 git lfs pull 无效，就删掉重 clone（**不要**加 SKIP_SMUDGE）：
      cd /mnt/workspace
      rm -rf zfy-v5-fullft-pkg
      git lfs install
      git clone https://oauth2:<token>@www.modelscope.cn/datasets/yangwenjie1231/zfy-v5-fullft-pkg.git
""")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
