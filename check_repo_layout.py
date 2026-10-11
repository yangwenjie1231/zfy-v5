# -*- coding: utf-8 -*-
"""核对云仓库里 data/ 与 logs/ 的相对位置，确保 train_v5.py 的路径推导正确。

本机布局（开发时）：
    D:\A\zfy-v5\GPT-SoVITS\        <- 检出根
    D:\A\zfy-v5\data\zfy_v5\       <- 数据（检出根之外！）

云仓库布局（我打包时铺开的）：
    <clone>/                       <- 检出根
    <clone>/data/zfy_v5/           <- 数据（**在检出根之内**）
    <clone>/logs/zfy_v5/           <- 特征（**在检出根之内**）

这个差异会让 train_v5.py 在本机能跑、在云上找不到文件（或反之）。
本脚本实测云仓库结构并给出结论。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

tok_p = Path.home() / ".modelscope" / "credentials" / "git_token"
tok = tok_p.read_text(encoding="utf-8").strip()
url = (f"https://oauth2:{tok}@www.modelscope.cn/datasets/"
       f"yangwenjie1231/zfy-v5-fullft-pkg.git")

tmp = Path(tempfile.mkdtemp(prefix="ms_layout_"))
try:
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none",
         "--no-checkout", url, "r"],
        cwd=str(tmp), capture_output=True, env=env)
    repo = tmp / "r"
    out = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD"],
                         cwd=str(repo), capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    paths = [p for p in out.strip().split("\n") if p]

    print("=" * 72)
    print("云仓库关键路径")
    print("=" * 72)
    PATS = [
        "config.py",
        "GPT_SoVITS/configs/s2.json",
        "data/zfy_v5/zfy_v5.list",
        "data/zfy_v5/wav32k/",
        "logs/zfy_v5/2-name2text.txt",
        "logs/zfy_v5/6-name2semantic.tsv",
        "logs/zfy_v5/4-cnhubert/",
        "_tools/notebook_preflight.py",
    ]
    for pat in PATS:
        hit = [p for p in paths if p.startswith(pat) or pat in p]
        n = f"  ({len(hit)} 个)" if pat.endswith("/") else ""
        first = hit[0] if hit else "**缺失**"
        print(f"  {pat:34s} {first}{n}")

    print()
    print("=" * 72)
    print("结论：路径推导")
    print("=" * 72)
    has_data = any(p.startswith("data/zfy_v5/") for p in paths)
    has_logs = any(p.startswith("logs/zfy_v5/") for p in paths)
    print(f"  data/zfy_v5 在检出根内 : {has_data}")
    print(f"  logs/zfy_v5 在检出根内 : {has_logs}")
    print()
    if has_data and has_logs:
        print("  => 云上 WORK 应为 <检出根>/data/zfy_v5")
        print("     本机 WORK 为 D:\\A\\voiceclone\\... 之外的位置")
        print("     所以 train_v5.py 需要**探测两种布局**，不能写死。")
    print()
    print("  检出根判定标志: config.py + GPT_SoVITS/ 同时存在")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
