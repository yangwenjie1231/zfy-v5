# -*- coding: utf-8 -*-
"""在真实远程仓库上测试 fix_and_verify_lfs.py（建一个"跳过 LFS"的 clone）。

验证两件事：
  1. 检查逻辑能识别出所有指针文件
  2. --fix 能修多少（若远程对象完好，应能全修好）
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PY = r"C:\Users\A\miniconda3\envs\voice\python.exe"
SCRIPT = Path(r"D:\A\zfy-v5\fix_and_verify_lfs.py")

tok = (Path.home() / ".modelscope" / "credentials" / "git_token") \
    .read_text(encoding="utf-8").strip()
url = ("https://oauth2:" + tok + "@www.modelscope.cn/datasets/"
       "yangwenjie1231/zfy-v5-fullft-pkg.git")

tmp = Path(tempfile.mkdtemp(prefix="lfs_test_"))
try:
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    print("clone（跳过 LFS 内容）…", flush=True)
    r = subprocess.run(["git", "clone", "--depth", "1", url, "r"],
                       cwd=str(tmp), capture_output=True, env=env)
    print(f"  exit {r.returncode}")
    repo = tmp / "r"

    (repo / "_tools").mkdir(exist_ok=True)
    shutil.copy2(SCRIPT, repo / "_tools" / "fix_and_verify_lfs.py")

    print()
    print("=" * 78)
    print("测试 1：仅检查（应报出大量指针）")
    print("=" * 78)
    r2 = subprocess.run([PY, "_tools/fix_and_verify_lfs.py", "--root", "."],
                        cwd=str(repo), capture_output=True)
    out = r2.stdout.decode("utf-8", "replace")
    err = r2.stderr.decode("utf-8", "replace")
    print(out[-3000:])
    if err.strip():
        print("STDERR:", err[-800:])
    print(f"exit={r2.returncode}")

    print()
    print("=" * 78)
    print("测试 2：--fix（清除缓存后重新拉取）")
    print("=" * 78)
    r3 = subprocess.run(
        [PY, "_tools/fix_and_verify_lfs.py", "--root", ".", "--fix"],
        cwd=str(repo), capture_output=True, timeout=3600)
    out3 = r3.stdout.decode("utf-8", "replace")
    err3 = r3.stderr.decode("utf-8", "replace")
    print(out3[-4000:])
    if err3.strip():
        print("STDERR:", err3[-1200:])
    print(f"exit={r3.returncode}")

    # 抽查关键文件
    print()
    print("=" * 78)
    print("抽查关键文件体积")
    print("=" * 78)
    for rel, exp in [
        ("GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth",
         766385517),
        ("logs/zfy_v5/2-name2text.txt", 109000),
        ("logs/zfy_v5/5-wav32k", None),
        ("logs/zfy_v5/4-cnhubert", None),
    ]:
        p = repo / rel
        if p.is_dir():
            files = list(p.iterdir())
            if files:
                szs = sorted(f.stat().st_size for f in files)
                print(f"  {rel:52s} {len(files)} 个，中位 {szs[len(szs)//2]:,d}")
        elif p.exists():
            sz = p.stat().st_size
            mark = "OK" if (exp is None or abs(sz - exp) / exp < 0.1) else "**异常**"
            print(f"  {rel:52s} {sz:,d}  期望约 {exp:,d}  {mark}")
        else:
            print(f"  {rel:52s} 缺失")
finally:
    keep = os.environ.get("KEEP")
    if keep:
        print(f"\n保留: {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
