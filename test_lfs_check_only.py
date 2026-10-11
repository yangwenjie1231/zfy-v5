# -*- coding: utf-8 -*-
"""便宜地测试 LFS 检查逻辑（不下载 3.1GB）。

上一个测试跑了 `git lfs fetch --all`（3.1GB），太重。
这里只测**解析与判定**部分：克隆时跳过 LFS，然后只跑 --check（离线读指针），
不跑 --fix。
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

tmp = Path(tempfile.mkdtemp(prefix="lfs_chk_"))
try:
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    print("clone（跳过 LFS 内容，只取指针）…", flush=True)
    r = subprocess.run(["git", "clone", "--depth", "1", url, "r"],
                       cwd=str(tmp), capture_output=True, env=env)
    print(f"  exit {r.returncode}")
    repo = tmp / "r"
    (repo / "_tools").mkdir(exist_ok=True)
    shutil.copy2(SCRIPT, repo / "_tools" / "fix_and_verify_lfs.py")

    print()
    print("=" * 78)
    print("只跑 --check（离线读指针，不下载）")
    print("=" * 78)
    r2 = subprocess.run([PY, "_tools/fix_and_verify_lfs.py", "--root", "."],
                        cwd=str(repo), capture_output=True, timeout=1200)
    out = r2.stdout.decode("utf-8", "replace")
    err = r2.stderr.decode("utf-8", "replace")
    print(out)
    if err.strip():
        print("STDERR:", err[-1000:])
    print(f"exit={r2.returncode}  (期望 1 = 检出问题)")

    # 独立核对：手工数几个
    print()
    print("=" * 78)
    print("独立核对（不依赖脚本逻辑）")
    print("=" * 78)
    for rel in ("logs/zfy_v5/2-name2text.txt",
                "logs/zfy_v5/5-wav32k",
                "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth",
                "data/zfy_v5/zfy_v5.list"):
        p = repo / rel
        if p.is_dir():
            files = [f for f in p.iterdir() if f.is_file()]
            ptrs = sum(1 for f in files
                       if f.stat().st_size < 1024
                       and f.read_bytes()[:40].startswith(b"version https://git-lfs"))
            print(f"  {rel:52s} {len(files):4d} 个，指针 {ptrs}")
        elif p.exists():
            sz = p.stat().st_size
            isp = sz < 1024 and p.read_bytes()[:40].startswith(b"version https://git-lfs")
            print(f"  {rel:52s} {sz:,d} 字节  {'指针' if isp else '内容'}")
        else:
            print(f"  {rel:52s} 缺失")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
