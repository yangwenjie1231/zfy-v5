# -*- coding: utf-8 -*-
"""确认新补的推理目录/文件已推送到远端。"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
tok = (Path.home() / ".modelscope" / "credentials" / "git_token") \
    .read_text(encoding="utf-8").strip()
url = ("https://oauth2:" + tok + "@www.modelscope.cn/datasets/"
       "yangwenjie1231/zfy-v5-fullft-pkg.git")

PATTERNS = [
    "GPT_SoVITS/BigVGAN/",
    "GPT_SoVITS/feature_extractor/",
    "GPT_SoVITS/Accel/",
    "GPT_SoVITS/sv.py",
    "GPT_SoVITS/s1_train.py",
    "_tools/install_infer_deps.sh",
    "_tools/cloud_synth_sweep.py",
    "tools/AP_BWE_main/",
    "GPT_SoVITS/eres2net/",
    "ref/default.wav",
]

tmp = Path(tempfile.mkdtemp(prefix="ms_chk2_"))
try:
    r = subprocess.run(["git", "clone", "--depth", "1", url, "r"],
                       cwd=str(tmp), capture_output=True,
                       env=os.environ.copy(), timeout=1800)
    if r.returncode != 0:
        print("clone 失败:", r.stderr.decode("utf-8", "replace")[-300:])
        sys.exit(1)
    repo = tmp / "r"
    out = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD"],
                         cwd=str(repo), capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    paths = [p for p in out.strip().split("\n") if p]
    print(f"远端文件总数: {len(paths)}")
    print()
    bad = []
    for pat in PATTERNS:
        n = sum(1 for p in paths if p == pat or p.startswith(pat))
        tag = "OK  " if n else "缺失"
        if not n:
            bad.append(pat)
        print(f"  {tag} {pat:38s} {n:4d}")
    print()
    if bad:
        print(f"!! 仍缺 {len(bad)} 项: {bad}")
    else:
        print("✓ 推理所需文件全部到位")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
