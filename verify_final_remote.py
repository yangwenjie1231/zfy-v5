# -*- coding: utf-8 -*-
"""最终验证远端仓库完整性与关键文件。"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
NS, REPO = "yangwenjie1231", "zfy-v5-fullft-pkg"
tok = (Path.home() / ".modelscope" / "credentials" / "git_token") \
    .read_text(encoding="utf-8").strip()
url = f"https://oauth2:{tok}@www.modelscope.cn/datasets/{NS}/{REPO}.git"

KEY = [
    "_tools/cloud_synth_sweep.py",
    "_tools/sdk_repair_lfs.py",
    "_tools/install_ffmpeg.sh",
    "_tools/patch_strict_audio.py",
    "_tools/zfy_v5_fullft.ipynb",
    "ref/default.wav",
    "ref/ref.txt",
    "ref/zfh-e4.ckpt",
    "GPT_SoVITS/s2_train_v5_full.py",
    "logs/zfy_v5/2-name2text.txt",
    "logs/zfy_v5/6-name2semantic.tsv",
    "data/zfy_v5/zfy_v5.list",
    "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth",
]

tmp = Path(tempfile.mkdtemp(prefix="ms_final_"))
try:
    env = os.environ.copy()
    r = subprocess.run(["git", "clone", "--depth", "1", url, "r"],
                       cwd=str(tmp), capture_output=True, env=env, timeout=1800)
    if r.returncode != 0:
        print("clone 失败:", r.stderr.decode("utf-8", "replace")[-300:])
        sys.exit(1)
    repo = tmp / "r"

    def git(*a):
        return subprocess.run(["git", *a], cwd=str(repo), capture_output=True,
                              text=True, encoding="utf-8",
                              errors="replace").stdout

    print("=" * 74)
    print("远端仓库状态")
    print("=" * 74)
    print("  提交  :", git("log", "-1", "--format=%h %s").strip()[:64])
    paths = [p for p in git("ls-tree", "-r", "--name-only", "HEAD")
             .strip().split("\n") if p]
    print("  文件数:", len(paths))

    print()
    print("=" * 74)
    print("关键文件")
    print("=" * 74)
    missing = []
    for k in KEY:
        ok = k in paths
        p = repo / k
        sz = p.stat().st_size if p.exists() else -1
        tag = "OK  " if ok else "缺失"
        print(f"  {tag} {k:56s} {sz:>9,d} 字节")
        if not ok:
            missing.append(k)

    print()
    print("=" * 74)
    print("cloud_synth_sweep.py 内容确认")
    print("=" * 74)
    p = repo / "_tools" / "cloud_synth_sweep.py"
    if p.exists():
        s = p.read_text(encoding="utf-8", errors="replace")
        for name, needle in [
            ("upload_results（回传功能）", "upload_results"),
            ("find_token", "find_token"),
            ("--upload-only", "--upload-only"),
            ("--no-upload", "--no-upload"),
            ("生产参考音频路径 ref/default.wav", "ref"),
        ]:
            print(f"  {'OK  ' if needle in s else 'FAIL'} {name}")
        print(f"  体积: {len(s)} 字节")
    else:
        print("  文件不存在")

    print()
    print("=" * 74)
    print("结论")
    print("=" * 74)
    if not missing and len(paths) > 1000:
        print(f"  ✓ 仓库完整（{len(paths)} 个文件），关键文件全部就位")
    else:
        print(f"  ✗ 缺失 {len(missing)} 个 / 文件数 {len(paths)}")
        for m in missing:
            print(f"    {m}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
