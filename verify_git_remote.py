# -*- coding: utf-8 -*-
"""用 git ls-tree 校验远程仓库内容（绕开 API 的 100 条分页限制）。

`get_dataset_files(recursive=True)` 只返回 100 条 —— 那是分页，
不能据此判断文件缺失。git 侧看才准。
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NS, REPO = "yangwenjie1231", "zfy-v5-fullft-pkg"

KEY = [
    "_tools/notebook_preflight.py",
    "_tools/patch_models_for_torch212.py",
    "_tools/MODELSCOPE_NOTEBOOK_GUIDE.md",
    "GPT_SoVITS/s2_train_v5_full.py",
    "GPT_SoVITS/module/models.py",
    "GPT_SoVITS/module/models_v5.py",
    "GPT_SoVITS/f5_tts/model/backbones/dit.py",
    "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth",
    "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth",
    "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin",
    "GPT_SoVITS/pretrained_models/chinese-hubert-base/pytorch_model.bin",
    "logs/zfy_v5/6-name2semantic.tsv",
    "logs/zfy_v5/2-name2text.txt",
    "data/zfy_v5/zfy_v5.list",
]

tok = (Path.home() / ".modelscope" / "credentials" / "git_token")
token = tok.read_text(encoding="utf-8").strip() if tok.exists() else ""
url = (f"https://oauth2:{token}@www.modelscope.cn/datasets/{NS}/{REPO}.git"
       if token else f"https://www.modelscope.cn/datasets/{NS}/{REPO}.git")

tmp = Path(tempfile.mkdtemp(prefix="ms_verify_"))
try:
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"     # 只取指针，不下 3GB 内容
    r = subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", url, "r"],
        cwd=str(tmp), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"clone 失败: {r.stderr[-400:]}")
        sys.exit(1)
    repo = tmp / "r"

    out = subprocess.run(["git", "ls-tree", "-r", "-l", "HEAD"],
                         cwd=str(repo), capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    lines = [l for l in out.split("\n") if l.strip()]
    # ls-tree 对 LFS 文件显示的是指针大小，实际大小要看 lfs ls-files
    lfs = subprocess.run(["git", "lfs", "ls-files", "-s"],
                         cwd=str(repo), capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    lfs_lines = [l for l in lfs.split("\n") if l.strip()]

    print("=" * 74)
    print("远程仓库（git ls-tree）")
    print("=" * 74)
    print(f"  文件总数    : {len(lines)}")
    print(f"  LFS 对象数  : {len(lfs_lines)}")

    # LFS 实际体积
    total = 0
    for l in lfs_lines:
        parts = l.split()
        if len(parts) >= 4:
            try:
                total += int(parts[3])
            except ValueError:
                pass
    print(f"  LFS 实际体积: {total/1e9:.2f} GB")

    # 非 LFS 文件的体积
    small = 0
    for l in lines:
        p = l.split()
        if len(p) >= 4 and p[3].isdigit():
            small += int(p[3])
    print(f"  其它文件体积: {small/1e6:.1f} MB")

    print()
    print("=" * 74)
    print("关键文件核查")
    print("=" * 74)
    paths = set()
    for l in lines:
        p = l.split(maxsplit=3)
        if len(p) >= 4:
            paths.add(p[3].strip())

    missing = []
    for k in KEY:
        hit = [p for p in paths if p.endswith(k) or k in p]
        if hit:
            print(f"  OK    {k}")
        else:
            print(f"  缺失  {k}")
            missing.append(k)

    print()
    print("=" * 74)
    print("zip 是否已移除")
    print("=" * 74)
    zips = [p for p in paths if p.lower().endswith(".zip")]
    print(f"  zip 文件: {len(zips)}  {'正确' if not zips else zips}")

    print()
    print("=" * 74)
    print("结论")
    print("=" * 74)
    if not missing and not zips:
        print("  ✓ 目录树结构完整，旧的 zip 已删除")
        print(f"  ✓ {len(lines)} 文件 / LFS {total/1e9:.2f} GB")
        print()
        print("  Notebook 里直接 clone（无需解压）：")
        print(f"    git clone https://www.modelscope.cn/datasets/{NS}/{REPO}.git")
    else:
        if missing:
            print(f"  缺失 {len(missing)} 个关键文件: {missing}")
        if zips:
            print(f"  仍有 zip: {zips}")
finally:
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
