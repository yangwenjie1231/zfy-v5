# -*- coding: utf-8 -*-
"""确认云仓库已更新（含修正后的 _tools/ 与新的 notebook）。"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

tok = (Path.home() / ".modelscope" / "credentials" / "git_token") \
    .read_text(encoding="utf-8").strip()
url = ("https://oauth2:" + tok + "@www.modelscope.cn/datasets/"
       "yangwenjie1231/zfy-v5-fullft-pkg.git")

tmp = Path(tempfile.mkdtemp(prefix="ms_chk_"))
try:
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", url, "r"],
        cwd=str(tmp), capture_output=True, env=env)
    repo = tmp / "r"

    def git(*a):
        return subprocess.run(["git", *a], cwd=str(repo), capture_output=True,
                              text=True, encoding="utf-8",
                              errors="replace").stdout

    print("=" * 70)
    print("云仓库最新提交")
    print("=" * 70)
    print("  " + git("log", "-1", "--format=%h %s").strip())

    paths = [p for p in git("ls-tree", "-r", "--name-only", "HEAD")
             .strip().split("\n") if p]
    print(f"\n  文件总数: {len(paths)}")

    print()
    print("=" * 70)
    print("关键文件")
    print("=" * 70)
    KEY = ["zfy_v5_fullft.ipynb",
           "_tools/notebook_preflight.py",
           "_tools/train_v5.py",
           "_tools/patch_models_for_torch212.py",
           "_tools/MODELSCOPE_NOTEBOOK_GUIDE.md",
           "GPT_SoVITS/s2_train_v5_full.py",
           "GPT_SoVITS/module/models.py"]
    for k in KEY:
        hit = [p for p in paths if k in p]
        print(f"  {'OK  ' if hit else '缺失'} {k}")

    # 检出 preflight 检查修正是否生效
    git("checkout", "HEAD", "--", "_tools/notebook_preflight.py")
    pf = repo / "_tools" / "notebook_preflight.py"
    if pf.exists():
        src = pf.read_text(encoding="utf-8", errors="replace")
        print()
        print("=" * 70)
        print("notebook_preflight.py 修正确认")
        print("=" * 70)
        checks = [
            ("含 find_gsv_root（自动定位检出根）", "find_gsv_root" in src),
            ("含 stdout.reconfigure（修 GBK 崩溃）", "reconfigure" in src),
            ("含 from pathlib import Path", "from pathlib import Path" in src),
            ("无旧写死路径 /mnt/workspace/zfy-v5/GPT-SoVITS",
             "/mnt/workspace/zfy-v5/GPT-SoVITS" not in src),
        ]
        for name, ok in checks:
            print(f"  {'OK  ' if ok else 'FAIL'} {name}")

    # 检查 train_v5 是否跨平台化
    git("checkout", "HEAD", "--", "_tools/train_v5.py")
    tv = repo / "_tools" / "train_v5.py"
    if tv.exists():
        src = tv.read_text(encoding="utf-8", errors="replace")
        print()
        print("train_v5.py 修正确认")
        print("=" * 70)
        for name, ok in [
            ("含 _detect_gsv_root", "_detect_gsv_root" in src),
            ("含 _detect_work（双布局）", "_detect_work" in src),
            ("含 GSV_ROOT 环境变量", "GSV_ROOT" in src),
        ]:
            print(f"  {'OK  ' if ok else 'FAIL'} {name}")

    # notebook 能否解析
    git("checkout", "HEAD", "--", "zfy_v5_fullft.ipynb")
    nbp = repo / "zfy_v5_fullft.ipynb"
    if nbp.exists():
        import json
        nb = json.loads(nbp.read_text(encoding="utf-8"))
        ncode = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        print()
        print(f"notebook: {len(nb['cells'])} 格（{ncode} 个 code），"
              f"{nbp.stat().st_size/1024:.1f} KB  -> 可解析")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
