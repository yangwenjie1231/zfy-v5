# -*- coding: utf-8 -*-
"""找出"指针 clone"的正确修复命令（实测，不猜）。

## 背景

用户在云上跑 `git lfs pull` **没有任何输出**，也没报错。
这很可疑：正常 pull 会打印下载进度。

可能原因：
  · git-lfs 认为对象已在缓存，跳过 fetch
  · 但工作区里的文件**仍是指针** -> 缺的是 `checkout` 这一步
  · 或 .gitattributes 不在工作区，git-lfs 不知道哪些文件该处理

## 本脚本

本地复现「SKIP_SMUDGE clone」，然后依次试各种修复命令，
找出**真正能把指针变成内容**的那条。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REL = "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"
EXPECT_MB = 766

tok = (Path.home() / ".modelscope" / "credentials" / "git_token") \
    .read_text(encoding="utf-8").strip()
url = ("https://oauth2:" + tok + "@www.modelscope.cn/datasets/"
       "yangwenjie1231/zfy-v5-fullft-pkg.git")


def run(cmd, cwd, env=None, timeout=1800):
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, cwd=str(cwd), env=e, timeout=timeout,
                       capture_output=True)
    out = (r.stdout or b"").decode("utf-8", "replace")
    err = (r.stderr or b"").decode("utf-8", "replace")
    return r.returncode, out, err


def state(repo: Path) -> str:
    p = repo / REL
    if not p.exists():
        return "缺失"
    sz = p.stat().st_size
    head = p.read_bytes()[:40]
    if sz < 1024 and head.startswith(b"version https://git-lfs"):
        return f"指针({sz}B)"
    return f"内容({sz/1e6:.1f}MB)"


tmp = Path(tempfile.mkdtemp(prefix="ms_fix_"))
try:
    print("=" * 74)
    print("0. 建立「跳过 LFS」的 clone（复现用户状态）")
    print("=" * 74)
    code, out, err = run(["git", "clone", "--depth", "1", url, "r"],
                         tmp, env={"GIT_LFS_SKIP_SMUDGE": "1"})
    repo = tmp / "r"
    print(f"  clone 退出码 {code}  状态: {state(repo)}")

    ga = repo / ".gitattributes"
    print(f"  .gitattributes 存在: {ga.exists()}")
    if ga.exists():
        txt = ga.read_text(encoding="utf-8", errors="replace")
        hits = [l for l in txt.split("\n") if "pth" in l or "ckpt" in l
                or "bin" in l]
        print(f"  含 .pth/.ckpt/.bin 规则: {len(hits)} 条")
        for h in hits[:4]:
            print(f"    {h}")

    print()
    print("=" * 74)
    print("1. 试 `git lfs install --local` + `git lfs pull`（用户试过的）")
    print("=" * 74)
    c1, o1, e1 = run(["git", "lfs", "install", "--local"], repo)
    print(f"  install --local: 退出码 {c1}")
    c2, o2, e2 = run(["git", "lfs", "pull"], repo)
    print(f"  lfs pull: 退出码 {c2}")
    print(f"    stdout: {o2.strip()[:300] or '(空)'}")
    print(f"    stderr: {e2.strip()[:300] or '(空)'}")
    print(f"  -> 状态: {state(repo)}")

    print()
    print("=" * 74)
    print("2. 试 `git lfs fetch --all` + `git lfs checkout`")
    print("=" * 74)
    c3, o3, e3 = run(["git", "lfs", "fetch", "--all"], repo)
    print(f"  fetch --all: 退出码 {c3}")
    print(f"    stdout: {o3.strip()[-300:] or '(空)'}")
    print(f"    stderr: {e3.strip()[-300:] or '(空)'}")
    c4, o4, e4 = run(["git", "lfs", "checkout"], repo)
    print(f"  lfs checkout: 退出码 {c4}")
    print(f"    stdout: {o4.strip()[-300:] or '(空)'}")
    print(f"    stderr: {e4.strip()[-300:] or '(空)'}")
    print(f"  -> 状态: {state(repo)}")

    print()
    print("=" * 74)
    print("3. 试强制重新 checkout（删指针后从 index 恢复）")
    print("=" * 74)
    p = repo / REL
    if p.exists() and state(repo).startswith("指针"):
        p.unlink()
        c5, o5, e5 = run(["git", "checkout", "--", REL], repo)
        print(f"  删除指针 + git checkout: 退出码 {c5}")
        print(f"    stderr: {e5.strip()[:200] or '(空)'}")
        print(f"  -> 状态: {state(repo)}")

    print()
    print("=" * 74)
    print("4. 试 `git lfs pull --include`（显式指定路径）")
    print("=" * 74)
    if state(repo).startswith("指针"):
        c6, o6, e6 = run(
            ["git", "lfs", "pull", "--include",
             "GPT_SoVITS/pretrained_models/**"], repo)
        print(f"  退出码 {c6}")
        print(f"    stdout: {o6.strip()[-300:] or '(空)'}")
        print(f"    stderr: {e6.strip()[-300:] or '(空)'}")
        print(f"  -> 状态: {state(repo)}")

    print()
    print("=" * 74)
    print("结论")
    print("=" * 74)
    st = state(repo)
    print(f"  最终状态: {st}")
    if st.startswith("内容"):
        print(f"  ✓ 修好了（{EXPECT_MB} MB 左右）")
    else:
        print("  ✗ 上述命令都没修好 -> 需要重新 clone（不带 SKIP_SMUDGE）")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
