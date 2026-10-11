# -*- coding: utf-8 -*-
"""确认 `git lfs ls-files` 的输出格式，以及 cat-file 解析是否正确。

上一版 fix_and_verify_lfs.py 返回 "取得声明体积: 0 个" 却报"全部完整" ——
假阳性。原因很可能是 `git lfs ls-files` 每行不只是路径，
而是 `<oid> <marker> <path>`，被整行当路径去 cat-file 就全部 missing。
"""
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

tmp = Path(tempfile.mkdtemp(prefix="fmt_"))
try:
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(["git", "clone", "--depth", "1", url, "r"],
                   cwd=str(tmp), capture_output=True, env=env)
    repo = tmp / "r"

    def run(args, timeout=600):
        r = subprocess.run(args, cwd=str(repo), capture_output=True,
                           timeout=timeout)
        return (r.returncode,
                (r.stdout or b"").decode("utf-8", "replace"),
                (r.stderr or b"").decode("utf-8", "replace"))

    print("=" * 76)
    print("1. git lfs ls-files（默认）前 3 行")
    print("=" * 76)
    c, o, e = run(["git", "lfs", "ls-files"])
    for line in o.split("\n")[:3]:
        print(f"  {line!r}")
    print(f"  总行数: {len([l for l in o.split(chr(10)) if l.strip()])}")

    print()
    print("2. git lfs ls-files -n（只要名字）前 3 行")
    c2, o2, e2 = run(["git", "lfs", "ls-files", "-n"])
    print(f"  退出码 {c2}")
    for line in o2.split("\n")[:3]:
        print(f"  {line!r}")
    if not o2.strip():
        print(f"  (空) stderr: {e2.strip()[:200]}")

    print()
    print("3. git lfs ls-files --long 前 3 行")
    c3, o3, e3 = run(["git", "lfs", "ls-files", "--long"])
    print(f"  退出码 {c3}")
    for line in o3.split("\n")[:3]:
        print(f"  {line!r}")

    print()
    print("4. 用 check-attr 找 LFS 文件（最稳的办法）")
    r = subprocess.run(
        ["git", "ls-files", "-z"], cwd=str(repo), capture_output=True,
        timeout=300)
    names = [n for n in r.stdout.decode("utf-8", "replace").split("\0") if n]
    print(f"  git ls-files: {len(names)} 个")

    r2 = subprocess.run(
        ["git", "check-attr", "--stdin", "-z", "filter"], cwd=str(repo),
        input=r.stdout, capture_output=True, timeout=600)
    parts = r2.stdout.decode("utf-8", "replace").split("\0")
    lfs = []
    for i in range(0, len(parts) - 2, 3):
        path, attr, val = parts[i], parts[i + 1], parts[i + 2]
        if val == "lfs":
            lfs.append(path)
    print(f"  LFS 属性为 lfs 的文件: {len(lfs)} 个")
    for p in lfs[:3]:
        sz = (repo / p).stat().st_size if (repo / p).exists() else -1
        print(f"    {p}  ({sz} 字节)")

    print()
    print("5. 用正确路径测 cat-file 解析")
    if lfs:
        sample = lfs[:5]
        inp = "".join(f"HEAD:{p}\n" for p in sample).encode()
        r3 = subprocess.run(["git", "cat-file", "--batch"], cwd=str(repo),
                            input=inp, capture_output=True, timeout=300)
        print(f"  输出 {len(r3.stdout)} 字节")
        head = r3.stdout[:200].decode("utf-8", "replace")
        print(f"  前 200 字节:\n{head}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
