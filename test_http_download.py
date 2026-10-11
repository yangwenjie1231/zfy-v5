# -*- coding: utf-8 -*-
"""验证：能否用 HTTP 直接下载魔搭仓库文件（绕开 git-lfs）。

## 为什么

云上 `git lfs pull` 出了两个问题：
  · 静默无输出、什么都没做
  · 下到一半截断（701,571,072 / 766,385,517）

而远程对象其实是完好的。与其跟 git-lfs 纠缠，不如直接用 HTTP：

    https://www.modelscope.cn/api/v1/datasets/<owner>/<repo>/repo
        ?Revision=master&FilePath=<path>

这个 URL 是 SDK 内部用的（我在 MsDataset 的日志里见过），
支持 Range 断点续传，并且能自己算 sha256 校验。

## 本脚本

  1. 测能否匿名/带 token 下载一个**小文件**（configs/s2.json, 1671 字节）
  2. 测下载一个**特征文件**（logs/zfy_v5/5-wav32k/*.wav）
  3. 测**断点续传**（Range 请求）
  4. 测下载**大文件的前 1 MB**（不完整下载，只为验证通路）
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NS, REPO = "yangwenjie1231", "zfy-v5-fullft-pkg"
BASE = f"https://www.modelscope.cn/api/v1/datasets/{NS}/{REPO}/repo"

tok_p = Path.home() / ".modelscope" / "credentials" / "git_token"
TOKEN = tok_p.read_text(encoding="utf-8").strip() if tok_p.exists() else ""

# 从远程 git 读指针（拿 oid + size 作为校验依据）
import os
import shutil
import subprocess
import tempfile

PTR_RE = re.compile(rb"oid sha256:([0-9a-f]{64})\s*\n\s*size (\d+)")


def url_for(path: str, revision: str = "master") -> str:
    q = urllib.parse.urlencode({"Revision": revision, "FilePath": path})
    return f"{BASE}?{q}"


def head(path: str) -> tuple[int, dict]:
    req = urllib.request.Request(url_for(path), method="HEAD")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {})
    except Exception as e:  # noqa: BLE001
        return -1, {"error": str(e)}


print("=" * 78)
print("1. HTTP 通路测试（小文件 configs/s2.json，期望 1671 字节）")
print("=" * 78)
P = "GPT_SoVITS/configs/s2.json"
st, hd = head(P)
print(f"  HEAD 状态: {st}")
for k in ("Content-Length", "Content-Type", "Accept-Ranges"):
    if k in hd:
        print(f"    {k}: {hd[k]}")

# 实际 GET
req = urllib.request.Request(url_for(P))
if TOKEN:
    req.add_header("Authorization", f"Bearer {TOKEN}")
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    print(f"  GET 成功: {len(data)} 字节")
    print(f"  sha256: {hashlib.sha256(data).hexdigest()[:32]}…")
    print(f"  内容前 60 字节: {data[:60]!r}")
except Exception as e:  # noqa: BLE001
    print(f"  GET 失败: {type(e).__name__}: {str(e)[:200]}")
    data = b""

# 对照：远程指针声明
tmp = Path(tempfile.mkdtemp(prefix="http_t_"))
try:
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    url = f"https://oauth2:{TOKEN}@www.modelscope.cn/datasets/{NS}/{REPO}.git"
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                    "--no-checkout", url, "r"], cwd=str(tmp),
                   capture_output=True, env=env)
    repo = tmp / "r"

    def ptr_of(path: str):
        r = subprocess.run(["git", "show", f"HEAD:{path}"],
                           cwd=str(repo), capture_output=True)
        m = PTR_RE.search(r.stdout or b"")
        return (m.group(1).decode(), int(m.group(2))) if m else None

    ref = ptr_of(P)
    print()
    if ref:
        oid, size = ref
        print(f"  远程声明: size={size}  oid={oid[:32]}…")
        if data:
            got = hashlib.sha256(data).hexdigest()
            print(f"  HTTP 内容哈希匹配: {got == oid}")
            print(f"  字节数与声明一致  : {len(data) == size}")
    else:
        print("  远程无指针（该文件不走 LFS）")

    print()
    print("=" * 78)
    print("2. 特征文件（logs/zfy_v5/5-wav32k/ 第一个）")
    print("=" * 78)
    p2 = "logs/zfy_v5/5-wav32k/au_greetEnvTalk_zhuangfy_lower_01.wav"
    ref2 = ptr_of(p2)
    st2, hd2 = head(p2)
    print(f"  HEAD {st2}  Content-Length={hd2.get('Content-Length')}")
    if ref2:
        print(f"  远程声明 size={ref2[1]:,d}  oid={ref2[0][:24]}…")
    req2 = urllib.request.Request(url_for(p2))
    if TOKEN:
        req2.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req2, timeout=180) as r:
            d2 = r.read()
        ok = ref2 and hashlib.sha256(d2).hexdigest() == ref2[0]
        print(f"  GET {len(d2):,d} 字节  哈希匹配: {ok}")
        print(f"  是 wav 吗: {d2[:4] == b'RIFF'}")
    except Exception as e:  # noqa: BLE001
        print(f"  GET 失败: {type(e).__name__}: {str(e)[:200]}")

    print()
    print("=" * 78)
    print("3. 断点续传（Range 请求）")
    print("=" * 78)
    big = "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"
    req3 = urllib.request.Request(url_for(big))
    req3.add_header("Range", "bytes=0-1048575")     # 前 1 MB
    if TOKEN:
        req3.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req3, timeout=180) as r:
            d3 = r.read()
        print(f"  Range 0-1048575 -> 状态 {r.status}，收到 {len(d3):,d} 字节")
        print(f"  支持 Range: {r.status == 206}")
    except Exception as e:  # noqa: BLE001
        print(f"  Range 失败: {type(e).__name__}: {str(e)[:200]}")

    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    print("""
  若上面 1-3 都成功，就写一个 HTTP 下载器替代 git lfs pull：
     · 读 git 指针拿 oid/size 作为校验依据
     · Range 断点续传（网络中断可续）
     · 逐个文件校验 sha256，不一致就重下
     · 并发几个线程加速
""")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
