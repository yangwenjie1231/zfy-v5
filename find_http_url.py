# -*- coding: utf-8 -*-
"""用**正确的 URL 格式**测 HTTP 直下（对照 SDK 日志）。

之前用 `https://www.modelscope.cn/api/v1/datasets/.../repo?...` 得到 404。
但 SDK 日志里实际用的是**不带 www** 的域名，且多两个参数：

    https://modelscope.cn/api/v1/datasets/<owner>/<repo>/repo
        ?Source=SDK&Revision=master&FilePath=README.md&View=False

本脚本逐一试几种组合，找出能用的那个，并验证：
  · 内容与 LFS 指针声明的 sha256 是否一致
  · 是否支持 Range 断点续传
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NS, REPO = "yangwenjie1231", "zfy-v5-fullft-pkg"
TOKEN = (Path.home() / ".modelscope" / "credentials" / "git_token") \
    .read_text(encoding="utf-8").strip()

P_SMALL = "GPT_SoVITS/configs/s2.json"           # 1671 字节（LFS）
P_WAV = "logs/zfy_v5/5-wav32k/au_greetEnvTalk_zhuangfy_lower_01.wav"
P_BIG = "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"

VARIANTS = [
    ("modelscope + Source=SDK + View=False",
     "https://modelscope.cn/api/v1/datasets/{ns}/{repo}/repo"
     "?Source=SDK&Revision=master&FilePath={path}&View=False"),
    ("modelscope + Revision + FilePath",
     "https://modelscope.cn/api/v1/datasets/{ns}/{repo}/repo"
     "?Revision=master&FilePath={path}"),
    ("www + Source=SDK + View=False",
     "https://www.modelscope.cn/api/v1/datasets/{ns}/{repo}/repo"
     "?Source=SDK&Revision=master&FilePath={path}&View=False"),
    ("modelscope resolve 接口",
     "https://modelscope.cn/api/v1/datasets/{ns}/{repo}/resolve/master/{path}"),
]


def fetch(url: str, rng: str | None = None, timeout=120):
    req = urllib.request.Request(url)
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    if rng:
        req.add_header("Range", rng)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), b""
    except Exception as e:  # noqa: BLE001
        return -1, {"error": str(e)}, b""


print("=" * 78)
print("试各种 URL 格式（目标: GPT_SoVITS/configs/s2.json，1671 字节）")
print("=" * 78)
winner = None
for name, tpl in VARIANTS:
    url = tpl.format(ns=NS, repo=REPO, path=urllib.parse.quote(P_SMALL))
    st, hd, body = fetch(url, timeout=60)
    n = len(body)
    looks_json = body[:1] in (b"{", b"[")
    print(f"  [{st:>4}] {name}")
    print(f"         {n} 字节  {'(JSON 内容)' if looks_json else ''}")
    if st == 200 and n > 100:
        winner = (name, tpl)
        print("         ^^^ 可用")
        break
    if st != 200 and hd.get("error"):
        print(f"         {hd['error'][:90]}")

if not winner:
    print("\n  所有格式都失败。可能该数据集是**私有**，HTTP API 需要别的鉴权方式。")
    sys.exit(1)

name, tpl = winner
print()
print("=" * 78)
print(f"用可用格式（{name}）验证哈希与 Range")
print("=" * 78)

# 取远程指针作为校验依据
tmp = Path(tempfile.mkdtemp(prefix="http2_"))
try:
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    url_git = f"https://oauth2:{TOKEN}@www.modelscope.cn/datasets/{NS}/{REPO}.git"
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                    "--no-checkout", url_git, "r"], cwd=str(tmp),
                   capture_output=True, env=env)
    repo = tmp / "r"

    import re
    PTR = re.compile(rb"oid sha256:([0-9a-f]{64})\s*\n\s*size (\d+)")

    def ptr_of(p):
        r = subprocess.run(["git", "show", f"HEAD:{p}"], cwd=str(repo),
                           capture_output=True)
        m = PTR.search(r.stdout or b"")
        return (m.group(1).decode(), int(m.group(2))) if m else None

    # 1) 小文件哈希
    ref = ptr_of(P_SMALL)
    st, hd, body = fetch(tpl.format(ns=NS, repo=REPO,
                                    path=urllib.parse.quote(P_SMALL)))
    if ref and body:
        got = hashlib.sha256(body).hexdigest()
        print(f"  小文件: {len(body)} 字节  声明 {ref[1]}  "
              f"哈希匹配 {got == ref[0]}")

    # 2) wav 特征文件
    ref2 = ptr_of(P_WAV)
    st2, hd2, b2 = fetch(tpl.format(ns=NS, repo=REPO,
                                    path=urllib.parse.quote(P_WAV)))
    print(f"  wav:    {len(b2):,d} 字节", end="")
    if ref2:
        print(f"  声明 {ref2[1]:,d}  哈希匹配 "
              f"{hashlib.sha256(b2).hexdigest() == ref2[0]}  "
              f"RIFF={b2[:4] == b'RIFF'}")
    else:
        print()

    # 3) Range 续传
    st3, hd3, b3 = fetch(tpl.format(ns=NS, repo=REPO,
                                    path=urllib.parse.quote(P_BIG)),
                         rng="bytes=0-1048575", timeout=180)
    print(f"  大文件 Range: 状态 {st3}  收到 {len(b3):,d} 字节  "
          f"支持续传={st3 == 206}")
    print(f"     Content-Range: {hd3.get('Content-Range')}")
    print(f"     Accept-Ranges: {hd3.get('Accept-Ranges')}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
print("=" * 78)
print("结论")
print("=" * 78)
print(f"""
  可用 URL 模板（Python 里用 .format 填 path）：
    {tpl}

  下一步：写一个 HTTP 下载器，逐文件：
    1. 从 git 指针拿 oid + size
    2. Range 断点续传下载（中断可续，避免上次那种截断）
    3. 校验 sha256，不符则重下
    4. 写回工作区
""")
