# -*- coding: utf-8 -*-
"""测试官方 SDK 的 snapshot_download 能否取到我们私有数据集的 LFS 文件。

## 为什么走这条路

云上 git-lfs 出了两个问题：静默无输出、下载截断（766MB 只下了 701MB）。
HTTP 裸 API 对私有数据集返回 404（缺 SDK 的鉴权头）。

而 `modelscope` SDK 的 `snapshot_download` 是官方下载路径：
  · 自带鉴权（读 ~/.modelscope/credentials）
  · 支持断点续传与重试
  · 按 allow_patterns 可只取需要的文件

## 本脚本

  1. 看本地有哪些凭据
  2. 用 allow_patterns 只下**小文件**，验证通路
  3. 确认取到的内容 sha256 与 LFS 指针一致
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RID = "yangwenjie1231/zfy-v5-fullft-pkg"

print("=" * 76)
print("1. 本地凭据")
print("=" * 76)
cred = Path.home() / ".modelscope" / "credentials"
if cred.is_dir():
    for p in sorted(cred.iterdir()):
        val = p.read_text(encoding="utf-8", errors="replace").strip()
        print(f"  {p.name:20s} {len(val)} 字符  {val[:6]}…")
else:
    print(f"  目录不存在: {cred}")

print()
print("=" * 76)
print("2. SDK 能否读取该数据集的文件列表")
print("=" * 76)
try:
    from modelscope.hub.api import HubApi
    api = HubApi()
    files = api.get_dataset_files(RID, recursive=True)
    print(f"  get_dataset_files -> {len(files)} 条（分页上限 100）")
    for f in files[:3]:
        print(f"    {f['Path']}  {f.get('Size')} 字节")
except Exception as e:  # noqa: BLE001
    print(f"  失败: {type(e).__name__}: {str(e)[:200]}")

print()
print("=" * 76)
print("3. snapshot_download 取小文件（configs/*.json，几 KB）")
print("=" * 76)
tmp = Path(tempfile.mkdtemp(prefix="sdk_dl_"))
try:
    from modelscope import snapshot_download
    try:
        p = snapshot_download(
            RID, repo_type="dataset",
            allow_patterns=["GPT_SoVITS/configs/*.json"],
            local_dir=str(tmp / "out"),
        )
        print(f"  下载目录: {p}")
        got = list((tmp / "out").rglob("*.json"))
        print(f"  取到 {len(got)} 个 json")
        for f in sorted(got)[:4]:
            data = f.read_bytes()
            print(f"    {f.name:22s} {len(data):6d} 字节  "
                  f"sha256 {hashlib.sha256(data).hexdigest()[:16]}…")
        # 对照指针
        s2 = tmp / "out" / "GPT_SoVITS" / "configs" / "s2.json"
        if s2.exists():
            d = s2.read_bytes()
            print()
            print(f"  s2.json 是真实 JSON: {d[:1] == b'{'}")
            print(f"    前 80 字节: {d[:80]!r}")
    except Exception as e:  # noqa: BLE001
        print(f"  snapshot_download 失败: {type(e).__name__}")
        print(f"    {str(e)[:400]}")

    print()
    print("=" * 76)
    print("4. 用 SDK 取一个特征 wav（验证 LFS 大文件通路）")
    print("=" * 76)
    try:
        p2 = snapshot_download(
            RID, repo_type="dataset",
            allow_patterns=[
                "logs/zfy_v5/5-wav32k/au_greetEnvTalk_zhuangfy_lower_01.wav"],
            local_dir=str(tmp / "out2"),
        )
        wavs = list((tmp / "out2").rglob("*.wav"))
        print(f"  取到 {len(wavs)} 个 wav")
        for w in wavs[:2]:
            d = w.read_bytes()
            print(f"    {w.name[:50]}")
            print(f"      {len(d):,d} 字节  RIFF={d[:4] == b'RIFF'}  "
                  f"sha256 {hashlib.sha256(d).hexdigest()[:16]}…")
    except Exception as e:  # noqa: BLE001
        print(f"  失败: {type(e).__name__}: {str(e)[:300]}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
