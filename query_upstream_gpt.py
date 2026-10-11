# -*- coding: utf-8 -*-
"""查上游（ModelScope）官方 GPT-SoVITS 仓库里有哪些 GPT 底模。

## 用户的问题
"全系更新的时候 GPT 底模应该也会更新吧，起码权重会不一样"

## 本地已确认的事实
  · config.py 里 v3/v4/v5dev/v5turbo/v2Pro/v2ProPlus **全部**用 s1v3.ckpt
  · 本地只有两个 GPT 底模（s1v3 + v2final 的 s1bert25hz）
  · 两者 sha256 不同、但结构完全相同（295 键、732 vocab、77.61M 参数）

## 本脚本
列出 ModelScope 上官方仓库的文件，看是否有比 s1v3 更新的 GPT。
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANDIDATE_REPOS = [
    "AI-ModelScope/GPT-SoVITS",
    "AI-ModelScope/GPT-SoVITS-Repack",
    "modelscope/GPT-SoVITS",
    "AI-ModelScope/gsv-v4-pretrained",
]

try:
    from modelscope.hub.api import HubApi
except ImportError:
    print("未安装 modelscope")
    sys.exit(1)

api = HubApi()

for rid in CANDIDATE_REPOS:
    print("=" * 78)
    print(f"仓库: {rid}")
    print("=" * 78)
    try:
        files = api.get_model_files(rid, recursive=True)
    except Exception as e:  # noqa: BLE001
        print(f"  查询失败: {type(e).__name__}: {str(e)[:150]}")
        print()
        continue

    print(f"  文件总数: {len(files)}")
    # 关注 GPT 相关
    gpt = [f for f in files
           if any(k in (f.get("Path") or "").lower()
                  for k in ("s1", "gpt", "bert25hz"))]
    print(f"  GPT 相关文件: {len(gpt)}")
    for f in gpt[:25]:
        sz = int(f.get("Size") or 0)
        print(f"    {f['Path']:70s} {sz/1e6:8.1f} MB")
    if len(gpt) > 25:
        print(f"    … 还有 {len(gpt)-25} 个")
    print()
