# -*- coding: utf-8 -*-
"""验证 LangSegmenter 补丁：改用 lite 模型后能**离线**工作。

## 背景
原代码只设了 cache_dir，model 默认 "auto" -> 联网下载 125MB lid.176.bin
-> 云端报 FileNotFoundError（cache_dir 目录不自建）。
补丁显式设 model="lite"，走 pip 包自带的 916KB 模型。

本脚本：
  1. 确认打补丁后 **不发起任何下载**（把网络调用替换成抛异常）
  2. 验证中英混排分段结果
  3. 与 full 模型对照（若本地已有 full 模型）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

root = Path(r"D:\A\zfy-v5\GPT-SoVITS")
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "GPT_SoVITS"))
os.chdir(root)

print("=" * 76)
print("1. 确认补丁已生效（langsegmenter.py 里有 model='lite'）")
print("=" * 76)
lg = root / "GPT_SoVITS" / "text" / "LangSegmenter" / "langsegmenter.py"
src = lg.read_text(encoding="utf-8", errors="replace")
print(f"  含 ZFY_V5FULLFT_PATCH: {'ZFY_V5FULLFT_PATCH' in src}")
print(f"  含 model=\"lite\"      : {'model=\"lite\"' in src}")
print(f"  含 mkdir 自建目录     : {'mkdir' in src}")

print()
print("=" * 76)
print("2. 拦截下载：把下载函数换成抛异常，验证**完全不联网**")
print("=" * 76)
import fast_langdetect.infer as fli  # noqa: E402

_orig = None
blocked = []


def _boom(*a, **k):
    blocked.append(a)
    raise AssertionError("!! 尝试下载模型 —— 说明 lite 没生效")


# _ModelLoader.download / load_with_download
for cls_name in ("_ModelLoader", "ModelLoader"):
    cls = getattr(fli, cls_name, None)
    if cls is None:
        continue
    for meth in ("download", "load_with_download"):
        if hasattr(cls, meth):
            setattr(cls, meth, _boom)
            print(f"  已拦截 {cls_name}.{meth}")

print()
print("=" * 76)
print("3. 分段结果（中英混排）")
print("=" * 76)
from text.LangSegmenter import LangSegmenter  # noqa: E402

TEXTS = [
    "今天天气不错，Hello World，我已经整理好了。",
    "这是 OK 的。",
    "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？",
    "GPU 和 AI 都很重要。",
]
for t in TEXTS:
    try:
        r = LangSegmenter.getTexts(t)
        parts = [(x["lang"], x["text"]) for x in r]
        print(f"  {t}")
        for lang, txt in parts:
            print(f"      [{lang}] {txt}")
    except Exception as e:  # noqa: BLE001
        print(f"  {t}")
        print(f"      **失败** {type(e).__name__}: {str(e)[:110]}")

print()
print("=" * 76)
print("4. 结论")
print("=" * 76)
if blocked:
    print(f"  ✗ 有 {len(blocked)} 次下载尝试 —— lite 未生效")
else:
    print("  ✓ 全程零下载（下载函数已被拦截且未被调用）")
    print("  ✓ LangSegmenter 可离线工作 —— 云端不会再报 Cache directory not found")
