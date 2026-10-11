# -*- coding: utf-8 -*-
"""确认打包清单里各推理必需目录的覆盖情况。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, r"D:\A\zfy-v5")
import pack_for_cloud as P  # noqa: E402

rels = [rel.replace("\\", "/") for _, rel in P.collect()]
print("=== 推理必需目录的打包覆盖 ===")
PATTERNS = [
    "GPT_SoVITS/BigVGAN",
    "GPT_SoVITS/feature_extractor",
    "GPT_SoVITS/Accel",
    "GPT_SoVITS/eres2net",
    "GPT_SoVITS/module",
    "GPT_SoVITS/AR",
    "GPT_SoVITS/TTS_infer_pack",
    "GPT_SoVITS/text",
    "tools/AP_BWE_main",
    "tools/acceleration.py",
    "tools/portable_runtime.py",
    "tools/audio_sr.py",
    "tools/my_utils.py",
    "GPT_SoVITS/sv.py",
]
for pat in PATTERNS:
    n = sum(1 for r in rels if r == pat or r.startswith(pat + "/"))
    status = "OK" if n else "缺失"
    print(f"  {status:4s} {pat:36s} {n:5d} 个文件")

print()
print("总文件数:", len(rels))
