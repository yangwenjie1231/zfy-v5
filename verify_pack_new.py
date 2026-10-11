# -*- coding: utf-8 -*-
"""确认新增资产已进入打包清单，并算总体积。"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\A\zfy-v5")
import pack_for_cloud as P  # noqa: E402

items = P.collect()
rels = {rel.replace("\\", "/"): p for p, rel in items}
tot = sum(p.stat().st_size for p in rels.values())

print(f"总文件: {len(rels)}")
print(f"总体积: {tot/1e9:.2f} GB")
print()
print("=" * 74)
print("新增资产确认")
print("=" * 74)
CHECK = [
    "ref/v2pro_sovits.pth",
    "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
    "_tools/cloud_push_weights.py",
    "_tools/cloud_synth_sweep.py",
    "_tools/install_infer_deps.sh",
    "GPT_SoVITS/BigVGAN/bigvgan.py",
    "GPT_SoVITS/sv.py",
]
for k in CHECK:
    p = rels.get(k)
    tag = "OK  " if p else "缺失"
    sz = f"{p.stat().st_size/1e6:8.1f} MB" if p else ""
    print(f"  {tag} {k:62s} {sz}")

print()
print("=" * 74)
print("云端合成 v2Pro 所需资产链")
print("=" * 74)
CHAIN = [
    ("参考音频", "ref/default.wav"),
    ("参考文本", "ref/ref.txt"),
    ("GPT (zfh-e4)", "ref/zfh-e4.ckpt"),
    ("v2Pro SoVITS", "ref/v2pro_sovits.pth"),
    ("SV 模型 (ERes2NetV2)",
     "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"),
    ("sv.py", "GPT_SoVITS/sv.py"),
    ("BigVGAN", "GPT_SoVITS/BigVGAN/bigvgan.py"),
    ("feature_extractor", "GPT_SoVITS/feature_extractor/cnhubert.py"),
    ("AP_BWE_main", "tools/AP_BWE_main/models/model.py"),
]
missing = []
for label, k in CHAIN:
    ok = k in rels
    if not ok:
        missing.append(label)
    print(f"  {'OK  ' if ok else '缺失'} {label:24s} {k}")
print()
print("结论:", "✓ v2Pro 合成链路完整" if not missing
      else f"✗ 缺 {missing}")
