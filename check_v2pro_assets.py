# -*- coding: utf-8 -*-
"""检查 v2Pro 在云端合成所需的资产是否齐全。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\A\zfy-v5")
import pack_for_cloud as P  # noqa: E402

rels = set(rel.replace("\\", "/") for _, rel in P.collect())

print("=" * 72)
print("已打包清单里的 v2Pro / 共用资产")
print("=" * 72)
CHECK = [
    ("v2Pro SV 模型 (ERes2NetV2)",
     "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"),
    ("v5 声码器", "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth"),
    ("v5dev SoVITS 底模",
     "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"),
    ("s1v3 GPT", "GPT_SoVITS/pretrained_models/s1v3.ckpt"),
    ("生产参考音频", "ref/default.wav"),
    ("生产 GPT (zfh-e4)", "ref/zfh-e4.ckpt"),
    ("BigVGAN", "GPT_SoVITS/BigVGAN/bigvgan.py"),
    ("feature_extractor", "GPT_SoVITS/feature_extractor/cnhubert.py"),
    ("sv.py", "GPT_SoVITS/sv.py"),
    ("eres2net/kaldi.py", "GPT_SoVITS/eres2net/kaldi.py"),
    ("tools/AP_BWE_main", "tools/AP_BWE_main/models/model.py"),
]
for label, rel in CHECK:
    ok = rel in rels
    print(f"  {'OK  ' if ok else '缺失'} {label:30s} {rel}")

print()
print("=" * 72)
print("本地文件（上传源）")
print("=" * 72)
LOCAL = [
    ("生产 v2Pro SoVITS 权重",
     Path(r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh_v2pro_e12_s384.pth")),
    ("SV 模型",
     Path(r"D:\A\zfy-v5\GPT-SoVITS\GPT_SoVITS\pretrained_models\sv"
          r"\pretrained_eres2netv2w24s4ep4.ckpt")),
    ("生产参考音频",
     Path(r"D:\A\voiceclone\zfh-models-v2pro\ref\default.wav")),
    ("生产 GPT",
     Path(r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh-e4.ckpt")),
]
for label, p in LOCAL:
    if p.exists():
        print(f"  OK   {label:26s} {p.stat().st_size/1e6:8.1f} MB")
    else:
        print(f"  缺失 {label:26s} {p}")

print()
print("=" * 72)
print("结论")
print("=" * 72)
missing = [r for _, r in CHECK if r not in rels]
if missing:
    print("  需补进打包清单:")
    for m in missing:
        print(f"    {m}")
else:
    print("  已打包资产齐全")
print("""
  还需要**新上传**（本地有、远端没有）：
    · 生产 v2Pro SoVITS 权重 (~129 MB)  -> 云端才能合成 v2Pro 基准
  合并进 ref/ 目录，与 zfh-e4.ckpt 一起。
""")
