# -*- coding: utf-8 -*-
"""用仓库自带的 load_sovits_new 正确读取带版本头的权重，判定微调方式。

`.pth` 前 2 字节是版本头（00-08），直接 torch.load 会
`UnpicklingError: unpickling stack underflow`。
`process_ckpt.load_sovits_new()` 会先剥掉头部。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, r"D:\A\zfy-v5\GPT-SoVITS")
sys.path.insert(0, r"D:\A\zfy-v5\GPT-SoVITS\GPT_SoVITS")
from GPT_SoVITS.process_ckpt import (  # noqa: E402
    load_sovits_new, get_sovits_version_from_path_fast,
)

TARGETS = [
    ("v2 SoVITS (生产)", r"D:\A\voiceclone\zfh-models\torch_weights\zfh_e6_s186.pth"),
    ("v2Pro SoVITS (生产)", r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh_v2pro_e12_s384.pth"),
    ("v2Pro SoVITS e10", r"D:\A\voiceclone\GPT-SoVITS\SoVITS_weights_v2Pro\zfh_v2pro_e10_s320.pth"),
    ("v5 LoRA e8", r"D:\A\zfy-v5\GPT-SoVITS\SoVITS_weights_v5\zfy_v5_e8_s3256_l32.pth"),
    ("v5 LoRA e1", r"D:\A\zfy-v5\GPT-SoVITS\SoVITS_weights_v5\zfy_v5_e1_s407_l32.pth"),
    ("v5 底模 s2Gv5dev", r"D:\A\zfy-v5\GPT-SoVITS\GPT_SoVITS\pretrained_models\gsv-v5-pretrained\s2Gv5dev.pth"),
    ("v2 底模 s2G2333k", r"D:\A\voiceclone\GPT-SoVITS\GPT_SoVITS\pretrained_models\gsv-v2final-pretrained\s2G2333k.pth"),
]

print("=" * 92)
print(f"{'模型':22s} {'版本头':>18s} {'体积':>9s} {'键数':>6s} {'参数量':>10s} "
      f"{'lora_A/B':>9s} {'判定':>12s}")
print("=" * 92)
rows = []
for label, p in TARGETS:
    path = Path(p)
    if not path.exists():
        print(f"{label:22s} {'[缺失]':>18s}  {p}")
        continue
    size = path.stat().st_size / 1e6
    try:
        ver = get_sovits_version_from_path_fast(str(path))
    except Exception as e:  # noqa: BLE001
        ver = f"<{type(e).__name__}>"
    try:
        w = load_sovits_new(str(path))
    except Exception as e:  # noqa: BLE001
        print(f"{label:22s} {str(ver):>18s} {size:8.1f}M  读取失败 {type(e).__name__}")
        continue
    if not isinstance(w, dict):
        print(f"{label:22s} {str(ver):>18s} {size:8.1f}M  非字典 {type(w)}")
        continue
    keys = list(w.keys())
    a = sum(1 for k in keys if "lora_A" in k)
    b = sum(1 for k in keys if "lora_B" in k)
    n = sum(v.numel() for v in w.values() if hasattr(v, "numel"))
    verdict = "LoRA" if (a or b) else "全量"
    rows.append((label, ver, size, len(keys), n, a + b, verdict))
    print(f"{label:22s} {str(ver):>18s} {size:8.1f}M {len(keys):6d} "
          f"{n/1e6:9.2f}M {a+b:9d} {verdict:>12s}")

print()
print("=" * 92)
print("说明")
print("=" * 92)
print("""
  · 「版本头」由 process_ckpt 解析，形如 ['v2','v2Pro',False]；
    最后一项 True 表示 **LoRA 权重**（如 v5dev/v5turbo 是 07/08）。
  · 判定 LoRA 的直接证据是权重里出现 `lora_A` / `lora_B` 低秩矩阵。
""")
