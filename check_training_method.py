# -*- coding: utf-8 -*-
"""核实 v2 / v2Pro / v5 各自是「全量微调」还是「LoRA 微调」。

判据（不靠猜，靠权重张量）：
  1. **文件体积**：LoRA 只存低秩增量（很小）；全量存整个模型
  2. **键名**：LoRA 会出现 `lora_A` / `lora_B` / `base_layer`
  3. **键数量**：LoRA 的键数通常与全量相当，但结构不同

同时核对训练脚本：
  · `s2_train.py`      = 全量（不读 lora_rank）
  · `s2_train_v3_lora.py` = LoRA（用 peft.LoraConfig）
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGETS = [
    ("v2 SoVITS (生产用)", r"D:\A\voiceclone\zfh-models\torch_weights\zfh_e6_s186.pth"),
    ("v2Pro SoVITS (生产用)", r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh_v2pro_e12_s384.pth"),
    ("v5 LoRA e8", r"D:\A\zfy-v5\GPT-SoVITS\SoVITS_weights_v5\zfy_v5_e8_s3256_l32.pth"),
    ("v5 底模 s2Gv5dev", r"D:\A\zfy-v5\GPT-SoVITS\GPT_SoVITS\pretrained_models\gsv-v5-pretrained\s2Gv5dev.pth"),
]

print("=" * 84)
print("权重张量分析")
print("=" * 84)
for label, p in TARGETS:
    path = Path(p)
    if not path.exists():
        print(f"\n{label}\n  [缺失] {p}")
        continue
    size_mb = path.stat().st_size / 1e6
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        print(f"\n{label}\n  读取失败: {type(e).__name__}: {e}")
        continue
    w = d.get("weight", d) if isinstance(d, dict) else d
    if not isinstance(w, dict):
        print(f"\n{label}\n  非字典结构: {type(w)}")
        continue
    keys = list(w.keys())
    lora_a = [k for k in keys if "lora_A" in k]
    lora_b = [k for k in keys if "lora_B" in k]
    base = [k for k in keys if "base_layer" in k]
    n_params = sum(v.numel() for v in w.values() if hasattr(v, "numel"))

    print(f"\n{label}")
    print(f"  文件        : {path.name}")
    print(f"  体积        : {size_mb:.1f} MB")
    print(f"  键数        : {len(keys)}")
    print(f"  总参数量    : {n_params/1e6:.2f} M")
    print(f"  lora_A 键   : {len(lora_a)}")
    print(f"  lora_B 键   : {len(lora_b)}")
    print(f"  base_layer  : {len(base)}")
    if lora_a or lora_b:
        n_lora = sum(w[k].numel() for k in lora_a + lora_b)
        print(f"  LoRA 参数量 : {n_lora/1e6:.2f} M "
              f"({n_lora/max(n_params,1)*100:.2f}% of total)")
        print(f"  判定        : **LoRA 微调**")
        print(f"  示例键      : {lora_a[:2] + lora_b[:2]}")
    else:
        print(f"  判定        : **全量微调**（无 lora_A/B 键）")

print()
print("=" * 84)
print("训练脚本核对")
print("=" * 84)
gsv = Path(r"D:\A\voiceclone\GPT-SoVITS\GPT_SoVITS")
for name, note in [("s2_train.py", "v2/v2Pro 用这个"),
                   ("s2_train_v3_lora.py", "v3/v4/v5 用这个")]:
    f = gsv / name
    if not f.exists():
        print(f"  {name:24s} [缺失]")
        continue
    src = f.read_text(encoding="utf-8", errors="replace")
    has_peft = "peft" in src
    has_lora_cfg = "LoraConfig" in src
    reads_rank = "lora_rank" in src
    print(f"  {name:24s} ({note})")
    print(f"      import peft     : {has_peft}")
    print(f"      LoraConfig      : {has_lora_cfg}")
    print(f"      读取 lora_rank  : {reads_rank}")
    print(f"      => {'LoRA 微调' if has_lora_cfg else '全量微调'}")

print()
print("=" * 84)
print("结论")
print("=" * 84)
print("""
  v2 / v2Pro  -> s2_train.py            = **全量微调**
                 （train_v2pro.py 里写的 lora_rank=32 是无效配置，
                   s2_train.py 根本不读这个键）

  v5dev/v5turbo -> s2_train_v3_lora.py  = **LoRA 微调**
                 （webui.py 的分支决定：非 v1/v2/v2Pro/v2ProPlus 全走 LoRA）
""")
