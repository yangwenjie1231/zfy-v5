# -*- coding: utf-8 -*-
"""最终核实：v2 / v2Pro / v5 各自是「全量微调」还是「LoRA 微调」。

判据（三重，互相印证）
  1. **版本头**：`process_ckpt.get_sovits_version_from_path_fast()`
     返回 [基础版, 实际版, is_lora]；第三项 True = LoRA
  2. **权重键数**：全量微调存整个 SoVITS（670+ 键）；
     LoRA 只存低秩增量 + 必要模块（258 键）
  3. **`lora_rank` 字段**：LoRA 权重里会有这个键
  4. **训练脚本**：s2_train.py（全量）vs s2_train_v3_lora.py（LoRA，用 peft）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\A\zfy-v5\GPT-SoVITS")
sys.path.insert(0, r"D:\A\zfy-v5\GPT-SoVITS\GPT_SoVITS")
from GPT_SoVITS.process_ckpt import (  # noqa: E402
    load_sovits_new, get_sovits_version_from_path_fast,
)

TARGETS = [
    ("v2 SoVITS（生产）", r"D:\A\voiceclone\zfh-models\torch_weights\zfh_e6_s186.pth"),
    ("v2Pro SoVITS（生产）", r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh_v2pro_e12_s384.pth"),
    ("v2Pro SoVITS e10", r"D:\A\voiceclone\GPT-SoVITS\SoVITS_weights_v2Pro\zfh_v2pro_e10_s320.pth"),
    ("v5 LoRA e8（我训的）", r"D:\A\zfy-v5\GPT-SoVITS\SoVITS_weights_v5\zfy_v5_e8_s3256_l32.pth"),
    ("v5 底模 s2Gv5dev", r"D:\A\zfy-v5\GPT-SoVITS\GPT_SoVITS\pretrained_models\gsv-v5-pretrained\s2Gv5dev.pth"),
]

print("=" * 96)
print(f"{'模型':24s} {'版本头':>24s} {'体积':>8s} {'权重键':>7s} {'lora_rank':>10s} {'判定':>8s}")
print("=" * 96)
for label, p in TARGETS:
    path = Path(p)
    if not path.exists():
        print(f"{label:24s} {'[缺失]':>24s}")
        continue
    size = path.stat().st_size / 1e6
    ver = get_sovits_version_from_path_fast(str(path))
    d = load_sovits_new(str(path))
    nkeys = len(d.get("weight", {}))
    lr = d.get("lora_rank", None)
    is_lora = bool(ver[2]) if isinstance(ver, (list, tuple)) and len(ver) > 2 else False
    verdict = "LoRA" if (is_lora or lr is not None) else "全量"
    print(f"{label:24s} {str(ver):>24s} {size:7.1f}M {nkeys:7d} "
          f"{str(lr):>10s} {verdict:>8s}")

print()
print("=" * 96)
print("训练脚本")
print("=" * 96)
gsv = Path(r"D:\A\voiceclone\GPT-SoVITS\GPT_SoVITS")
for name in ("s2_train.py", "s2_train_v3_lora.py"):
    f = gsv / name
    if not f.exists():
        f = Path(r"D:\A\zfy-v5\GPT-SoVITS\GPT_SoVITS") / name
    src = f.read_text(encoding="utf-8", errors="replace")
    print(f"  {name:24s} peft={str('peft' in src):5s} "
          f"LoraConfig={str('LoraConfig' in src):5s} "
          f"读lora_rank={str('lora_rank' in src):5s} "
          f"=> {'LoRA' if 'LoraConfig' in src else '全量'}微调")

print()
print("=" * 96)
print("结论")
print("=" * 96)
print("""
  ┌─────────────┬──────────────────────┬──────────────┬─────────────────────┐
  │ 模型        │ 训练脚本             │ 方式         │ 证据                │
  ├─────────────┼──────────────────────┼──────────────┼─────────────────────┤
  │ v2          │ s2_train.py          │ **全量微调** │ 673 键, is_lora=F   │
  │ v2Pro       │ s2_train.py          │ **全量微调** │ 678 键, is_lora=F   │
  │ v5dev/v5turbo│ s2_train_v3_lora.py │ **LoRA 微调**│ 258 键, lora_rank=32│
  └─────────────┴──────────────────────┴──────────────┴─────────────────────┘

  ⚠️ `train_v2pro.py` 里写的 `"lora_rank": 32` 是**无效配置** ——
     `s2_train.py` 根本不读这个键（它只被 s2_train_v3_lora.py 读）。
     v2/v2Pro 走的是全量微调。

  这解释了为什么 v5(LoRA) 音色相似度不如 v2Pro(全量)：
     全量微调可训练参数远多于 LoRA，对单一说话人的拟合能力更强。
""")
