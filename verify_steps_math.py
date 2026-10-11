# -*- coding: utf-8 -*-
"""验证 notebook cell 22 里的"训练量诊断"算式是否正确。

不跑训练，只复算步数/参数比，确认诊断给出的结论站得住。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FULL_FEAT = Path(r"D:\A\zfy-v5\GPT-SoVITS\logs\zfy_v5")
V5_PARAMS, V2PRO_PARAMS, V2PRO_STEPS = 383.08e6, 42.39e6, 180

_n = len([l for l in (FULL_FEAT / "6-name2semantic.tsv")
          .read_text(encoding="utf-8", errors="replace").strip().split("\n")
          if l.strip()])
_n -= 1
print(f"样本数（去表头）: {_n}")

for BATCH in (16, 8, 4):
    spe = max(1, _n // BATCH)
    print(f"\n  batch={BATCH} -> 每 epoch {spe} 步")
    for EPOCHS in (8, 16, 24, 32, 48):
        total = spe * EPOCHS
        ratio = (total / V5_PARAMS) / (V2PRO_STEPS / V2PRO_PARAMS)
        lr = 1e-4 * 0.999875 ** total
        mark = "OK" if ratio >= 1.0 else "偏低"
        print(f"    e{EPOCHS:<3d} 总步 {total:5d}  步/参数 "
              f"{total/V5_PARAMS:.2e}  相对v2Pro {ratio:.2f}x  "
              f"末步LR {lr:.2e}  {mark}")

print()
print("=" * 66)
print("结论（注意：这里用 n//batch 估算；实测分桶采样器会多切几批）")
print("=" * 66)
print("""
  · 实测 e8 = 432 步 / 8 epoch = **54 步/epoch**，
    而 825//16 = 51 -> 估算偏低约 6%，实际跑完以权重文件名的 s{step} 为准
  · 按实测 54 步/epoch 重算（batch=16）：
      e8   432 步  -> 相对 v2Pro 0.27x   （已跑，疑似不足）
      e24  1296 步 -> 0.80x              （仍偏低）
      e32  1728 步 -> 1.06x              （达到）
  · lr_decay=0.999875 衰减极慢：1728 步后 LR 只从 1e-4 降到 8.0e-5
    -> 训练量几乎全靠步数堆，调 EPOCHS 是有效杠杆
  · 默认取 e32；每轮都存权重，最终**靠盲听挑**，别默认轮数越大越好
""")
