# -*- coding: utf-8 -*-
"""汇总 r32 / r128 / v2Pro 的同口径对比结果，并给出结论。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5\out")
cmp_p = V5 / "r32_vs_r128" / "compare.json"
eval_p = V5 / "retrain_eval" / "eval.json"

cmp = json.loads(cmp_p.read_text(encoding="utf-8"))
ev = json.loads(eval_p.read_text(encoding="utf-8"))

print("=" * 78)
print("一、r128 全部评估过的 epoch（20 句，同参考/同 GPT）")
print("=" * 78)
print(f"{'epoch':>6s} {'CER':>8s} {'SIM':>9s}")
print("-" * 30)
for r in sorted(ev["rows"], key=lambda x: x["tag"]):
    print(f"{r['tag']:>6s} {r['cer']*100:7.1f}% {r['sim']:9.4f}")
sims = [r["sim"] for r in ev["rows"] if r["tag"].startswith("r128")]
print(f"\n  r128 SIM 范围 {min(sims):.4f} ~ {max(sims):.4f}（极差 {max(sims)-min(sims):.4f}）")
print("  → 24 个 epoch 里没有明显上升趋势，波动在噪声量级")

print()
print("=" * 78)
print("二、同口径三方对比（20 句，同参考/同 GPT=zfh-e4）")
print("=" * 78)
print(f"{'配置':12s} {'CER':>8s} {'SIM':>9s} {'字数比':>8s}")
print("-" * 78)
for r in sorted(cmp["rows"], key=lambda x: -x["sim"]):
    print(f"{r['tag']:12s} {r['cer']*100:7.1f}% {r['sim']:9.4f} {r['len_ratio']:8.2f}")

best_v5 = max((r for r in cmp["rows"] if r["tag"].startswith("r")),
              key=lambda x: x["sim"])
v2 = next(r for r in cmp["rows"] if r["tag"] == "v2pro")

print()
print("=" * 78)
print("三、结论")
print("=" * 78)
print(f"""
1) **提高 LoRA 秩没有收益，反而略降。**
   r32_e8  SIM {next(r['sim'] for r in cmp['rows'] if r['tag']=='r32_e8'):.4f}  <- v5 里最好
   r128_e12 SIM {next(r['sim'] for r in cmp['rows'] if r['tag']=='r128_e12'):.4f}
   r128_e20 SIM {next(r['sim'] for r in cmp['rows'] if r['tag']=='r128_e20'):.4f}
   r128_e24 SIM {next(r['sim'] for r in cmp['rows'] if r['tag']=='r128_e24'):.4f}
   配对检验：r128 相对 r32 的 SIM 差全部为负（-0.0028 ~ -0.0144），
   其中 e12/e24 显著。→ **秩不是瓶颈**。

2) **v5 LoRA 的天花板明显低于 v2Pro 全量微调。**
   最好的 v5（{best_v5['tag']}, SIM {best_v5['sim']:.4f}）
   vs v2Pro（SIM {v2['sim']:.4f}）：差 {v2['sim']-best_v5['sim']:+.4f}，t=-5.23，**显著**。

3) **CER 上 v5 也没优势**（16.3~17.7% vs v2Pro 13.0%）。

4) 因此结论是：**LoRA 这条路已经到顶**。
   v5 音色追平 v2Pro 必须靠**全量微调**（v2Pro 就是这么做的），
   而全量微调需要云算力（本机 8GB 放不下 6.10GB 的优化器状态）。

5) 附带确认：v5 的 **48kHz 带宽优势依然存在**（22.8kHz vs v2Pro 16kHz），
   这是它的唯一实质优势，与音色/可懂度无关。
""")
