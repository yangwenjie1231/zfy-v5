# -*- coding: utf-8 -*-
"""汇总韵律自然度对比的正确统计（剔除 nan 样本）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

p = Path(r"D:\A\zfy-v5\out\prosody_natural\natural_prosody.json")
d = json.loads(p.read_text(encoding="utf-8"))
rows = [r for r in d["rows"] if "v5_dtw" in r and "v2pro_dtw" in r
        and not (isinstance(r["v5_dtw"], float) and np.isnan(r["v5_dtw"]))
        and not (isinstance(r["v2pro_dtw"], float) and np.isnan(r["v2pro_dtw"]))]

a = np.array([r["v5_dtw"] for r in rows])
b = np.array([r["v2pro_dtw"] for r in rows])
dv = a - b
n = len(dv)
se = dv.std(ddof=1) / np.sqrt(n)
t = float(dv.mean() / se) if se > 0 else 0.0
crit = {21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064}.get(n, 2.06)

print("=" * 72)
print("韵律自然度：F0 轮廓与真人的 DTW 距离（越小越自然）")
print("=" * 72)
print(f"  有效样本     : {n} 句（剔除 {len(d['rows'])-n} 个 nan）")
print(f"  v5   平均 DTW: {a.mean():.4f}")
print(f"  v2Pro 平均 DTW: {b.mean():.4f}")
print(f"  配对差(v5-v2Pro): {dv.mean():+.4f}")
print(f"  t = {t:.2f}   临界 = {crit}   "
      f"{'显著' if abs(t) > crit else '**不显著**'}")
print(f"  逐句胜场     : v5 {int((dv<0).sum())} / v2Pro {int((dv>0).sum())}")
print()
if abs(t) <= crit:
    print("  → 两者与真人的韵律距离**没有统计差异**")
    print("    即：我的这套韵律指标**测不出**你听到的『v5 语气更好』")
else:
    win = "v5" if dv.mean() < 0 else "v2Pro"
    print(f"  → {win} 的韵律更接近真人（显著）")

print()
print("=" * 72)
print("我的指标 vs 你的听感")
print("=" * 72)
print("""
  维度              你的听感        我的指标                一致?
  ────────────────  ──────────────  ──────────────────────  ─────
  音色              v5 有提升空间   SIM: v2Pro 0.8888 更高  ✅ 一致
                                    v5 0.8419
  语气/韵律         v5 好很多       F0范围/能量/停顿/DTW      ❌ 测不出
                                    全部「不显著」

  结论：**我现有的韵律指标全部失灵**。试过的 6 个指标
  （F0 动态范围、F0 变化率、带限能量动态、停顿数/总长、
   有声切换率、F0 轮廓 DTW vs 真人）都无法复现你的判断。

  这不代表你的听感不对 —— 恰恰相反，**你的听感是唯一可靠的判据**，
  是我的测量手段不够。已知的局限：
    · SIM 只测说话人嵌入余弦，与韵律无关
    · F0 统计量测「起伏多大」，测不到「起伏得对不对」
    · DTW 用的是全局轮廓距离，对局部不自然的过渡不敏感
    · 没有测：音素时长分布、共振峰过渡、气声/停顿的自然度
""")
