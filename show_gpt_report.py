# -*- coding: utf-8 -*-
"""读 gpt_ab_report.json 并渲染四方对比表。

## 为什么要单独写
`compare_gpt.py` 的输出被我的 `Select-String` 过滤掉了汇总表
（过滤模式没匹配到中文表头）。直接从 JSON 读更可靠。

## 四个变体
    base             生产 zfh-e4.ckpt（v2 底模 + 81 条语料）
    A_lr2e3_e8       s1v3 底模, LR 2e-3, 8 轮（2200 步）
    B_lr2e3_e8       v2 底模,   LR 2e-3, 8 轮（2200 步）
    B_lr2e5_e8       v2 底模,   LR 2e-5, 8 轮（2200 步）  <- 降 LR 实验
    B_lr2e3_e2       v2 底模,   LR 2e-3, 2 轮（550 步）   <- 减轮数实验

## 判据（GPT 影响韵律，不影响音色）
    · 时长CV 越低 = 同类句子越一致（**最重要**）
    · 语速σ 越低 = 韵律越稳
    · 早停越少 = 越可靠
    · 语速应在 3~5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = Path(r"D:\A\zfy-v5\dist\gpt_ab\gpt_ab_report.json")
if not P.exists():
    print(f"  !! 找不到 {P}")
    sys.exit(1)
d = json.loads(P.read_text(encoding="utf-8"))
r = d["results"]

ORDER = ["base", "A_lr2e3_e8", "B_lr2e3_e8", "B_lr2e5_e8",
         "B_lr2e3_e2_e2", "A_lr2e3_e2", "B_lr2e3_e2"]
present = [k for k in ORDER if k in r]
extra = [k for k in r if k not in present]

print("=" * 104)
print("GPT 四方对比（**只换 GPT，SoVITS 固定 v2Pro**，20 句）")
print("=" * 104)
hdr = (f"  {'组':<22s} {'句数':>4s} {'平均时长':>9s} {'时长CV':>8s} "
       f"{'语速':>6s} {'语速σ':>7s} {'停顿段':>7s} {'静音s':>7s} {'早停':>5s}")
print(hdr)
print("  " + "-" * 98)
for k in present + extra:
    v = r[k]
    print(f"  {k:<22s} {v['n']:4d} {v['dur_mean']:8.2f}s {v['dur_cv']:7.1f}% "
          f"{v['speed_mean']:6.2f} {v['speed_std']:7.2f} "
          f"{v['sil_runs']:7.1f} {v['sil_sec']:7.2f} {v['early_stop']:5d}")

b = r.get("base")
if b:
    print()
    print("=" * 104)
    print("相对基线 base（生产 zfh-e4）")
    print("=" * 104)
    print(f"  {'组':<22s} {'Δ时长':>9s} {'Δ时长CV':>9s} {'Δ语速':>8s} "
          f"{'Δ语速σ':>9s} {'Δ早停':>7s}   判定")
    print("  " + "-" * 96)
    for k in present + extra:
        if k == "base":
            continue
        v = r[k]
        dc = v["dur_cv"] - b["dur_cv"]
        ds = v["speed_mean"] - b["speed_mean"]
        dsd = v["speed_std"] - b["speed_std"]
        de = v["early_stop"] - b["early_stop"]
        # 评分：CV 与语速σ 权重最高（GPT 直接影响韵律稳定性）
        score = 0
        score += 1 if dc < -1 else (-1 if dc > 1 else 0)
        score += 1 if dsd < -0.02 else (-1 if dsd > 0.02 else 0)
        score += 1 if de < 0 else (-1 if de > 0 else 0)
        verdict = ("✓ 优于 base" if score > 0
                   else ("✗ 差于 base" if score < 0 else "~ 与 base 相当"))
        print(f"  {k:<22s} {v['dur_mean']-b['dur_mean']:+8.2f}s {dc:+8.1f}% "
              f"{ds:+8.2f} {dsd:+9.2f} {de:+7d}   {verdict}")

print()
print("=" * 104)
print("判读")
print("=" * 104)
print("""
  两个实验设计：
    · **降 LR**：B_lr2e5_e8（LR 2e-5）vs B_lr2e3_e8（LR 2e-3）—— 其余全同
    · **减轮数**：B_lr2e3_e2_e2（2 轮 / 550 步）vs B_lr2e3_e8（8 轮 / 2200 步）

  关键问句：
    1. LR 2e-5 的时长CV 是否**接近 base（40.8%）**？
       -> 是 = 证明是 LR 问题，GPT 微调**值得继续**
       -> 否 = LR 不是主因
    2. 2 轮的时长CV 是否**好于 8 轮**？
       -> 是 = 证明**过拟合**（步数过多）
       -> 否 = 过拟合不是主因

  ⚠️ 注意：20 句的 CV 有噪声（技能里警告 CER 双峰分布，4 句差异可达 20pp）。
     但 **7 组一致的趋势**比单组数值更可信。
""")
print("=" * 104)
