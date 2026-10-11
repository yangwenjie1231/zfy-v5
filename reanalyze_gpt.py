# -*- coding: utf-8 -*-
"""修正指标：用「时长比值」的 CV，而不是「原始时长」的 CV。

## 我之前的指标缺陷
`compare_gpt.py` 报的 `时长CV = std(时长)/mean(时长)`。
但 20 句测试文本**长度差异极大**：
    "好。"              2 字
    "再好的武器，也得..." 35 字
-> 时长从 0.6s 到 8.6s，**CV 必然很大**（40%+），
   而这个 CV **主要反映句子长度差异，不是韵律稳定性**。

## 正确指标：时长比值
    ratio = 实际时长 / (字数 / 4.0)
其中 4.0 是"正常中文语速"基准（技能里给的 3~5 字/秒的中点）。
`ratio` 把长度因素除掉后，**其离散度才反映韵律稳定性**：
    · ratio ≈ 1.0      -> 语速正常
    · std(ratio) 小     -> 同类句子时长一致（韵律稳）
    · std(ratio) 大     -> 韵律飘

## 本脚本
从已保存的 wav 重新计算：
    · mean(ratio) / std(ratio) / CV(ratio)   <- 核心
    · 语速 mean/std
    · 早停句数（ratio < 0.6 或 语速 > 7）
    · 静音段数/静音时长

## 用法
    python reanalyze_gpt.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
AB = V5 / "dist" / "gpt_ab"

TEXTS = [
    "让我想想。", "好。",
    "迎敌或是迎客，都得整装后再出发。",
    "出门前，要不要先喝杯茶提提神？", "嗯……我理一理。",
    "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？",
    "武陵城外有一片梨树林，每年春天都会开满白色的花。",
    "她就在那阵花香里，只是他自己不知道。",
    "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。",
    "这件事我三点五十分再去处理。",
    "管理员，今天的实验数据我已经整理好了。",
    "清波城那边的情况，我一直在关注。",
    "若是要谈，就该拿出诚意来。",
    "我答应你，会好好照顾自己。",
    "这些话，我本想留到以后再说的。",
    "你来了。坐吧，茶还热着。",
    "有些事，不是努力就能改变的。",
    "但不去试，就永远不知道结果。",
    "汤汤说，我是个念旧的人。",
    "或许吧。毕竟这座城，承载了太多。",
]
STRIP = re.compile(r"[，。？！、…,.?!\s]")


def silence_stats(x: np.ndarray, sr: int, thresh_db: float = -45.0):
    nf = 256
    if len(x) < nf * 4:
        return 0, 0.0
    m = (len(x) // nf) * nf
    rmsf = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
    ref = float(np.sqrt((x ** 2).mean())) + 1e-12
    sil = rmsf < ref * (10 ** (thresh_db / 20))
    runs, prev = 0, False
    for v in sil:
        if v and not prev:
            runs += 1
        prev = v
    return runs, float(sil.sum() * nf / sr)


print("=" * 100)
print("修正指标：时长比值 ratio = 实际时长 / (字数/4.0)")
print("=" * 100)
print("  ratio ≈ 1.0 = 语速正常（4 字/秒）")
print("  std(ratio) 小 = 韵律稳（**核心指标**）")
print()

rows = {}
for d in sorted(AB.iterdir()):
    if not d.is_dir():
        continue
    vals = []
    for w in sorted(d.glob("*.wav")):
        m = re.match(r"(\d+)", w.stem)
        if not m:
            continue
        i = int(m.group(1)) - 1
        if not (0 <= i < len(TEXTS)):
            continue
        try:
            x, sr = sf.read(str(w), always_2d=True, dtype="float64")
        except Exception:  # noqa: BLE001
            continue
        x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
        if len(x) < sr * 0.2:
            continue
        dur = len(x) / sr
        n_ch = len(STRIP.sub("", TEXTS[i]))
        runs, sil_t = silence_stats(x, sr)
        vals.append({
            "i": i + 1, "dur": dur, "n_ch": n_ch,
            "speed": n_ch / dur,
            "ratio": dur / (n_ch / 4.0),
            "sil_runs": runs, "sil_sec": sil_t,
        })
    if vals:
        rows[d.name] = vals

if not rows:
    print("  !! 没有数据（先跑 compare_gpt.py）")
    sys.exit(1)

ORDER = ["base", "A_lr2e3_e8", "B_lr2e3_e8", "B_lr2e5_e8",
         "B_lr2e3_e2_e2", "A_lr2e3_e2", "B_lr2e3_e2", "B_lr2e5_e2"]
keys = [k for k in ORDER if k in rows] + [k for k in rows if k not in ORDER]

print("=" * 100)
print("修正后的对比表")
print("=" * 100)
print(f"  {'组':<22s} {'句数':>4s} {'ratio均值':>9s} {'ratioσ':>8s} "
      f"{'ratioCV':>8s} {'语速':>6s} {'语速σ':>7s} {'停顿段':>7s} {'早停':>5s}")
print("  " + "-" * 96)
summ = {}
for k in keys:
    v = rows[k]
    rat = np.array([x["ratio"] for x in v])
    sp = np.array([x["speed"] for x in v])
    early = int(sum(1 for x in v if x["ratio"] < 0.6 or x["speed"] > 7))
    summ[k] = {
        "n": len(v),
        "ratio_mean": float(rat.mean()),
        "ratio_std": float(rat.std()),
        "ratio_cv": float(rat.std() / rat.mean() * 100),
        "speed_mean": float(sp.mean()),
        "speed_std": float(sp.std()),
        "sil_runs": float(np.mean([x["sil_runs"] for x in v])),
        "early": early,
    }
    s = summ[k]
    print(f"  {k:<22s} {s['n']:4d} {s['ratio_mean']:9.3f} {s['ratio_std']:8.3f} "
          f"{s['ratio_cv']:7.1f}% {s['speed_mean']:6.2f} {s['speed_std']:7.2f} "
          f"{s['sil_runs']:7.1f} {s['early']:5d}")

b = summ.get("base")
if b:
    print()
    print("=" * 100)
    print("相对 base（生产 zfh-e4）")
    print("=" * 100)
    print(f"  {'组':<22s} {'Δratio':>8s} {'Δratioσ':>9s} {'ΔratioCV':>10s} "
          f"{'Δ语速':>8s} {'Δ早停':>7s}   判定")
    print("  " + "-" * 94)
    for k in keys:
        if k == "base":
            continue
        s = summ[k]
        dr = s["ratio_mean"] - b["ratio_mean"]
        drs = s["ratio_std"] - b["ratio_std"]
        drc = s["ratio_cv"] - b["ratio_cv"]
        dsp = s["speed_mean"] - b["speed_mean"]
        de = s["early"] - b["early"]
        # 评分：ratioσ 与 ratioCV 是韵律稳定性的直接度量
        score = 0
        score += 1 if drs < -0.01 else (-1 if drs > 0.01 else 0)
        score += 1 if abs(dr) < 0.03 else (-1 if abs(dr) > 0.08 else 0)
        score += 1 if de < 0 else (-1 if de > 0 else 0)
        verdict = ("✓ 优于 base" if score > 0
                   else ("✗ 差于 base" if score < 0 else "~ 相当"))
        print(f"  {k:<22s} {dr:+8.3f} {drs:+9.3f} {drc:+9.1f}% "
              f"{dsp:+8.2f} {de:+7d}   {verdict}")

print()
print("=" * 100)
print("两个实验的直接对比")
print("=" * 100)
pairs = [
    ("降 LR", "B_lr2e3_e8", "B_lr2e5_e8"),
    ("减轮数", "B_lr2e3_e8", "B_lr2e3_e2_e2"),
]
for name, a, c in pairs:
    if a in summ and c in summ:
        sa, sc = summ[a], summ[c]
        print(f"\n  【{name}】{a}  ->  {c}")
        print(f"    ratioσ   {sa['ratio_std']:.3f} -> {sc['ratio_std']:.3f}  "
              f"({sc['ratio_std']-sa['ratio_std']:+.3f})")
        print(f"    ratioCV  {sa['ratio_cv']:.1f}% -> {sc['ratio_cv']:.1f}%  "
              f"({sc['ratio_cv']-sa['ratio_cv']:+.1f}%)")
        print(f"    语速     {sa['speed_mean']:.2f} -> {sc['speed_mean']:.2f}")
        print(f"    早停     {sa['early']} -> {sc['early']}")

print()
print("=" * 100)
print("结论")
print("=" * 100)
if b:
    best = min((summ[k]["ratio_std"], k) for k in keys)
    print(f"  ratioσ 最小（韵律最稳）: **{best[1]}**  ({best[0]:.3f})")
    print(f"  base 的 ratioσ        : {b['ratio_std']:.3f}")
    print()
    print(f"  base 的 ratio 均值     : {b['ratio_mean']:.3f}  "
          f"（1.0 = 4 字/秒）")
    for k in keys:
        if k == "base":
            continue
        print(f"    {k:<22s} ratio={summ[k]['ratio_mean']:.3f}  "
              f"σ={summ[k]['ratio_std']:.3f}")
print("=" * 100)

out = AB / "reanalyze_report.json"
out.write_text(json.dumps({"summary": summ, "per_file": rows},
                          ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  -> {out}")
