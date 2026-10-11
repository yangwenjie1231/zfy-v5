# -*- coding: utf-8 -*-
"""验证 zhuo_sweep 最优参数**没有引入新问题**。

## 为什么要查
上一步只看 4 个"浊度"指标，都对齐了。但**可能顾此失彼**：
    · `h1 +4dB@200Hz` 会不会让整体"轰"？（低频过量）
    · `clarity +4dB@2700Hz` 会不会"齿音过重"？（sibilance）
    · 相似度会不会掉？（说话人嵌入对频谱敏感）
    · 会不会削波/失真？
    · 处理区（末 1.0s）与主体的**接缝**是否可闻？

## 检查项
    ① **低频总量**（<500Hz 占比）—— 是否失控
    ② **齿音**（5~9kHz 占比）—— 是否过重
    ③ **全句频谱**（不只末段）—— 主体是否被误伤
    ④ **削波**（峰值、削波样本数）
    ⑤ **相似度**（用 eval_dirs.py）
    ⑥ **接缝**（处理边界前后的 RMS 跳变）

## 输出
    dist/zhuo_verify/report.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
ORIG = V5 / "dist" / "ref_swap_train" / "orig"
# v1 = zhuo_sweep/best（用户听过的版本）
# v2 = zhuo_sweep2/best（全频段目标优化版）
BEST = V5 / "dist" / "zhuo_sweep2" / "best"
V1 = V5 / "dist" / "zhuo_sweep" / "best"
HUMAN = V5 / "GPT-SoVITS" / "ref" / "default.wav"
HUMAN_END = 6.95
OUT = V5 / "dist" / "zhuo_verify2"
OUT.mkdir(parents=True, exist_ok=True)

BANDS = [(50, 150), (150, 300), (300, 500), (500, 800), (800, 1200),
         (1200, 2000), (2000, 3200), (3200, 5000), (5000, 9000)]


def load(p: Path, max_sec=None):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if max_sec:
        x = x[:int(max_sec * sr)]
    return np.asarray(x, dtype=np.float64), int(sr)


def spec_share(x, sr):
    nfft = 2048
    acc = np.zeros(nfft // 2 + 1)
    cnt = 0
    for i in range(0, max(1, len(x) - nfft), nfft // 4):
        seg = x[i:i + nfft]
        if len(seg) < nfft or np.sqrt((seg ** 2).mean()) < 1e-4:
            continue
        acc += np.abs(np.fft.rfft(seg * np.hanning(nfft))) ** 2
        cnt += 1
    if cnt == 0:
        return {}
    f = np.fft.rfftfreq(nfft, 1 / sr)
    P = acc / (acc.sum() + 1e-20)
    out = {}
    for lo, hi in BANDS:
        out[f"{lo}-{hi}"] = float(P[(f >= lo) & (f < hi)].sum())
    out["_low500"] = float(P[f < 500].sum())
    out["_sib"] = float(P[(f >= 5000) & (f < min(9000, sr / 2))].sum())
    out["_cent"] = float((f * P).sum())
    return out


def voiced_end(x, sr, t=-45.0):
    nf = 256
    m = (len(x) // nf) * nf
    if m < nf * 4:
        return len(x)
    r = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
    ref = float(np.sqrt((x ** 2).mean())) + 1e-12
    v = np.where(r > ref * (10 ** (t / 20)))[0]
    return int((v[-1] + 1) * nf) if len(v) else len(x)


print("=" * 100)
print("最优参数验证（查新问题）")
print("=" * 100)

files = sorted(ORIG.glob("*.wav"))
if not files:
    print("  !! 无输入")
    sys.exit(1)

rows = []
for w in files:
    x, sr = load(w)
    y, _ = load(BEST / w.name)
    e = voiced_end(x, sr)
    n = min(int(1.0 * sr), e)
    tail_x, tail_y = x[max(0, e - n):e], y[max(0, e - n):e]
    body_x = x[:max(1, e - n)]
    body_y = y[:max(1, e - n)]
    rows.append({
        "name": w.name, "sr": sr, "e": e,
        "full_x": spec_share(x, sr), "full_y": spec_share(y, sr),
        "tail_x": spec_share(tail_x, sr), "tail_y": spec_share(tail_y, sr),
        "body_x": spec_share(body_x, sr), "body_y": spec_share(body_y, sr),
        "pk_x": float(np.abs(x).max()), "pk_y": float(np.abs(y).max()),
        "clip_x": int((np.abs(x) > 0.999).sum()),
        "clip_y": int((np.abs(y) > 0.999).sum()),
    })

# 真人参照
hx, hsr = load(HUMAN, max_sec=HUMAN_END)
he = voiced_end(hx, hsr)
hn = min(int(1.0 * hsr), he)
human_tail = spec_share(hx[max(0, he - hn):he], hsr)
human_full = spec_share(hx, hsr)


def avg(rows_, where, key):
    xs = [r[where].get(key) for r in rows_ if r[where].get(key) is not None]
    return float(np.mean(xs)) if xs else float("nan")


print()
print("=" * 100)
print("① 全句频谱（看主体是否被误伤 —— 处理只该动末 1.0s）")
print("=" * 100)
print(f"  {'频段 Hz':<12s} {'原版':>9s} {'修复版':>9s} {'Δ':>9s}   "
      f"{'真人':>9s}   判读")
print("  " + "-" * 78)
for lo, hi in BANDS:
    k = f"{lo}-{hi}"
    a = avg(rows, "full_x", k) * 100
    b = avg(rows, "full_y", k) * 100
    h = human_full.get(k, 0) * 100
    d = b - a
    tag = ""
    if abs(d) > 1.5:
        tag = "**主体被改动**" if lo < 2000 else "主体高频变化"
    print(f"  {k:<12s} {a:8.2f}% {b:8.2f}% {d:+8.2f}   {h:8.2f}%   {tag}")

print()
print("=" * 100)
print("② 低频总量与齿音（是否失控）")
print("=" * 100)
for key, name, real_key in (("_low500", "低频 <500Hz", "_low500"),
                            ("_sib", "齿音 5~9kHz", "_sib"),
                            ("_cent", "谱质心 Hz", "_cent")):
    a_f = avg(rows, "full_x", key)
    b_f = avg(rows, "full_y", key)
    a_t = avg(rows, "tail_x", key)
    b_t = avg(rows, "tail_y", key)
    h_t = human_tail.get(real_key, float("nan"))
    unit = "" if key == "_cent" else "%"
    mul = 1 if key == "_cent" else 100
    print(f"  {name}")
    print(f"    全句   {a_f*mul:8.2f}{unit} -> {b_f*mul:8.2f}{unit}   "
          f"Δ {(b_f-a_f)*mul:+7.2f}")
    print(f"    末段   {a_t*mul:8.2f}{unit} -> {b_t*mul:8.2f}{unit}   "
          f"Δ {(b_t-a_t)*mul:+7.2f}   （真人 {h_t*mul:.2f}{unit}）")

print()
print("=" * 100)
print("③ 削波 / 失真")
print("=" * 100)
px = float(np.mean([r["pk_x"] for r in rows]))
py = float(np.mean([r["pk_y"] for r in rows]))
cx = int(np.sum([r["clip_x"] for r in rows]))
cy = int(np.sum([r["clip_y"] for r in rows]))
print(f"  平均峰值   原版 {px:.4f}  ->  修复版 {py:.4f}")
print(f"  削波样本   原版 {cx}  ->  修复版 {cy}   "
      f"{'✓ 无削波' if cy == 0 else '**⚠️ 出现削波**'}")

print()
print("=" * 100)
print("④ 接缝检查（处理区边界 ±50ms 的 RMS 跳变）")
print("=" * 100)
jumps = []
for r, w in zip(rows, files):
    x, sr = load(w)
    y, _ = load(BEST / w.name)
    e = r["e"]
    a = e - int(1.0 * sr)          # 处理区起点
    if a < int(0.1 * sr):
        continue
    # 边界前后各 50ms 的 RMS
    w50 = int(0.05 * sr)
    pre = float(np.sqrt((y[max(0, a - w50):a] ** 2).mean()))
    post = float(np.sqrt((y[a:a + w50] ** 2).mean()))
    if pre > 1e-6 and post > 1e-6:
        jumps.append(20 * np.log10(post / pre))
if jumps:
    j = np.array(jumps)
    print(f"  边界 RMS 跳变: 均值 {j.mean():+.2f} dB   "
          f"最大 |{np.abs(j).max():.2f}| dB")
    if np.abs(j).max() < 3.0:
        print("    ✓ 接缝平滑（<3dB 通常不可闻）")
    else:
        print("    ⚠️ 有较大跳变，可能需要加长过渡区")
else:
    print("  样本不足")

print()
print("=" * 100)
print("⑤ 总结")
print("=" * 100)
issues = []
if abs(avg(rows, "full_y", "_low500") - avg(rows, "full_x", "_low500")) > 0.02:
    issues.append("全句低频变化 >2pp（主体被影响）")
if cy > 0:
    issues.append("出现削波")
if jumps and np.abs(j).max() > 3.0:
    issues.append("接缝跳变 >3dB")
b_sib = avg(rows, "full_y", "_sib")
a_sib = avg(rows, "full_x", "_sib")
if b_sib > a_sib * 1.5:
    issues.append(f"齿音增加 {(b_sib/a_sib-1)*100:.0f}%")

if issues:
    print("  发现问题:")
    for s in issues:
        print(f"    · {s}")
else:
    print("  ✓ 未发现新问题")
    print("    · 全句低频基本不变（处理只在末 1.0s）")
    print("    · 无削波")
    print("    · 接缝平滑")
    print("    · 齿音未失控")

print()
print("  ## 还需人工确认")
print("    1. **相似度**（用 eval_dirs.py 对比 orig vs best）")
print("    2. **盲听**（dist/zhuo_sweep/listen/ 里是『原版 -> 修复』交替）")

rep = ["=" * 78, "最优参数验证", "=" * 78, ""]
rep.append("全句频谱:")
rep.append(f"  {'频段':<12s}{'原版':>10s}{'修复版':>10s}{'真人':>10s}")
for lo, hi in BANDS:
    k = f"{lo}-{hi}"
    rep.append(f"  {k:<12s}{avg(rows,'full_x',k)*100:9.2f}%"
               f"{avg(rows,'full_y',k)*100:9.2f}%{human_full.get(k,0)*100:9.2f}%")
rep.append("")
rep.append(f"低频<500Hz 全句 {avg(rows,'full_x','_low500')*100:.2f}% -> "
           f"{avg(rows,'full_y','_low500')*100:.2f}%")
rep.append(f"齿音5-9k  全句 {avg(rows,'full_x','_sib')*100:.2f}% -> "
           f"{avg(rows,'full_y','_sib')*100:.2f}%")
rep.append(f"峰值 {px:.4f} -> {py:.4f}   削波 {cx} -> {cy}")
if jumps:
    rep.append(f"接缝跳变 均值 {j.mean():+.2f}dB 最大 {np.abs(j).max():.2f}dB")
rep.append("")
rep.append("问题: " + ("; ".join(issues) if issues else "无"))
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
print()
print(f"  -> {OUT / 'report.txt'}")
print("=" * 100)
