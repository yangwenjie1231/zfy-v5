# -*- coding: utf-8 -*-
"""「尾音浊」参数扫描 —— 目标：把末段指标压到真人附近。

## 当前差距（实测，末段 1.0s）
    指标            原版    处理后    真人      处理后差距
    H1-H2 dB        +9.24   +10.65   +13.61   **-2.96**  <- 核心（发声紧度）
    300-500 Hz       6.79     3.92     3.41    +0.51   ✓ 已对齐
    500-800 Hz      10.84     5.42     4.92    +0.50   ✓ 已对齐
    2000-3200 Hz     3.33     7.92     5.79    **+2.13**  <- 过冲（可能尖）

## 新增手段：F0 区提升（提高 H1-H2）
H1-H2 是「第一谐波 vs 第二谐波」的幅度差：
    F0 ≈ 200Hz（女声） -> H1 在 200Hz，H2 在 400Hz
    **提升 ~200Hz 相对 ~400Hz -> H1-H2 增大 -> 更"松/柔"，不"浊"**

⚠️ 但 200Hz 提升过多会"轰"。所以扫描里同时看：
    · H1-H2 是否靠近真人
    · 低频占比是否失控

## 扫描维度（54 组）
    boom_db    (300-500Hz 衰减) : -1, -3
    mud_db     (500-800Hz 衰减) : -3, -5, -7
    clarity_db (2000-3200 提升) : +2, +4, +6
    h1_db      (F0 区提升)      : 0, +2, +4

## 评分（越低越好）
    score = |ΔH1H2|/3.0 + |Δ300-500|/2.0 + |Δ500-800|/3.0 + |Δ2000-3200|/2.0
    （每项按"距真人的差距"归一化，pp 与 dB 分别定标）

## 输出
    dist/zhuo_sweep/report.txt
    dist/zhuo_sweep/best/       （最优参数的结果）
    dist/zhuo_sweep/listen/     （最优 vs 原版 交替，便于盲听）
"""
from __future__ import annotations

import itertools
import json
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
SRC = V5 / "dist" / "ref_swap_train" / "orig"
HUMAN = V5 / "GPT-SoVITS" / "ref" / "default.wav"
HUMAN_END = 6.95            # 真人主语音结束点（排除文本外多余段）
OUT = V5 / "dist" / "zhuo_sweep"
OUT.mkdir(parents=True, exist_ok=True)

# 真人目标（实测）
TARGET = {"h": 13.61, "300-500": 3.41, "500-800": 4.92, "2000-3200": 5.79}
SCALE = {"h": 3.0, "300-500": 2.0, "500-800": 3.0, "2000-3200": 2.0}

TAIL_SEC = 1.0


def load(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return np.asarray(x, dtype=np.float64), int(sr)


def voiced_end(x, sr, thresh_db=-45.0):
    nf = 256
    m = (len(x) // nf) * nf
    if m < nf * 4:
        return len(x)
    r = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
    ref = float(np.sqrt((x ** 2).mean())) + 1e-12
    v = np.where(r > ref * (10 ** (thresh_db / 20)))[0]
    return int((v[-1] + 1) * nf) if len(v) else len(x)


def peaking(x, sr, f0, q, db):
    if abs(db) < 0.05:
        return x
    try:
        from scipy.signal import filtfilt, iirpeak
        b, a = iirpeak(min(max(f0 / (sr / 2), 1e-4), 0.99), q)
        band = filtfilt(b, a, x)
    except Exception:  # noqa: BLE001
        return x
    g = 10 ** (db / 20)
    return (x - band) + band * g


def process(x, sr, prm):
    y = x.copy()
    end = voiced_end(y, sr)
    n = min(int(TAIL_SEC * sr), end)
    if n < int(0.05 * sr):
        return y
    a = end - n
    seg = y[a:end].copy()
    if prm["boom"] < -0.05:
        seg = peaking(seg, sr, 400.0, 1.0, prm["boom"])
    if prm["h1"] > 0.05:
        # F0 区提升（低 Q，覆盖 150~260Hz）
        seg = peaking(seg, sr, 200.0, 0.7, prm["h1"])
    if prm["mud"] < -0.05:
        seg = peaking(seg, sr, 650.0, 1.0, prm["mud"])
    if prm["clarity"] > 0.05:
        seg = peaking(seg, sr, 2700.0, 0.9, prm["clarity"])
    rn = min(int(0.08 * sr), len(seg))
    if rn > 8:
        r = np.linspace(0, 1, rn)
        seg[:rn] = seg[:rn] * r + y[a:a + rn] * (1 - r)
    y[a:end] = seg
    pk = float(np.abs(y).max())
    if pk > 0.99:
        y = y * (0.99 / pk)
    return y


def h1h2(x, sr):
    vals = []
    n = 2048
    for i in range(0, max(1, len(x) - n), n // 2):
        seg = x[i:i + n]
        if len(seg) < n or np.sqrt((seg ** 2).mean()) < 1e-3:
            continue
        S = np.abs(np.fft.rfft(seg * np.hanning(n)))
        f = np.fft.rfftfreq(n, 1 / sr)
        z = seg - seg.mean()
        ac = np.correlate(z, z, "full")[len(z) - 1:]
        ac /= (ac[0] + 1e-12)
        lo, hi = int(sr / 350), min(int(sr / 70), len(ac) - 1)
        if hi <= lo:
            continue
        f0 = sr / (lo + int(np.argmax(ac[lo:hi])))
        if not (70 <= f0 <= 350):
            continue

        def pk(fr):
            m = (f > fr * 0.75) & (f < fr * 1.25)
            return S[m].max() if m.sum() else np.nan
        a_, b_ = pk(f0), pk(2 * f0)
        if np.isfinite(a_) and np.isfinite(b_) and a_ > 0 and b_ > 0:
            vals.append(20 * np.log10(a_ / b_))
    return float(np.median(vals)) if vals else float("nan")


def bands(x, sr):
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
    for lo, hi in ((300, 500), (500, 800), (2000, 3200)):
        out[f"{lo}-{hi}"] = float(P[(f >= lo) & (f < hi)].sum())
    out["_low"] = float(P[f < 500].sum())
    return out


def measure(files):
    acc = {k: [] for k in ("h", "300-500", "500-800", "2000-3200", "_low")}
    for w, x, sr, y in files:
        e = voiced_end(x, sr)
        n = min(int(TAIL_SEC * sr), e)
        tb, ta = x[max(0, e - n):e], y[max(0, e - n):e]
        acc["h"].append(h1h2(ta, sr))
        for k, v in bands(ta, sr).items():
            acc[k].append(v)
    return {k: float(np.nanmean(v)) if v else float("nan")
            for k, v in acc.items()}


print("=" * 104)
print("「尾音浊」参数扫描")
print("=" * 104)
print(f"  目标（真人）: H1-H2 {TARGET['h']:+.2f}   "
      f"300-500 {TARGET['300-500']:.2f}%   "
      f"500-800 {TARGET['500-800']:.2f}%   "
      f"2000-3200 {TARGET['2000-3200']:.2f}%")

raw = []
for w in sorted(SRC.glob("*.wav")):
    x, sr = load(w)
    raw.append((w, x, sr, x))          # 原版（y=x）
base = measure(raw)
print(f"\n  原版: H1-H2 {base['h']:+.2f}   "
      f"300-500 {base['300-500']*100:.2f}%   "
      f"500-800 {base['500-800']*100:.2f}%   "
      f"2000-3200 {base['2000-3200']*100:.2f}%")

grid = list(itertools.product((-1.0, -3.0), (-3.0, -5.0, -7.0),
                              (2.0, 4.0, 6.0), (0.0, 2.0, 4.0)))
print(f"\n  扫描 {len(grid)} 组 ...")


def score(m):
    s = 0.0
    s += abs(m["h"] - TARGET["h"]) / SCALE["h"]
    for k in ("300-500", "500-800", "2000-3200"):
        s += abs(m[k] * 100 - TARGET[k]) / SCALE[k]
    return s


rows = []
for i, (bm, md, cl, h1) in enumerate(grid, 1):
    prm = {"boom": bm, "mud": md, "clarity": cl, "h1": h1}
    fs = []
    for w, x, sr, _ in raw:
        fs.append((w, x, sr, process(x, sr, prm)))
    m = measure(fs)
    rows.append({"boom": bm, "mud": md, "clarity": cl, "h1": h1,
                 "h": m["h"], "b300": m["300-500"] * 100,
                 "b500": m["500-800"] * 100, "b2k": m["2000-3200"] * 100,
                 "low": m["_low"] * 100, "score": score(m)})
    if i % 9 == 0 or i == len(grid):
        print(f"    ...{i}/{len(grid)}")

rows.sort(key=lambda r: r["score"])

print()
print("=" * 104)
print("TOP 15")
print("=" * 104)
print(f"  {'#':>3s} {'boom':>6s} {'mud':>6s} {'clar':>6s} {'h1':>5s} "
      f"{'H1-H2':>8s} {'300-500':>8s} {'500-800':>8s} {'2-3.2k':>8s} "
      f"{'评分':>7s}")
print("  " + "-" * 92)
tag = lambda r: f"{r['boom']:.0f}/{r['mud']:.0f}/{r['clarity']:.0f}/{r['h1']:.0f}"
print(f"  {'—':>3s} {'**原版**':>6s} {'':>6s} {'':>6s} {'':>5s} "
      f"{base['h']:+8.2f} {base['300-500']*100:8.2f} "
      f"{base['500-800']*100:8.2f} {base['2000-3200']*100:8.2f} "
      f"{score(base):7.4f}")
for i, r in enumerate(rows[:15], 1):
    print(f"  {i:3d} {r['boom']:6.0f} {r['mud']:6.0f} {r['clarity']:6.0f} "
          f"{r['h1']:5.0f} {r['h']:+8.2f} {r['b300']:8.2f} {r['b500']:8.2f} "
          f"{r['b2k']:8.2f} {r['score']:7.4f}")

best = rows[0]
print()
print("=" * 104)
print("最优参数")
print("=" * 104)
print(f"  boom {best['boom']:+.0f}dB@400Hz   mud {best['mud']:+.0f}dB@650Hz   "
      f"clarity {best['clarity']:+.0f}dB@2700Hz   h1 {best['h1']:+.0f}dB@200Hz")
print()
print(f"  {'指标':<14s} {'原版':>9s} {'最优':>9s} {'真人':>9s} {'Δ(最优-真人)':>13s}")
print("  " + "-" * 58)
for k, nm, key in (("h", "H1-H2 dB", "h"),
                   ("300-500", "300-500Hz%", "b300"),
                   ("500-800", "500-800Hz%", "b500"),
                   ("2000-3200", "2000-3200Hz%", "b2k")):
    bv = base[k] * (1 if k == "h" else 100)
    av = best[key]
    print(f"  {nm:<14s} {bv:9.2f} {av:9.2f} {TARGET[k]:9.2f} "
          f"{av-TARGET[k]:+13.2f}")

# 导出最优
bdir = OUT / "best"
bdir.mkdir(parents=True, exist_ok=True)
prm = {"boom": best["boom"], "mud": best["mud"],
       "clarity": best["clarity"], "h1": best["h1"]}
for w, x, sr, _ in raw:
    sf.write(str(bdir / w.name), process(x, sr, prm), sr)
print()
print(f"  最优结果 -> {bdir}")

# 盲听包：原版 vs 最优 交替
ldir = OUT / "listen"
ldir.mkdir(parents=True, exist_ok=True)
for w, x, sr, _ in raw:
    y = process(x, sr, prm)
    try:
        from math import gcd

        from scipy.signal import resample_poly
        def r48(v):
            if sr == 48000:
                return v
            g = gcd(int(sr), 48000)
            return resample_poly(v, 48000 // g, int(sr) // g)
        xa, ya = r48(x), r48(y)
    except Exception:  # noqa: BLE001
        xa, ya = x, y
    def nz(v):
        r = float(np.sqrt((v ** 2).mean())) + 1e-12
        v = v * (0.06 / r)
        pk = float(np.abs(v).max())
        return v * (0.99 / pk) if pk > 0.99 else v
    gap = np.zeros(int(0.7 * 48000))
    sf.write(str(ldir / f"{w.stem}_原版vs修复.wav"),
             np.concatenate([nz(xa), gap, nz(ya)]), 48000)
print(f"  盲听包（原版 -> 修复）-> {ldir}")

print()
print("=" * 104)
print("命令")
print("=" * 104)
print(f"  python _tools/fix_zhuo.py --in <输入> --out <输出> \\")
print(f"      --boom-db {best['boom']:.0f} --mud-db {best['mud']:.0f} \\")
print(f"      --clarity-db {best['clarity']:.0f}")

rep = ["=" * 80, "「尾音浊」参数扫描", "=" * 80, ""]
rep.append(f"目标(真人): H1-H2 {TARGET['h']:+.2f}  300-500 {TARGET['300-500']:.2f}%"
           f"  500-800 {TARGET['500-800']:.2f}%  2000-3200 {TARGET['2000-3200']:.2f}%")
rep.append(f"原版:       H1-H2 {base['h']:+.2f}  300-500 {base['300-500']*100:.2f}%"
           f"  500-800 {base['500-800']*100:.2f}%  2000-3200 {base['2000-3200']*100:.2f}%")
rep.append("")
rep.append(f"{'#':>3s} {'boom':>6s}{'mud':>6s}{'clar':>6s}{'h1':>5s}"
           f"{'H1-H2':>9s}{'300-500':>9s}{'500-800':>9s}{'2-3.2k':>9s}{'score':>8s}")
for i, r in enumerate(rows[:20], 1):
    rep.append(f"{i:3d} {r['boom']:6.0f}{r['mud']:6.0f}{r['clarity']:6.0f}"
               f"{r['h1']:5.0f}{r['h']:+9.2f}{r['b300']:9.2f}{r['b500']:9.2f}"
               f"{r['b2k']:9.2f}{r['score']:8.4f}")
rep.append("")
rep.append(f"最优: boom {best['boom']:.0f} mud {best['mud']:.0f} "
           f"clarity {best['clarity']:.0f} h1 {best['h1']:.0f}")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
(OUT / "sweep.json").write_text(json.dumps(
    {"target": TARGET, "base": {k: (v if k == "h" else v * 100)
                                for k, v in base.items()},
     "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  -> {OUT / 'report.txt'}")
print("=" * 104)
