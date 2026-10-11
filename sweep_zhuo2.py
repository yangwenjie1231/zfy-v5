# -*- coding: utf-8 -*-
"""「尾音浊」优化 v2 —— 用**全频段距离**做目标，避免顾此失彼。

## v1 的错误（我的验证脚本抓到了）
`sweep_zhuo.py` 只优化 4 个窄带指标（H1-H2 / 300-500 / 500-800 / 2000-3200），
结果**别的指标变差**：
    末段谱质心  原版 764 -> 修复 718   （真人 **1073**）← 更远
    末段齿音    原版 2.28% -> 1.86%   （真人 **5.33%**）← 更远
    全句 150-300Hz +5.18pp             ← 低频轰
**根因**：`h1 +4dB@200Hz` 堆了低频，与"谱质心要高、齿音要足"**方向冲突**。

## v2 的目标函数：全频段加权距离
对末段，把 9 个频段 + 谱质心 + H1-H2 都纳入，用**归一化距离**求和：

    score = Σ_band  w_b · |share_model - share_human| / scale_b
          + w_c · |centroid_model - centroid_human| / 200
          + w_h · |H1H2_model - H1H2_human| / 3

权重设计（避免"顾此失彼"）：
    · 每个频段都有权重（不再只盯 4 个）
    · 谱质心权重加大（它是"明亮度"的总量指标）
    · H1-H2 权重适中（它是发声方式，滤波只能近似）

## 搜索空间
    boom_db    : 0, -2, -4          (400Hz)
    mud_db     : -2, -4, -6         (650Hz)
    clarity_db : +2, +4, +6         (2700Hz)
    h1_db      : 0, +1.5, +3        (200Hz, Q 0.7)   <- 收窄上限，避免轰
    air_db     : 0, +2, +4          (6000Hz)         <- 新增：补齿音/空气感
共 3^5 = 243 组

## 接缝
过渡区从 80ms 加长到 **150ms**，并用**等功率交叉淡化**。

## 输出
    dist/zhuo_sweep2/report.txt
    dist/zhuo_sweep2/best/
    dist/zhuo_sweep2/listen/     原版 -> v1修复 -> v2修复（三段对比）
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
SRC = V5 / "dist" / "ref_swap_train" / "orig"
V1 = V5 / "dist" / "zhuofix"                  # 用户听过的版本
HUMAN = V5 / "GPT-SoVITS" / "ref" / "default.wav"
HUMAN_END = 6.95
OUT = V5 / "dist" / "zhuo_sweep2"
OUT.mkdir(parents=True, exist_ok=True)
TAIL_SEC = 1.0

BANDS = [(50, 150), (150, 300), (300, 500), (500, 800), (800, 1200),
         (1200, 2000), (2000, 3200), (3200, 5000), (5000, 9000)]
# 每段的容差（pp）—— 低频段容差大（本身占比高），高频段容差小
SCALE = {f"{lo}-{hi}": (6.0 if hi <= 500 else 3.0 if hi <= 2000 else 1.5)
         for lo, hi in BANDS}


def load(p: Path, max_sec=None):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if max_sec:
        x = x[:int(max_sec * sr)]
    return np.asarray(x, dtype=np.float64), int(sr)


def voiced_end(x, sr, t=-45.0):
    nf = 256
    m = (len(x) // nf) * nf
    if m < nf * 4:
        return len(x)
    r = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
    ref = float(np.sqrt((x ** 2).mean())) + 1e-12
    v = np.where(r > ref * (10 ** (t / 20)))[0]
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


RAMP_SEC = 0.15          # v1 是 0.08，加长到 0.15


def process(x, sr, prm):
    y = x.copy()
    end = voiced_end(y, sr)
    n = min(int(TAIL_SEC * sr), end)
    if n < int(0.05 * sr):
        return y
    a = end - n
    orig = y[a:end].copy()
    seg = orig.copy()
    for key, f0, q in (("boom", 400.0, 1.0), ("h1", 200.0, 0.7),
                       ("mud", 650.0, 1.0), ("clarity", 2700.0, 0.9),
                       ("air", 6000.0, 0.8)):
        if abs(prm.get(key, 0.0)) > 0.05:
            seg = peaking(seg, sr, f0, q, prm[key])
    # 等功率交叉淡化（v1 是线性，且只有 80ms）
    rn = min(int(RAMP_SEC * sr), len(seg))
    if rn > 8:
        t = np.linspace(0, 1, rn)
        # equal-power：cos/sin 交叉
        g_in = np.cos(t * np.pi / 2)
        g_out = np.sin(t * np.pi / 2)
        seg[:rn] = seg[:rn] * g_out + orig[:rn] * g_in
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


def spec(x, sr):
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
        return None, None
    f = np.fft.rfftfreq(nfft, 1 / sr)
    P = acc / (acc.sum() + 1e-20)
    return f, P


def feats(x, sr):
    f, P = spec(x, sr)
    if f is None:
        return None
    out = {f"{lo}-{hi}": float(P[(f >= lo) & (f < hi)].sum()) * 100
           for lo, hi in BANDS}
    out["_cent"] = float((f * P).sum())
    out["_h"] = h1h2(x, sr)
    return out


print("=" * 104)
print("「尾音浊」优化 v2（全频段目标 + 更长过渡）")
print("=" * 104)

# 真人目标
hx, hsr = load(HUMAN, max_sec=HUMAN_END)
he = voiced_end(hx, hsr)
hn = min(int(TAIL_SEC * hsr), he)
HT = feats(hx[max(0, he - hn):he], hsr)
print(f"  真人目标（末段 {hn/hsr:.2f}s）:")
print("    " + "  ".join(f"{k}={v:.2f}" for k, v in HT.items()
                          if not k.startswith("_")))

raw = []
for w in sorted(SRC.glob("*.wav")):
    x, sr = load(w)
    raw.append((w, x, sr))

# 三个基准
def eval_dir(d: Path, label: str):
    fs = []
    for w, x, sr in raw:
        p = d / w.name
        y = load(p)[0] if p.exists() else x
        e = voiced_end(y, sr)
        n = min(int(TAIL_SEC * sr), e)
        fs.append(feats(y[max(0, e - n):e], sr))
    fs = [f for f in fs if f]
    return {k: float(np.mean([f[k] for f in fs])) for k in fs[0]}


def dist(fm):
    s = 0.0
    for k in fm:
        if k.startswith("_"):
            continue
        s += abs(fm[k] - HT[k]) / SCALE[k]
    s += abs(fm["_cent"] - HT["_cent"]) / 200.0
    s += abs(fm["_h"] - HT["_h"]) / 3.0
    return s


print()
base_orig = eval_dir(SRC, "orig")
print(f"  原版     score={dist(base_orig):.4f}   "
      f"centroid={base_orig['_cent']:.0f}  H1H2={base_orig['_h']:+.2f}  "
      f"2-3.2k={base_orig['2000-3200']:.2f}%  5-9k={base_orig['5000-9000']:.2f}%")
if V1.is_dir():
    base_v1 = eval_dir(V1, "v1")
    print(f"  v1修复   score={dist(base_v1):.4f}   "
          f"centroid={base_v1['_cent']:.0f}  H1H2={base_v1['_h']:+.2f}  "
          f"2-3.2k={base_v1['2000-3200']:.2f}%  5-9k={base_v1['5000-9000']:.2f}%")

grid = list(itertools.product((0.0, -2.0, -4.0), (-2.0, -4.0, -6.0),
                              (2.0, 4.0, 6.0), (0.0, 1.5, 3.0),
                              (0.0, 2.0, 4.0)))
print(f"\n  扫描 {len(grid)} 组 ...")

rows = []
for i, (bm, md, cl, h1, air) in enumerate(grid, 1):
    prm = {"boom": bm, "mud": md, "clarity": cl, "h1": h1, "air": air}
    fs = []
    for w, x, sr in raw:
        y = process(x, sr, prm)
        e = voiced_end(y, sr)
        n = min(int(TAIL_SEC * sr), e)
        fs.append(feats(y[max(0, e - n):e], sr))
    fs = [f for f in fs if f]
    if not fs:
        continue
    fm = {k: float(np.mean([f[k] for f in fs])) for k in fs[0]}
    rows.append({**prm, "cent": fm["_cent"], "h": fm["_h"],
                 "b2k": fm["2000-3200"], "sib": fm["5000-9000"],
                 "low": fm["150-300"], "score": dist(fm)})
    if i % 27 == 0 or i == len(grid):
        print(f"    ...{i}/{len(grid)}")

rows.sort(key=lambda r: r["score"])
print()
print("=" * 104)
print("TOP 15（全频段距离最小）")
print("=" * 104)
print(f"  {'#':>3s} {'boom':>5s}{'mud':>5s}{'clar':>5s}{'h1':>5s}{'air':>5s} "
      f"{'score':>8s} {'质心':>7s} {'H1H2':>7s} {'2-3.2k':>7s} "
      f"{'5-9k':>7s} {'150-300':>8s}")
print("  " + "-" * 96)
print(f"  {'—':>3s} {'**真人**':>5s}{'':>5s}{'':>5s}{'':>5s}{'':>5s} "
      f"{0.0:8.4f} {HT['_cent']:7.0f} {HT['_h']:+7.2f} "
      f"{HT['2000-3200']:7.2f} {HT['5000-9000']:7.2f} {HT['150-300']:8.2f}")
print(f"  {'—':>3s} {'**原版**':>5s}{'':>5s}{'':>5s}{'':>5s}{'':>5s} "
      f"{dist(base_orig):8.4f} {base_orig['_cent']:7.0f} "
      f"{base_orig['_h']:+7.2f} {base_orig['2000-3200']:7.2f} "
      f"{base_orig['5000-9000']:7.2f} {base_orig['150-300']:8.2f}")
for i, r in enumerate(rows[:15], 1):
    print(f"  {i:3d} {r['boom']:5.0f}{r['mud']:5.0f}{r['clarity']:5.0f}"
          f"{r['h1']:5.1f}{r['air']:5.0f} {r['score']:8.4f} {r['cent']:7.0f} "
          f"{r['h']:+7.2f} {r['b2k']:7.2f} {r['sib']:7.2f} {r['low']:8.2f}")

best = rows[0]
print()
print("=" * 104)
print("最优")
print("=" * 104)
print(f"  boom {best['boom']:+.0f}  mud {best['mud']:+.0f}  "
      f"clarity {best['clarity']:+.0f}  h1 {best['h1']:+.1f}  "
      f"air {best['air']:+.0f}")
print()
print(f"  {'指标':<14s} {'真人':>10s} {'原版':>10s} {'v1修复':>10s} "
      f"{'v2最优':>10s}")
print("  " + "-" * 58)
for k, nm, key in (("_cent", "谱质心", "cent"), ("_h", "H1-H2", "h"),
                   ("2000-3200", "2000-3200Hz", "b2k"),
                   ("5000-9000", "5-9kHz齿音", "sib"),
                   ("150-300", "150-300Hz", "low")):
    line = f"  {nm:<14s} {HT[k]:10.2f} {base_orig[k]:10.2f} "
    line += f"{base_v1[k]:10.2f} " if V1.is_dir() else f"{'—':>10s} "
    line += f"{best[key]:10.2f}"
    print(line)

# 导出
bdir = OUT / "best"
bdir.mkdir(parents=True, exist_ok=True)
prm = {k: best[k] for k in ("boom", "mud", "clarity", "h1", "air")}
for w, x, sr in raw:
    sf.write(str(bdir / w.name), process(x, sr, prm), sr)
print()
print(f"  最优 -> {bdir}")

# 三段盲听包
ldir = OUT / "listen"
ldir.mkdir(parents=True, exist_ok=True)
for w, x, sr in raw:
    y2 = process(x, sr, prm)
    y1 = load(V1 / w.name)[0] if (V1 / w.name).exists() else x
    try:
        from math import gcd

        from scipy.signal import resample_poly

        def r48(v):
            if sr == 48000:
                return v
            g = gcd(int(sr), 48000)
            return resample_poly(v, 48000 // g, int(sr) // g)
        xs, y1s, y2s = r48(x), r48(y1), r48(y2)
    except Exception:  # noqa: BLE001
        xs, y1s, y2s = x, y1, y2

    def nz(v):
        r = float(np.sqrt((v ** 2).mean())) + 1e-12
        v = v * (0.06 / r)
        pk = float(np.abs(v).max())
        return v * (0.99 / pk) if pk > 0.99 else v
    gap = np.zeros(int(0.7 * 48000))
    sf.write(str(ldir / f"{w.stem}_3way.wav"),
             np.concatenate([nz(xs), gap, nz(y1s), gap, nz(y2s)]), 48000)
print(f"  三段盲听（原版 -> v1 -> v2）-> {ldir}")

print()
print("=" * 104)
print("命令")
print("=" * 104)
print(f"  python _tools/fix_zhuo.py --in <输入> --out <输出> \\")
print(f"      --boom-db {best['boom']:.0f} --mud-db {best['mud']:.0f} \\")
print(f"      --clarity-db {best['clarity']:.0f}")

rep = ["=" * 80, "「尾音浊」优化 v2", "=" * 80, ""]
rep.append(f"目标(真人末段): 质心 {HT['_cent']:.0f}  H1-H2 {HT['_h']:+.2f}  "
           f"2-3.2k {HT['2000-3200']:.2f}%  5-9k {HT['5000-9000']:.2f}%")
rep.append(f"原版:           质心 {base_orig['_cent']:.0f}  "
           f"H1-H2 {base_orig['_h']:+.2f}  2-3.2k {base_orig['2000-3200']:.2f}%  "
           f"5-9k {base_orig['5000-9000']:.2f}%")
rep.append("")
rep.append(f"{'#':>3s}{'boom':>6s}{'mud':>6s}{'clar':>6s}{'h1':>6s}{'air':>6s}"
           f"{'score':>9s}{'cent':>8s}{'H1H2':>8s}{'2-3.2k':>8s}{'5-9k':>8s}")
for i, r in enumerate(rows[:25], 1):
    rep.append(f"{i:3d}{r['boom']:6.0f}{r['mud']:6.0f}{r['clarity']:6.0f}"
               f"{r['h1']:6.1f}{r['air']:6.0f}{r['score']:9.4f}{r['cent']:8.0f}"
               f"{r['h']:+8.2f}{r['b2k']:8.2f}{r['sib']:8.2f}")
rep.append("")
rep.append(f"最优: boom {best['boom']:.0f} mud {best['mud']:.0f} "
           f"clarity {best['clarity']:.0f} h1 {best['h1']:.1f} "
           f"air {best['air']:.0f}")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
(OUT / "sweep.json").write_text(json.dumps(
    {"human": HT, "orig": base_orig, "rows": rows},
    ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  -> {OUT / 'report.txt'}")
print("=" * 104)
