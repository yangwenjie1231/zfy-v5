# -*- coding: utf-8 -*-
"""v4：**锁住已合格的指标**，只修偏离大的三项。

## 前三次失败的共同根源
没有先分清"哪些指标已经合格"，就笼统优化 -> 优化器把合格的也推歪了。

    指标            真人      原版     偏离        处理原则
    body150-800    78.35    81.59    +4%    **接近 -> 锁死（±1.5pp）**
    谱质心          898      650     -28%    需提升（但不过冲）
    500-800Hz       4.64    10.05   +117%   需削减（「浊」的主因）
    air6-9k         2.33     0.92    -61%    需提升（但不过冲）
    H1-H2          13.68    10.92    -20%    需提升（发声紧度）

## v4 的三条硬约束（双向！）
    ① `|body - 78.35| <= 1.5`   <- v2/v3 都违反了这条
    ② `|质心 - 898| <= 60`       <- 不许过头（v2 是 +29，勉强；v3 是 -347，惨）
    ③ `|air - 2.33| <= 0.5`      <- v2 的 2.08 合格，v3 的 0.65 不合格

## 目标
    在满足①②③的前提下，最小化：
        |500-800 - 4.64|  （浊）
        |H1-H2 - 13.68|   （发声紧度）

## 手段（只动必要的）
    · `mud`（削 500-800 的"糊"）—— 中心频率扫 480~700
    · `h1`（提 200Hz，增 H1-H2 且**增加** body —— 与①兼容）
    · `air`（提 6-9k）—— 小幅度
    · **不加 presence**（v3 证明会推高 body）

## 输出
    dist/zhuo_v4/report.txt
    dist/zhuo_v4/best/
    dist/zhuo_v4/listen/   原版 -> v2 -> v4
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
ORIG = V5 / "dist" / "ref_swap_train" / "orig"
V2 = V5 / "dist" / "zhuo_sweep2" / "best"
HUMAN = V5 / "GPT-SoVITS" / "ref" / "default.wav"
HUMAN_END = 6.95
OUT = V5 / "dist" / "zhuo_v4"
OUT.mkdir(parents=True, exist_ok=True)
TAIL_SEC = 1.0
SR = 16000


def load16(p: Path, max_sec=None):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if max_sec:
        x = x[:int(max_sec * sr)]
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return np.asarray(x, dtype=np.float64)


def voiced_end(x, sr=SR, t=-45.0):
    n = int(0.01 * sr)
    m = (len(x) // n) * n
    if m < n * 4:
        return len(x)
    r = np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))
    ref = r.max() + 1e-12
    v = np.where(r > ref * (10 ** (t / 20)))[0]
    return int((v[-1] + 1) * n) if len(v) else len(x)


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
    orig = y[a:end].copy()
    seg = orig.copy()
    if abs(prm.get("h1", 0)) > 0.05:
        seg = peaking(seg, sr, 200.0, 0.7, prm["h1"])
    if abs(prm.get("mud", 0)) > 0.05:
        seg = peaking(seg, sr, prm["mud_hz"], 1.0, prm["mud"])
    if abs(prm.get("air", 0)) > 0.05:
        seg = peaking(seg, sr, 6500.0, 0.8, prm["air"])
    rn = min(int(0.15 * sr), len(seg))
    if rn > 8:
        t = np.linspace(0, 1, rn)
        seg[:rn] = seg[:rn] * np.sin(t * np.pi / 2) + \
            orig[:rn] * np.cos(t * np.pi / 2)
    y[a:end] = seg
    pk = float(np.abs(y).max())
    if pk > 0.99:
        y = y * (0.99 / pk)
    return y


def h1h2(x, sr=SR):
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


def feats(x, sr=SR):
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
        return None
    f = np.fft.rfftfreq(nfft, 1 / sr)
    P = acc / (acc.sum() + 1e-20)
    return {"cent": float((f * P).sum()),
            "body": float(P[(f >= 150) & (f < 800)].sum()) * 100,
            "air": float(P[(f >= 6000) & (f < min(9000, SR / 2))].sum()) * 100,
            "b500_800": float(P[(f >= 500) & (f < 800)].sum()) * 100,
            "h": h1h2(x)}


print("=" * 104)
print("v4：锁住已合格的 body，只修偏离大的三项")
print("=" * 104)

hx = load16(HUMAN, max_sec=HUMAN_END)
he = voiced_end(hx)
hn = min(int(TAIL_SEC * SR), he)
HT = feats(hx[max(0, he - hn):he])
print(f"  真人: 质心 {HT['cent']:.0f}  body {HT['body']:.2f}%  "
      f"500-800 {HT['b500_800']:.2f}%  air {HT['air']:.2f}%  "
      f"H1-H2 {HT['h']:+.2f}")

files = sorted(ORIG.glob("*.wav"))


def eval_prm(prm):
    fs = []
    for w in files:
        x = load16(ORIG / w.name)
        y = process(x, SR, prm)
        e = voiced_end(y)
        n = min(int(TAIL_SEC * SR), e)
        ft = feats(y[max(0, e - n):e])
        if ft:
            fs.append(ft)
    if not fs:
        return None
    return {k: float(np.nanmean([f[k] for f in fs])) for k in fs[0]}


base = eval_prm({"h1": 0.0, "mud": 0.0, "mud_hz": 550.0, "air": 0.0})
print(f"  原版: 质心 {base['cent']:.0f}  body {base['body']:.2f}%  "
      f"500-800 {base['b500_800']:.2f}%  air {base['air']:.2f}%  "
      f"H1-H2 {base['h']:+.2f}")

# ---------- 硬约束（双向）----------
C_BODY = 1.5          # |body - 真人| <= 1.5pp
C_CENT = 60.0         # |质心 - 真人| <= 60
C_AIR = 0.5           # |air - 真人| <= 0.5pp
print()
print(f"  硬约束: |body-{HT['body']:.2f}|<={C_BODY}   "
      f"|质心-{HT['cent']:.0f}|<={C_CENT:.0f}   "
      f"|air-{HT['air']:.2f}|<={C_AIR}")
print(f"  原版是否满足: body {abs(base['body']-HT['body']):.2f}"
      f"{'✓' if abs(base['body']-HT['body'])<=C_BODY else '✗'}   "
      f"质心 {abs(base['cent']-HT['cent']):.0f}"
      f"{'✓' if abs(base['cent']-HT['cent'])<=C_CENT else '✗'}   "
      f"air {abs(base['air']-HT['air']):.2f}"
      f"{'✓' if abs(base['air']-HT['air'])<=C_AIR else '✗'}")

grid = list(itertools.product(
    (0.0, 1.0, 2.0, 3.0),              # h1（提 200Hz，增 H1-H2 且增 body）
    (-1.0, -2.0, -3.0, -4.0),          # mud（削 500-800）
    (480.0, 550.0, 620.0),             # mud_hz
    (0.0, 0.8, 1.6),                   # air（提 6-9k，小幅）
))
print(f"\n  扫描 {len(grid)} 组 ...")

rows = []
for i, (h1, md, mhz, ar) in enumerate(grid, 1):
    prm = {"h1": h1, "mud": md, "mud_hz": mhz, "air": ar}
    m = eval_prm(prm)
    if m is None:
        continue
    ok = (abs(m["body"] - HT["body"]) <= C_BODY
          and abs(m["cent"] - HT["cent"]) <= C_CENT
          and abs(m["air"] - HT["air"]) <= C_AIR)
    cost = (abs(m["b500_800"] - HT["b500_800"]) / 3.0
            + abs(m["h"] - HT["h"]) / 3.0)
    rows.append({**prm, **m, "ok": ok, "cost": cost})
    if i % 32 == 0 or i == len(grid):
        print(f"    ...{i}/{len(grid)}")

ok_rows = sorted([r for r in rows if r["ok"]], key=lambda r: r["cost"])
print(f"\n  满足硬约束: **{len(ok_rows)} / {len(rows)}**")

if not ok_rows:
    print("  !! 无满足约束的组合")
    print("     -> 说明「修浊」与「保持 body/质心/空气感」**存在根本冲突**")
    print("     -> 后处理这条路到极限了")
    sys.exit(1)

print()
print("=" * 104)
print("TOP 15（满足硬约束 + 代价最小）")
print("=" * 104)
print(f"  {'#':>3s}{'h1':>5s}{'mud':>5s}{'mudHz':>7s}{'air':>5s} "
      f"{'质心':>7s} {'body':>8s} {'500-800':>8s} {'air':>7s} {'H1-H2':>8s} "
      f"{'cost':>7s}")
print("  " + "-" * 96)
print(f"  {'—':>3s}{'真人':>5s}{'':>5s}{'':>7s}{'':>5s} {HT['cent']:7.0f} "
      f"{HT['body']:7.2f}% {HT['b500_800']:7.2f}% {HT['air']:6.2f}% "
      f"{HT['h']:+8.2f} {0.0:7.4f}")
print(f"  {'—':>3s}{'原版':>5s}{'':>5s}{'':>7s}{'':>5s} {base['cent']:7.0f} "
      f"{base['body']:7.2f}% {base['b500_800']:7.2f}% {base['air']:6.2f}% "
      f"{base['h']:+8.2f} "
      f"{(abs(base['b500_800']-HT['b500_800'])/3+abs(base['h']-HT['h'])/3):7.4f}")
for i, r in enumerate(ok_rows[:15], 1):
    print(f"  {i:3d}{r['h1']:5.1f}{r['mud']:5.0f}{r['mud_hz']:7.0f}"
          f"{r['air']:5.1f} {r['cent']:7.0f} {r['body']:7.2f}% "
          f"{r['b500_800']:7.2f}% {r['air']:6.2f}% {r['h']:+8.2f} "
          f"{r['cost']:7.4f}")

best = ok_rows[0]
print()
print("=" * 104)
print("v4 最优")
print("=" * 104)
print(f"  h1 {best['h1']:+.1f}dB@200Hz   mud {best['mud']:+.0f}dB"
      f"@{best['mud_hz']:.0f}Hz   air {best['air']:+.1f}dB@6500Hz")
print()
print(f"  {'指标':<14s} {'真人':>10s} {'原版':>10s} {'v4':>10s} "
      f"{'v2':>10s} {'v3':>10s}  约束")
print("  " + "-" * 74)
for k, nm in (("cent", "谱质心"), ("body", "body150-800%"),
              ("b500_800", "500-800Hz%"), ("air", "air6-9k%"),
              ("h", "H1-H2 dB")):
    line = f"  {nm:<14s} {HT[k]:10.2f} {base[k]:10.2f} {best[k]:10.2f}"
    if V2.is_dir():
        v2v = 0.0
        line += f" {v2v:10.2f}" if k == "x" else ""
    line += f" {'':>10s}"
    print(line.rstrip())
print()
print("  （v2 数值见 dist/zhuo_audit/report.txt）")

# 导出
bdir = OUT / "best"
bdir.mkdir(parents=True, exist_ok=True)
prm = {k: best[k] for k in ("h1", "mud", "mud_hz", "air")}
for w in files:
    x = load16(ORIG / w.name)
    sf.write(str(bdir / w.name), process(x, SR, prm), SR)
print(f"\n  v4 -> {bdir}")

ldir = OUT / "listen"
ldir.mkdir(parents=True, exist_ok=True)
for w in files:
    x = load16(ORIG / w.name)
    y4 = process(x, SR, prm)
    y2 = load16(V2 / w.name) if (V2 / w.name).exists() else x
    try:
        from math import gcd

        from scipy.signal import resample_poly

        def r48(v):
            g = gcd(SR, 48000)
            return resample_poly(v, 48000 // g, SR // g) if SR != 48000 else v
        xs, y2s, y4s = r48(x), r48(y2), r48(y4)
    except Exception:  # noqa: BLE001
        xs, y2s, y4s = x, y2, y4

    def nz(v):
        r = float(np.sqrt((v ** 2).mean())) + 1e-12
        v = v * (0.06 / r)
        pk = float(np.abs(v).max())
        return v * (0.99 / pk) if pk > 0.99 else v
    gap = np.zeros(int(0.7 * 48000))
    sf.write(str(ldir / f"{w.stem}_3way.wav"),
             np.concatenate([nz(xs), gap, nz(y2s), gap, nz(y4s)]), 48000)
print(f"  三段盲听（原版 -> v2 -> v4）-> {ldir}")

rep = ["=" * 80, "v4（锁住 body，只修偏离大的）", "=" * 80, ""]
rep.append(f"硬约束: |body-{HT['body']:.2f}|<={C_BODY}  "
           f"|质心-{HT['cent']:.0f}|<={C_CENT:.0f}  "
           f"|air-{HT['air']:.2f}|<={C_AIR}")
rep.append(f"满足约束 {len(ok_rows)}/{len(rows)}")
rep.append("")
rep.append(f"{'指标':<16s}{'真人':>10s}{'原版':>10s}{'v4':>10s}")
for k, nm in (("cent", "质心"), ("body", "body%"),
              ("b500_800", "500-800%"), ("air", "air%"), ("h", "H1-H2")):
    rep.append(f"{nm:<16s}{HT[k]:10.2f}{base[k]:10.2f}{best[k]:10.2f}")
rep.append("")
rep.append(f"v4 参数: h1 {best['h1']:.1f} mud {best['mud']:.0f}"
           f"@{best['mud_hz']:.0f} air {best['air']:.1f}")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
(OUT / "sweep.json").write_text(json.dumps(
    {"human": HT, "orig": base, "best": best, "n_ok": len(ok_rows),
     "n_total": len(rows), "rows": ok_rows[:60]},
    ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  -> {OUT / 'report.txt'}")
print("=" * 104)
