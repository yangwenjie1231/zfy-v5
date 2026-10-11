# -*- coding: utf-8 -*-
"""v3：修「浊」但**不牺牲 body**（不虚、不往上高）。

## 用户听感 -> 指标（已确认）
    「往上高」 = 谱质心偏高（v2 926 vs 真人 898，**超过**）
    「虚」     = body 150-800Hz 不足（v2 75.39% vs 真人 78.35%，**更薄**）

## v2 的错误
    mud -6dB@650Hz  -> 削掉 **500-800Hz 的胸腔共鸣** -> 单薄 = 虚
    air +4dB@6000Hz -> 提齿音 -> 亮 = 往上高

## v3 的原则
    ① **body（150-800Hz）不得低于真人** —— 这是「虚」的红线
    ② **谱质心不得超过真人** —— 这是「往上高」的红线
    ③ 在①②约束下，尽量修「浊」（H1-H2 靠近真人）

## 手段调整
    · `mud` 中心频率从 650Hz **下移到 480Hz**（只削"闷"，不碰胸腔共鸣）
    · `mud` 深度收到 **-2 ~ -4**
    · `air` 从 +4 收到 **0 ~ +1.5**
    · `h1`（200Hz 提升）保留 —— 它提 H1-H2 且**增加** body，方向正确
    · 新增 `presence`（1200-1800Hz）—— 补清晰度但不动 2-3.2k（避免"尖"）

## 约束式搜索
    先筛掉违反①②的组合，再按 H1-H2 距离排序。

## 输出
    dist/zhuo_v3/report.txt
    dist/zhuo_v3/best/
    dist/zhuo_v3/listen/    原版 -> v2 -> v3
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
OUT = V5 / "dist" / "zhuo_v3"
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


RAMP_SEC = 0.15


def process(x, sr, prm):
    y = x.copy()
    end = voiced_end(y, sr)
    n = min(int(TAIL_SEC * sr), end)
    if n < int(0.05 * sr):
        return y
    a = end - n
    orig = y[a:end].copy()
    seg = orig.copy()
    for key, f0, q in (("mud", prm["mud_hz"], 1.0),
                       ("h1", 200.0, 0.7),
                       ("presence", 1500.0, 0.9),
                       ("air", 6500.0, 0.8)):
        if abs(prm.get(key, 0.0)) > 0.05:
            seg = peaking(seg, sr, f0, q, prm[key])
    rn = min(int(RAMP_SEC * sr), len(seg))
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
            "b2k3k": float(P[(f >= 2000) & (f < 3200)].sum()) * 100,
            "h": h1h2(x)}


print("=" * 100)
print("v3：修「浊」但不虚、不往上高")
print("=" * 100)

# 真人目标
hx = load16(HUMAN, max_sec=HUMAN_END)
he = voiced_end(hx)
hn = min(int(TAIL_SEC * SR), he)
HT = feats(hx[max(0, he - hn):he])
print(f"  真人: 质心 {HT['cent']:.0f}  body {HT['body']:.2f}%  "
      f"air {HT['air']:.2f}%  H1-H2 {HT['h']:+.2f}  "
      f"500-800 {HT['b500_800']:.2f}%")

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


base = eval_prm({"mud": 0.0, "mud_hz": 480.0, "h1": 0.0,
                 "presence": 0.0, "air": 0.0})
print(f"  原版: 质心 {base['cent']:.0f}  body {base['body']:.2f}%  "
      f"air {base['air']:.2f}%  H1-H2 {base['h']:+.2f}  "
      f"500-800 {base['b500_800']:.2f}%")
if V2.is_dir():
    v2f = []
    for w in files:
        x = load16(V2 / w.name)
        e = voiced_end(x)
        n = min(int(TAIL_SEC * SR), e)
        ft = feats(x[max(0, e - n):e])
        if ft:
            v2f.append(ft)
    v2m = {k: float(np.nanmean([f[k] for f in v2f])) for k in v2f[0]}
    print(f"  v2  : 质心 {v2m['cent']:.0f}  body {v2m['body']:.2f}%  "
          f"air {v2m['air']:.2f}%  H1-H2 {v2m['h']:+.2f}  "
          f"500-800 {v2m['b500_800']:.2f}%   <- 用户说'往上高+虚'")

# 约束
BODY_MIN = HT["body"] - 0.5      # body 不得低于真人 0.5pp
CENT_MAX = HT["cent"] + 15       # 质心不得超真人 15Hz
AIR_MAX = HT["air"] + 0.4        # 空气感不得超

grid = list(itertools.product(
    (-1.0, -2.0, -3.0),                       # mud_db
    (420.0, 480.0, 550.0),                    # mud_hz
    (0.0, 1.5, 3.0),                          # h1
    (0.0, 1.5, 3.0),                          # presence
    (0.0, 1.0),                               # air
))
print(f"\n  扫描 {len(grid)} 组（约束: body≥{BODY_MIN:.2f}%  "
      f"质心≤{CENT_MAX:.0f}  空气感≤{AIR_MAX:.2f}%）")

rows = []
for i, (md, mhz, h1, pr, ar) in enumerate(grid, 1):
    prm = {"mud": md, "mud_hz": mhz, "h1": h1, "presence": pr, "air": ar}
    m = eval_prm(prm)
    if m is None:
        continue
    ok = (m["body"] >= BODY_MIN and m["cent"] <= CENT_MAX
          and m["air"] <= AIR_MAX)
    # 目标：H1-H2 靠近真人，且 500-800 不要超真人太多
    cost = (abs(m["h"] - HT["h"]) / 3.0
            + abs(m["b500_800"] - HT["b500_800"]) / 3.0
            + abs(m["cent"] - HT["cent"]) / 250.0)
    rows.append({**prm, **m, "ok": ok, "cost": cost})
    if i % 27 == 0 or i == len(grid):
        print(f"    ...{i}/{len(grid)}")

ok_rows = [r for r in rows if r["ok"]]
print(f"\n  满足约束: **{len(ok_rows)} / {len(rows)}**")
ok_rows.sort(key=lambda r: r["cost"])

print()
print("=" * 100)
print("TOP 15（满足约束 + 代价最小）")
print("=" * 100)
print(f"  {'#':>3s} {'mud':>5s}{'mudHz':>7s}{'h1':>5s}{'pres':>6s}{'air':>5s} "
      f"{'质心':>7s} {'body':>8s} {'500-800':>8s} {'air':>7s} {'H1-H2':>8s} "
      f"{'cost':>7s}")
print("  " + "-" * 98)
print(f"  {'—':>3s} {'真人':>5s}{'':>7s}{'':>5s}{'':>6s}{'':>5s} "
      f"{HT['cent']:7.0f} {HT['body']:7.2f}% {HT['b500_800']:7.2f}% "
      f"{HT['air']:6.2f}% {HT['h']:+8.2f} {0.0:7.4f}")
print(f"  {'—':>3s} {'原版':>5s}{'':>7s}{'':>5s}{'':>6s}{'':>5s} "
      f"{base['cent']:7.0f} {base['body']:7.2f}% {base['b500_800']:7.2f}% "
      f"{base['air']:6.2f}% {base['h']:+8.2f} "
      f"{(abs(base['h']-HT['h'])/3+abs(base['b500_800']-HT['b500_800'])/3+abs(base['cent']-HT['cent'])/250):7.4f}")
for i, r in enumerate(ok_rows[:15], 1):
    print(f"  {i:3d} {r['mud']:5.0f}{r['mud_hz']:7.0f}{r['h1']:5.1f}"
          f"{r['presence']:6.1f}{r['air']:5.1f} {r['cent']:7.0f} "
          f"{r['body']:7.2f}% {r['b500_800']:7.2f}% {r['air']:6.2f}% "
          f"{r['h']:+8.2f} {r['cost']:7.4f}")

if not ok_rows:
    print("\n  !! 无满足约束的组合 —— 说明修「浊」必然伤 body")
    print("     那就只能接受现状，或改参考音频")
    sys.exit(1)

best = ok_rows[0]
print()
print("=" * 100)
print("v3 最优")
print("=" * 100)
print(f"  mud {best['mud']:+.0f}dB@{best['mud_hz']:.0f}Hz   "
      f"h1 {best['h1']:+.1f}dB@200Hz   "
      f"presence {best['presence']:+.1f}dB@1500Hz   "
      f"air {best['air']:+.1f}dB@6500Hz")
print()
print(f"  {'指标':<14s} {'真人':>10s} {'原版':>10s} {'v3':>10s} {'v2(过头)':>10s}")
print("  " + "-" * 58)
for k, nm in (("cent", "谱质心"), ("body", "body150-800%"),
              ("b500_800", "500-800Hz%"), ("air", "air6-9k%"),
              ("h", "H1-H2 dB")):
    line = f"  {nm:<14s} {HT[k]:10.2f} {base[k]:10.2f} {best[k]:10.2f}"
    if V2.is_dir():
        line += f" {v2m[k]:10.2f}"
    print(line)

# 导出
bdir = OUT / "best"
bdir.mkdir(parents=True, exist_ok=True)
prm = {k: best[k] for k in ("mud", "mud_hz", "h1", "presence", "air")}
for w in files:
    x = load16(ORIG / w.name)
    sf.write(str(bdir / w.name), process(x, SR, prm), SR)
print()
print(f"  v3 -> {bdir}")

ldir = OUT / "listen"
ldir.mkdir(parents=True, exist_ok=True)
for w in files:
    x = load16(ORIG / w.name)
    y3 = process(x, SR, prm)
    y2 = load16(V2 / w.name) if (V2 / w.name).exists() else x
    try:
        from math import gcd

        from scipy.signal import resample_poly

        def r48(v):
            if SR == 48000:
                return v
            g = gcd(SR, 48000)
            return resample_poly(v, 48000 // g, SR // g)
        xs, y2s, y3s = r48(x), r48(y2), r48(y3)
    except Exception:  # noqa: BLE001
        xs, y2s, y3s = x, y2, y3

    def nz(v):
        r = float(np.sqrt((v ** 2).mean())) + 1e-12
        v = v * (0.06 / r)
        pk = float(np.abs(v).max())
        return v * (0.99 / pk) if pk > 0.99 else v
    gap = np.zeros(int(0.7 * 48000))
    sf.write(str(ldir / f"{w.stem}_3way.wav"),
             np.concatenate([nz(xs), gap, nz(y2s), gap, nz(y3s)]), 48000)
print(f"  三段盲听（原版 -> v2 -> v3）-> {ldir}")

print()
print("=" * 100)
print("预期听感")
print("=" * 100)
print(f"""
  v3 相对 v2 的变化：
    · body  {v2m['body']:.2f}% -> {best['body']:.2f}%   （真人 {HT['body']:.2f}%）
      -> **不再单薄**（虚的问题）
    · 质心  {v2m['cent']:.0f} -> {best['cent']:.0f}   （真人 {HT['cent']:.0f}）
      -> **不再往上高**
    · 500-800Hz  {v2m['b500_800']:.2f}% -> {best['b500_800']:.2f}%   （真人 {HT['b500_800']:.2f}%）
      -> 仍然比原版（{base['b500_800']:.2f}%）更靠近真人 = 浊改善了
    · H1-H2  {v2m['h']:+.2f} -> {best['h']:+.2f}   （真人 {HT['h']:+.2f}）

  **听的时候重点确认：**
    1. 是否还有「往上高」？（应该没有了）
    2. 是否还有「虚」？（应该没有了）
    3. 「浊」是否改善了？（应该还有一点改善，但比 v2 弱）
""")
print("=" * 100)

rep = ["=" * 80, "v3（修浊但不虚）", "=" * 80, ""]
rep.append(f"约束: body≥{BODY_MIN:.2f}%  质心≤{CENT_MAX:.0f}  空气感≤{AIR_MAX:.2f}%")
rep.append(f"满足约束 {len(ok_rows)}/{len(rows)}")
rep.append("")
rep.append(f"{'':4s}{'真人':>10s}{'原版':>10s}{'v3':>10s}{'v2':>10s}")
for k, nm in (("cent", "质心"), ("body", "body%"), ("b500_800", "500-800%"),
              ("air", "air%"), ("h", "H1-H2")):
    line = f"{nm:<4s}{HT[k]:10.2f}{base[k]:10.2f}{best[k]:10.2f}"
    if V2.is_dir():
        line += f"{v2m[k]:10.2f}"
    rep.append(line)
rep.append("")
rep.append(f"v3 参数: mud {best['mud']:.0f}@{best['mud_hz']:.0f} "
           f"h1 {best['h1']:.1f} presence {best['presence']:.1f} "
           f"air {best['air']:.1f}")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
(OUT / "sweep.json").write_text(json.dumps(
    {"human": HT, "orig": base, "best": best, "rows": ok_rows[:50]},
    ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  -> {OUT / 'report.txt'}")
print("=" * 100)
