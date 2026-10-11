# -*- coding: utf-8 -*-
"""修「尾音浊」—— 与之前的高频衰减**相反**的方向。

## 「浊」的声学定义（实测，同文本 A/B）
    H1-H2（发声紧度）  模型 +8.41 dB  vs 真人 +13.61 dB   **Δ -5.20**
        -> H1-H2 小 = 声门闭合紧 = "挤压/浊"

    末段频段（模型 - 真人）:
        300-500 Hz   +2.83pp   <- 闷
        500-800 Hz   +5.97pp   <- **多一倍**（糊）
        2000-3200 Hz -2.34pp   <- **清晰度频段不足**

## 为什么之前的后处理方向错了
`postprocess_tail.py` 做的是 **lowpass tilt -10dB@6kHz**
-> 把 2~3.2kHz 的清晰度也削掉了 -> **更浊**

## 正确做法
    ① **500~800Hz 搁架衰减**（-3~6dB）—— 去掉"糊"
    ② **2~3.5kHz 提升**（+3~5dB）—— 补清晰度
    ③ **300~500Hz 轻度衰减**（-1~3dB）—— 去掉"闷"
    ④ **H1-H2 增大**：这是发声方式，**滤波改不了**。
       近似做法：**轻微高频增强 + 低频削减**能提升 H1-H2 的读数，
       但真正的 H1-H2 需要模型层面改变（谱倾斜补偿可近似）。

## 只作用于末段
主体不动（避免整句音色变化）。过渡区 80ms 平滑。

## 参数
    --tail-sec    作用区长度（默认 1.0s）
    --mud-db      500~800Hz 衰减（默认 -4）
    --mud-hz      中心频率（默认 650）
    --clarity-db  2~3.5kHz 提升（默认 +4）
    --clarity-hz  中心频率（默认 2700）
    --boom-db     300~500Hz 衰减（默认 -2）

## 用法
    python fix_zhuo.py --in dist/ref_swap_train/orig --out dist/zhuofix
    python sweep_zhuo.py            # 参数扫描
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="indir", required=True)
ap.add_argument("--out", dest="outdir", required=True)
ap.add_argument("--tail-sec", type=float, default=1.0)
ap.add_argument("--mud-db", type=float, default=-4.0)
ap.add_argument("--mud-hz", type=float, default=650.0)
ap.add_argument("--mud-q", type=float, default=1.0)
ap.add_argument("--clarity-db", type=float, default=4.0)
ap.add_argument("--clarity-hz", type=float, default=2700.0)
ap.add_argument("--clarity-q", type=float, default=0.9)
ap.add_argument("--boom-db", type=float, default=-2.0)
ap.add_argument("--boom-hz", type=float, default=400.0)
args = ap.parse_args()

SRC = (V5 / args.indir) if not Path(args.indir).is_absolute() else Path(args.indir)
DST = (V5 / args.outdir) if not Path(args.outdir).is_absolute() else Path(args.outdir)
DST.mkdir(parents=True, exist_ok=True)


def load(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return np.asarray(x, dtype=np.float64), int(sr)


def voiced_end(x: np.ndarray, sr: int, thresh_db: float = -45.0) -> int:
    nf = 256
    m = (len(x) // nf) * nf
    if m < nf * 4:
        return len(x)
    r = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
    ref = float(np.sqrt((x ** 2).mean())) + 1e-12
    v = np.where(r > ref * (10 ** (thresh_db / 20)))[0]
    return int((v[-1] + 1) * nf) if len(v) else len(x)


def peaking(x: np.ndarray, sr: int, f0: float, q: float,
            db: float) -> np.ndarray:
    """参数均衡器（peaking EQ）—— 用 scipy biquad。

    db > 0 提升，db < 0 衰减。零相位（filtfilt）。
    """
    if abs(db) < 0.05:
        return x
    try:
        from scipy.signal import iirpeak, filtfilt
        # iirpeak 给的是带通；用它做"提取该频段"，再按 db 混合
        b, a = iirpeak(f0 / (sr / 2), q)
        band = filtfilt(b, a, x)
    except Exception:  # noqa: BLE001
        return x
    g = 10 ** (db / 20)
    # x = (x - band) + band*g  等价于 提升/衰减该频段
    return (x - band) + band * g


def process(x: np.ndarray, sr: int) -> np.ndarray:
    y = x.copy()
    end = voiced_end(y, sr)
    n_tail = min(int(args.tail_sec * sr), end)
    if n_tail < int(0.05 * sr):
        return y
    a = end - n_tail
    seg = y[a:end].copy()

    # ---- 三段 EQ ----
    if args.boom_db < -0.05:
        seg = peaking(seg, sr, args.boom_hz, 1.0, args.boom_db)
    if args.mud_db < -0.05:
        seg = peaking(seg, sr, args.mud_hz, args.mud_q, args.mud_db)
    if args.clarity_db > 0.05:
        seg = peaking(seg, sr, args.clarity_hz, args.clarity_q,
                      args.clarity_db)

    # ---- 平滑过渡（前 80ms 淡入处理，避免接缝）----
    ramp_n = min(int(0.08 * sr), len(seg))
    if ramp_n > 8:
        r = np.linspace(0, 1, ramp_n)
        seg[:ramp_n] = seg[:ramp_n] * r + y[a:a + ramp_n] * (1 - r)
    y[a:end] = seg

    # ---- 峰值保护 ----
    pk = float(np.abs(y).max())
    if pk > 0.99:
        y = y * (0.99 / pk)
    return y


def h1h2(x: np.ndarray, sr: int) -> float:
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


def band_share(x: np.ndarray, sr: int) -> dict:
    nfft = 2048
    hop = nfft // 4
    acc = np.zeros(nfft // 2 + 1)
    cnt = 0
    for i in range(0, max(1, len(x) - nfft), hop):
        seg = x[i:i + nfft]
        if len(seg) < nfft or np.sqrt((seg ** 2).mean()) < 1e-4:
            continue
        acc += np.abs(np.fft.rfft(seg * np.hanning(nfft))) ** 2
        cnt += 1
    if cnt == 0:
        return {}
    acc /= cnt
    f = np.fft.rfftfreq(nfft, 1 / sr)
    P = acc / (acc.sum() + 1e-20)
    out = {}
    for lo, hi in ((300, 500), (500, 800), (800, 1200),
                   (1200, 2000), (2000, 3200), (3200, 5000)):
        m = (f >= lo) & (f < hi)
        out[f"{lo}-{hi}"] = float(P[m].sum())
    return out


print("=" * 100)
print("修「尾音浊」（低频削 + 清晰度提）")
print("=" * 100)
print(f"  输入 {SRC}")
print(f"  输出 {DST}")
print(f"  参数 作用区 {args.tail_sec}s   "
      f"闷({args.mud_hz:.0f}Hz) {args.mud_db:+.1f}dB   "
      f"清晰({args.clarity_hz:.0f}Hz) {args.clarity_db:+.1f}dB   "
      f"轰鸣({args.boom_hz:.0f}Hz) {args.boom_db:+.1f}dB")

wavs = sorted(SRC.glob("*.wav"))
if not wavs:
    print(f"  !! {SRC} 无 wav")
    sys.exit(1)

before, after = [], []
for w in wavs:
    x, sr = load(w)
    y = process(x, sr)
    sf.write(str(DST / w.name), y, sr)
    e = voiced_end(x, sr)
    n = min(int(1.0 * sr), e)
    tb = x[max(0, e - n):e]
    ta = y[max(0, e - n):e]
    before.append({"h": h1h2(tb, sr), **band_share(tb, sr)})
    after.append({"h": h1h2(ta, sr), **band_share(ta, sr)})

print()
print("=" * 100)
print("效果（末段 1.0s）")
print("=" * 100)


def m(rows, k):
    xs = [r[k] for r in rows if k in r and not np.isnan(r[k])]
    return float(np.mean(xs)) if xs else float("nan")


print(f"  {'指标':<14s} {'处理前':>10s} {'处理后':>10s} {'变化':>10s}  "
      f"{'真人':>10s}")
print("  " + "-" * 62)
for k, name, real in (("h", "H1-H2 dB", 13.61),
                      ("300-500", "300-500Hz", 3.41),
                      ("500-800", "500-800Hz", 4.92),
                      ("2000-3200", "2000-3200Hz", 5.79)):
    b, a = m(before, k), m(after, k)
    unit = "%" if k != "h" else ""
    if unit == "%":
        b, a, real = b * 100, a * 100, real
    print(f"  {name:<14s} {b:10.2f} {a:10.2f} {a-b:+10.2f}  {real:10.2f}")

print()
print("  判读（目标：靠近真人）")
hb, ha = m(before, "h"), m(after, "h")
print(f"    H1-H2   {hb:+.2f} -> {ha:+.2f}   （真人 +13.61）"
      f"  {'✓ 改善' if ha > hb else '✗ 未改善'}")
print("       ⚠️ H1-H2 是**发声方式**（声门闭合紧度），滤波**改不了本质**")
print("          但它随谱倾斜变化 —— 削低频/提高频会小幅提升读数")
for k, name, real in (("500-800", "500-800Hz", 4.92),
                      ("2000-3200", "2000-3200Hz", 5.79)):
    b, a = m(before, k) * 100, m(after, k) * 100
    closer = abs(a - real) < abs(b - real)
    print(f"    {name:<10s} {b:.2f}% -> {a:.2f}%   （真人 {real:.2f}%）"
          f"  {'✓ 更接近' if closer else '✗ 更远'}")

print()
print("=" * 100)
print("下一步")
print("=" * 100)
print(f"""
  ## 参数扫描
      python _tools/sweep_zhuo.py

  ## 盲听
      听 {DST} 里的 wav，重点听**句末最后一个字**
        · 是否比原来"清楚"了？
        · 是否变得"尖"了（过度提升）？

  ## ⚠️ 根本限制
    · 滤波只能改**频谱包络**，改不了**发声方式**（H1-H2 的根源）
    · 真正的修复需要**模型层面**（重训 SoVITS）
    · 但重训已验证 v5 全量微调**没超过 v2Pro**
    · 所以后处理是**当前唯一可行**的改善途径
""")
print("=" * 100)

rep = ["=" * 78, "修「尾音浊」报告", "=" * 78, ""]
rep.append(f"参数: tail {args.tail_sec}s  mud {args.mud_hz:.0f}Hz "
           f"{args.mud_db:+.1f}dB  clarity {args.clarity_hz:.0f}Hz "
           f"{args.clarity_db:+.1f}dB  boom {args.boom_hz:.0f}Hz "
           f"{args.boom_db:+.1f}dB")
rep.append("")
rep.append(f"{'指标':<14s}{'处理前':>12s}{'处理后':>12s}{'真人':>12s}")
for k, name, real in (("h", "H1-H2 dB", 13.61),
                      ("300-500", "300-500Hz%", 3.41),
                      ("500-800", "500-800Hz%", 4.92),
                      ("2000-3200", "2000-3200Hz%", 5.79)):
    b, a = m(before, k), m(after, k)
    if k != "h":
        b, a = b * 100, a * 100
    rep.append(f"{name:<14s}{b:12.2f}{a:12.2f}{real:12.2f}")
(DST / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
(DST / "params.json").write_text(json.dumps(vars(args), ensure_ascii=False,
                                            indent=2), encoding="utf-8")
print(f"  -> {DST / 'report.txt'}")
