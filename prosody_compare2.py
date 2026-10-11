# -*- coding: utf-8 -*-
"""韵律对比（修正版）：排除带宽混杂因素。

## 上一版的问题

`prosody_compare.py` 测得 v5 的能量动态 85.5 dB vs v2Pro 113.4 dB，
看着差很多。但这**可能是混杂**：

  v5 输出 48kHz，16-24kHz 有真实内容（-55dB）；
  v2Pro 输出 32kHz，16kHz 以上不存在。
  能量在 25ms 窗内全带计算 → v5 的"静音帧"被高频内容垫高 →
  动态范围被压缩，看起来"起伏小"。

**这不是韵律差异，是测量口径差异。**

## 本脚本

把两边的能量都限制在 **0-8kHz**（人耳韵律感知主要频段），
排除高频对静音帧的干扰，再看动态范围。
同时补一个更贴近"语气"的指标：

  · **F0 轮廓的短时变化量**（10ms 尺度的半音变化）——
    真实语音有连续的音高滑动，合成语音常偏"平"或"阶梯状"
  · **F0 有声帧连续性** —— 断裂多说明韵律不连贯
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")
SWEEP = V5 / "out" / "r32_sweep"
CMP = V5 / "out" / "r32_vs_r128"
OUT = V5 / "out" / "prosody2"


def band_energy_env(x, sr, lo=200.0, hi=8000.0, win_ms=25.0, hop_ms=10.0):
    """只在 [lo,hi] 频带内算能量包络（dB）。

    做法：FFT 后把带外 bin 清零再逆变换，然后在时域算 RMS。
    比直接滤波更干净，且不引入相位失真。
    """
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / sr)
    X[(f < lo) | (f > min(hi, sr / 2))] = 0.0
    xb = np.fft.irfft(X, n).astype(np.float32)

    win, hop = int(win_ms / 1000 * sr), int(hop_ms / 1000 * sr)
    nf = 1 + (len(xb) - win) // hop
    if nf < 2:
        return np.array([])
    idx = np.arange(win)[None, :] + hop * np.arange(nf)[:, None]
    e = np.sqrt(np.mean(xb[idx] ** 2, axis=1)) + 1e-12
    return 20 * np.log10(e)


def f0_track(x, sr, fmin=70.0, fmax=400.0, win_ms=40.0, hop_ms=10.0):
    win, hop = int(win_ms / 1000 * sr), int(hop_ms / 1000 * sr)
    nf = 1 + (len(x) - win) // hop
    if nf < 3:
        return np.array([]), np.array([], dtype=bool)
    idx = np.arange(win)[None, :] + hop * np.arange(nf)[:, None]
    fr = x[idx] * np.hanning(win)
    e = np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-12
    thr = max(np.percentile(e, 55), np.max(e) * 0.02)
    lo, hi = max(2, int(sr / fmax)), int(sr / fmin)
    f0 = np.full(nf, np.nan)
    for i in range(nf):
        if e[i] < thr:
            continue
        s = fr[i] - fr[i].mean()
        ac = np.correlate(s, s, mode="full")[len(s) - 1:]
        if ac[0] <= 0:
            continue
        seg = ac[lo:hi]
        if len(seg) < 2:
            continue
        k = int(np.argmax(seg)) + lo
        if seg.max() > 0.30 * ac[0]:
            f0[i] = sr / k
    voiced = ~np.isnan(f0)
    if voiced.sum() > 5:
        v = f0[voiced]
        med = np.median(v)
        v2 = v.copy()
        v2[(v > med * 1.8) | (v < med / 1.8)] = med
        f0[voiced] = v2
    return f0, voiced


def feats(p: Path) -> dict | None:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if len(x) < sr * 0.2:
        return None

    # 带限能量动态（0.2-8kHz，排除高频混杂）
    edb = band_energy_env(x, sr)
    if len(edb) < 5:
        return None
    env_dyn_b = float(np.percentile(edb, 95) - np.percentile(edb, 20))

    f0, voiced = f0_track(x, sr)
    if voiced.sum() < 8:
        return None
    stv = 12 * np.log2(f0[voiced] / 55.0 + 1e-9)
    rng = float(np.percentile(stv, 90) - np.percentile(stv, 10))

    # 短时 F0 变化量（相邻有声帧，10ms 尺度）
    d = np.abs(np.diff(stv))
    micro = float(np.median(d)) if len(d) else 0.0
    macro = float(np.mean(d)) if len(d) else 0.0

    # 有声帧连续性：有声/无声切换次数 / 秒
    switch = int(np.sum(np.diff(voiced.astype(int)) != 0))
    cont = switch / (len(x) / sr)

    return {"env_dyn_band_db": round(env_dyn_b, 2),
            "f0_range_st": round(rng, 2),
            "f0_micro_st": round(micro, 4),
            "f0_macro_st": round(macro, 4),
            "f0_switch_per_s": round(cont, 2),
            "voiced_ratio": round(float(voiced.mean()), 3)}


def paired(a, b):
    dv = np.asarray(a, float) - np.asarray(b, float)
    n = len(dv)
    if n < 3:
        return {"mean": None, "t": None, "sig": False}
    se = dv.std(ddof=1) / np.sqrt(n)
    t = float(dv.mean() / se) if se > 0 else 0.0
    crit = {19: 2.093, 20: 2.086}.get(n, 2.1)
    return {"mean": round(float(dv.mean()), 4), "t": round(t, 2),
            "crit": crit, "sig": bool(abs(t) > crit),
            "win": int((dv > 0).sum()), "lose": int((dv < 0).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5", default="r32_e02")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    groups = {}
    d5 = SWEEP / args.v5
    if d5.exists():
        groups[f"v5_{args.v5}"] = sorted(d5.glob("*.wav"))[: args.n]
    dv2 = CMP / "v2pro"
    if dv2.exists():
        groups["v2pro"] = sorted(dv2.glob("*.wav"))[: args.n]

    rows = {}
    print(f"{'组':18s} {'带限动态':>9s} {'F0范围':>8s} {'微变化':>8s} "
          f"{'宏变化':>8s} {'切换/秒':>8s}")
    print("-" * 70)
    for tag, ps in groups.items():
        fs = [f for f in (feats(p) for p in ps) if f]
        rows[tag] = fs
        if not fs:
            continue
        g = lambda k: float(np.mean([f[k] for f in fs]))  # noqa: E731
        print(f"{tag:18s} {g('env_dyn_band_db'):9.2f} {g('f0_range_st'):8.2f} "
              f"{g('f0_micro_st'):8.4f} {g('f0_macro_st'):8.4f} "
              f"{g('f0_switch_per_s'):8.2f}")

    v5tag = f"v5_{args.v5}"
    tests = {}
    if v5tag in rows and "v2pro" in rows:
        print("\n" + "=" * 74)
        print(f"配对检验：{v5tag} vs v2Pro（同 20 句、同 GPT、能量带限到 0.2-8k）")
        print("=" * 74)
        n = min(len(rows[v5tag]), len(rows["v2pro"]))
        for key, name in [("env_dyn_band_db", "能量动态(带限)"),
                          ("f0_range_st", "F0 动态范围"),
                          ("f0_micro_st", "F0 微变化(中位)"),
                          ("f0_macro_st", "F0 宏变化(均值)"),
                          ("f0_switch_per_s", "有声/无声切换率")]:
            a = [f[key] for f in rows[v5tag][:n]]
            b = [f[key] for f in rows["v2pro"][:n]]
            p = paired(a, b)
            tests[key] = p
            print(f"  {name:18s} 差={p['mean']:+9.4f}  t={p['t']:6.2f}  "
                  f"{'显著' if p['sig'] else '不显著'}  (v5更大 {p['win']}/{n})")

    (OUT / "prosody2.json").write_text(json.dumps(
        {"groups": rows, "tests": tests, "v5": v5tag},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果: {OUT/'prosody2.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
