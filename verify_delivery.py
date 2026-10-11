# -*- coding: utf-8 -*-
"""交付前最终校验：确认盲听集里 A 段与 B 段的带宽差异真实存在。

关键点：合成时**未做重采样**，所以源文件保留原生采样率：
  · v5   -> 48000 Hz，有 16-24kHz
  · v2Pro-> 32000 Hz，16kHz 以上物理上不存在
拼接成品升到 48kHz 只是容器变化，不会给 v2Pro 凭空造出高频。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FAIR = Path(r"D:\A\zfy-v5\out\fair_compare")
LISTEN = Path(r"D:\A\zfy-v5\out\listen")


def bands(p: Path):
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    nfft, hop = 4096, 1024
    nf = max(1, 1 + (len(x) - nfft) // hop)
    idx = np.arange(nfft)[None, :] + hop * np.arange(nf)[:, None]
    S = (np.abs(np.fft.rfft(x[idx] * np.hanning(nfft), axis=1)) ** 2).mean(axis=0)
    fr = np.fft.rfftfreq(nfft, 1 / sr)
    ref = float(np.mean(S[(fr >= 200) & (fr < 1000)])) + 1e-20
    out = {}
    for n, (a, b) in {"8-16k": (8000, 16000), "16-24k": (16000, 24000)}.items():
        # 必须把上界夹到奈奎斯特：否则 32kHz 文件会在 [16000,24000) 里
        # 命中**唯一一个奈奎斯特点**（fr 正好 16000），报出一个假数值。
        b = min(b, sr / 2)
        if b <= a:
            out[n] = None
            continue
        m = (fr >= a) & (fr < b)
        # 要求覆盖足够宽的频带，单点不算
        out[n] = (round(10 * np.log10(float(np.mean(S[m])) / ref + 1e-20), 1)
                  if m.sum() >= 8 else None)
    return sr, out


print("=" * 74)
print("源文件（合成输出，原生采样率，未重采样）")
print("=" * 74)
for tag in ("v5_e8", "v5_e4", "v5_e7", "v2pro"):
    ps = sorted((FAIR / tag).glob("*.wav"))[:3]
    for p in ps:
        sr, b = bands(p)
        hf = f"{b['16-24k']:6.1f}" if b["16-24k"] is not None else "    --"
        print(f"  {tag:8s} {p.name}  sr={sr}  "
              f"8-16k={b['8-16k']:6.1f}dB  16-24k={hf}dB")

print()
print("=" * 74)
print("交付盲听集")
print("=" * 74)
for f in sorted(LISTEN.glob("*.wav")):
    x, sr = sf.read(str(f))
    print(f"  {f.name:24s} sr={sr}  {len(x)/sr:6.1f}s  "
          f"peak={np.abs(x).max():.3f}  clip={int((np.abs(x) >= 0.999).sum())}")

print()
print("判读：v5 段 16-24k 有内容；v2Pro 段该频带不存在（32kHz 奈奎斯特 16kHz）。")
print("      盲听集输出 48kHz 只是容器，不会给 v2Pro 造出高频。")
