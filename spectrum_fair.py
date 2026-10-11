# -*- coding: utf-8 -*-
"""公平的带宽对比：v5(48kHz) vs v2Pro(32kHz)，含 16kHz 以上。

为什么需要这个：
    上一个脚本只算了 12-16k / 8-12k 一个比值，结论是 v5 更差（-4.97dB）。
    但这个指标**对 v5 不公平**：
      · v2Pro 输出 32kHz → 奈奎斯特 16kHz，**16kHz 以上物理上不存在**
      · v5    输出 48kHz → 奈奎斯特 24kHz，**可能有 16-24kHz 内容**
    如果 v5 把能量放到了 16-24k，只看 12-16k 会得出错误结论。

本脚本按完整频带逐段比较，并给出：
  · 各频带相对 200-1000Hz 的 dB
  · 有效带宽（能量掉到 -60dB 的位置）
  · 是否有 >16kHz 的真实内容（对比数字静音底）

同时把 v5 **底模（未微调）** 也测一遍，用来分离
「微调带来的变化」与「底模本身的性质」。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

V5_ROOT = Path(r"D:\A\zfy-v5")
GSV = V5_ROOT / "GPT-SoVITS"
FAIR = V5_ROOT / "out" / "fair_compare"
OUT = V5_ROOT / "out" / "spectrum_fair"

BANDS = [(200, 1000), (1000, 2000), (2000, 4000), (4000, 8000),
         (8000, 12000), (12000, 16000), (16000, 20000), (20000, 24000)]
NAMES = [f"{a//1000}-{b//1000}k" for a, b in BANDS]


def spectrum(path: Path, nfft=4096):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    hop = nfft // 4
    nf = max(1, 1 + (len(x) - nfft) // hop)
    idx = np.arange(nfft)[None, :] + hop * np.arange(nf)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * np.hanning(nfft), axis=1)) ** 2
    P = S.mean(axis=0)
    return np.fft.rfftfreq(nfft, 1 / sr), P, sr


def analyse(path: Path):
    f, P, sr = spectrum(path)
    ref = float(np.mean(P[(f >= 200) & (f < 1000)])) + 1e-20
    out = {}
    for (a, b), n in zip(BANDS, NAMES):
        if b > sr / 2:
            out[n] = None
            continue
        m = (f >= a) & (f < b)
        out[n] = round(10 * np.log10(float(np.mean(P[m])) / ref + 1e-20), 2)
    rel = 10 * np.log10(P / ref + 1e-20)
    hi = f[(rel > -60) & (f > 1000)]
    return {"sr": sr, "bands": out,
            "bandwidth_-60db_hz": round(float(hi.max()), 1) if len(hi) else None,
            "dur": round(len(f) and 0 or 0, 2)}


def group(tag: str, paths):
    if not paths:
        return None
    rows = [analyse(p) for p in paths]
    srs = {r["sr"] for r in rows}
    acc = {}
    for n in NAMES:
        v = [r["bands"][n] for r in rows if r["bands"][n] is not None]
        acc[n] = round(float(np.mean(v)), 2) if v else None
    bw = [r["bandwidth_-60db_hz"] for r in rows if r["bandwidth_-60db_hz"]]
    return {"tag": tag, "n": len(rows), "sr": sorted(srs),
            "bands": acc,
            "bandwidth_mean": round(float(np.mean(bw)), 0) if bw else None}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = []

    for tag in ("v5_e4", "v5_e7", "v5_e8", "v2pro"):
        d = FAIR / tag
        ps = sorted(d.glob("*.wav")) if d.exists() else []
        if ps:
            targets.append((tag, ps))

    # v5 底模（未微调）：同一批文本重新合成
    base_dir = V5_ROOT / "out" / "v5_base"
    if base_dir.exists():
        ps = sorted(base_dir.glob("*.wav"))
        if ps:
            targets.append(("v5dev_base", ps))

    results = []
    for tag, ps in targets:
        g = group(tag, ps)
        if g:
            results.append(g)

    hdr = (f"{'来源':14s} {'sr':>7s} " +
           " ".join(f"{n:>10s}" for n in NAMES) + f" {'-60dB带宽':>11s}")
    print(hdr)
    print("-" * len(hdr))
    for g in results:
        srs = ",".join(str(s) for s in g["sr"])
        cells = " ".join(
            (f"{g['bands'][n]:10.2f}" if g["bands"][n] is not None else f"{'--':>10s}")
            for n in NAMES)
        print(f"{g['tag']:14s} {srs:>7s} {cells} {g['bandwidth_mean']:11.0f}")

    print("\n说明：数值为相对 200-1000Hz 的频带能量(dB)；'--' = 超奈奎斯特，不存在。")

    # 关键判定
    v5 = next((g for g in results if g["tag"].startswith("v5_e")), None)
    v2 = next((g for g in results if g["tag"] == "v2pro"), None)
    verdict = {}
    if v5 and v2:
        b = v5["bands"]
        verdict = {
            "v5_has_16_20k": b.get("16-20k") is not None,
            "v5_16_20k_db": b.get("16-20k"),
            "v5_20_24k_db": b.get("20-24k"),
            "v2pro_has_16_20k": v2["bands"].get("16-20k") is not None,
            "v5_12_16k": b.get("12-16k"),
            "v2pro_12_16k": v2["bands"].get("12-16k"),
            "v5_bandwidth": v5["bandwidth_mean"],
            "v2pro_bandwidth": v2["bandwidth_mean"],
        }
        print("\n" + "=" * 78)
        print(f"v5 12-16k = {b.get('12-16k')} dB   |  v2Pro 12-16k = "
              f"{v2['bands'].get('12-16k')} dB")
        print(f"v5 16-20k = {b.get('16-20k')} dB   |  v2Pro 16-20k = 不存在"
              if b.get("16-20k") is not None else
              f"v5 16-20k = 不存在（与 v2Pro 同）")
        print(f"v5 有效带宽 {v5['bandwidth_mean']:.0f} Hz  |  "
              f"v2Pro {v2['bandwidth_mean']:.0f} Hz")

    (OUT / "spectrum.json").write_text(
        json.dumps({"groups": results, "verdict": verdict},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果: {OUT/'spectrum.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
