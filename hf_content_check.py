# -*- coding: utf-8 -*-
"""核查 v5 的 16-24kHz 内容是**真实语音**还是声码器噪声/混叠。

为什么必须查：
    上一轮我用「12-16k / 8-12k 比值」比较，得出 v5 更差 —— **那个指标是错的**：
    v5 的 8-12k 本身比 v2Pro 亮 4.25dB，比值法等于惩罚了"低频段更亮"。
    绝对比较下 v5 的 12-16k 与 v2Pro 相当，而 v5 还有 16-24k 内容。

    但「有能量」≠「是信号」。声码器上采样产生的镜像/混叠也会在
    16-24k 留能量。必须区分，否则会把伪影当优点。

判据（三个独立证据）：
  1. **谱平坦度**：语音的谐波结构 → 平坦度远小于 1；白噪 → 接近 1
  2. **与低频包络的相关性**：真实高频谐波随语音起落；
     混叠/镜像噪声通常与语音包络弱相关
  3. **与源素材对比**：自然录音在 16-24k 的能量水平作参照

参照组：终末地源素材（真 48kHz 录音）、v5 底模、v5 微调、v2Pro。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")
FAIR = V5 / "out" / "fair_compare"
OUT = V5 / "out" / "hf_check"
SRC = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频\终末地解包")


def load(p: Path):
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def band_stats(x, sr, lo, hi, nfft=2048):
    """返回该频带的（相对能量dB, 谱平坦度, 帧数）。

    `hi` 会被夹到奈奎斯特频率：否则 v2Pro(32kHz) 会"报出" 16-20k 的数值，
    而那个区间里其实只有 16kHz 那一个 bin（奈奎斯特点），是假信号。
    """
    hi = min(hi, sr / 2)
    if hi <= lo:
        return None
    hop = nfft // 4
    nf = 1 + (len(x) - nfft) // hop
    if nf < 2:
        return None
    idx = np.arange(nfft)[None, :] + hop * np.arange(nf)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * np.hanning(nfft), axis=1)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / sr)
    m = (f >= lo) & (f < hi)
    # 至少要覆盖半个倍频程，否则样本太少不足以判定
    if m.sum() < 8:
        return None
    B = S[:, m] + 1e-20
    # 谱平坦度 = 几何均值 / 算术均值
    flat = float(np.mean(np.exp(np.mean(np.log(B), axis=1)) /
                         (np.mean(B, axis=1) + 1e-20)))
    ref = float(np.mean(S[:, (f >= 200) & (f < 1000)])) + 1e-20
    lvl = 10 * np.log10(float(np.mean(B)) / ref + 1e-20)
    return {"level_db": round(lvl, 2), "flatness": round(flat, 4), "frames": nf}


def env_corr(x, sr, lo, hi, nfft=1024, hop=256):
    """高频带包络 vs 低频带（200-4000）包络的相关系数。

    真实语音：高低频同起同落 -> 高相关
    静态噪声/混叠：与语音包络关系弱 -> 低相关
    """
    hi = min(hi, sr / 2)
    if hi <= lo:
        return None
    nf = 1 + (len(x) - nfft) // hop
    if nf < 5:
        return None
    idx = np.arange(nfft)[None, :] + hop * np.arange(nf)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * np.hanning(nfft), axis=1)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / sr)
    low = np.mean(S[:, (f >= 200) & (f < 4000)], axis=1) + 1e-20
    m = (f >= lo) & (f < hi)
    if m.sum() < 8:
        return None
    high = np.mean(S[:, m], axis=1) + 1e-20
    a = np.log(low) - np.log(low).mean()
    b = np.log(high) - np.log(high).mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return round(float(np.dot(a, b) / d), 4) if d > 0 else None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    # 自然源素材参照
    srcs = sorted(SRC.rglob("*.wav"))[:20]
    nat = {"8-12k": [], "12-16k": [], "16-20k": [], "20-24k": []}
    nat_ec = {"16-20k": [], "20-24k": []}
    nat_flat = {"16-20k": [], "20-24k": []}
    for p in srcs:
        x, sr = load(p)
        if sr != 48000:
            continue
        for n, (a, b) in {"8-12k": (8000, 12000), "12-16k": (12000, 16000),
                          "16-20k": (16000, 20000), "20-24k": (20000, 24000)}.items():
            st = band_stats(x, sr, a, b)
            if st:
                nat[n].append(st["level_db"])
                if n in nat_flat:
                    nat_flat[n].append(st["flatness"])
        for n, (a, b) in {"16-20k": (16000, 20000), "20-24k": (20000, 24000)}.items():
            c = env_corr(x, sr, a, b)
            if c is not None:
                nat_ec[n].append(c)

    def summarize(tag, dirs):
        for tag2, d in dirs:
            if not d.exists():
                continue
            # 注意：源素材是**分层目录**（干员语音/剧情语音/战斗语音），
            # 用 glob 会一个都找不到 → 必须 rglob。
            ps = sorted(d.rglob("*.wav"))[:40] if tag2.startswith("源素材") \
                else sorted(d.glob("*.wav"))
            if not ps:
                continue
            acc = {"8-12k": [], "12-16k": [], "16-20k": [], "20-24k": []}
            ec = {"16-20k": [], "20-24k": []}
            fl = {"16-20k": [], "20-24k": []}
            srs = set()
            for p in ps:
                x, sr = load(p)
                srs.add(sr)
                for n, (a, b) in {"8-12k": (8000, 12000), "12-16k": (12000, 16000),
                                  "16-20k": (16000, 20000),
                                  "20-24k": (20000, 24000)}.items():
                    st = band_stats(x, sr, a, b)
                    if st:
                        acc[n].append(st["level_db"])
                        if n in fl:
                            fl[n].append(st["flatness"])
                for n, (a, b) in {"16-20k": (16000, 20000),
                                  "20-24k": (20000, 24000)}.items():
                    c = env_corr(x, sr, a, b)
                    if c is not None:
                        ec[n].append(c)
            rows.append({
                "tag": tag2, "n": len(ps), "sr": sorted(srs),
                "levels": {k: (round(float(np.mean(v)), 2) if v else None)
                           for k, v in acc.items()},
                "flatness_16_20k": (round(float(np.mean(fl["16-20k"])), 4)
                                    if fl["16-20k"] else None),
                "flatness_20_24k": (round(float(np.mean(fl["20-24k"])), 4)
                                    if fl["20-24k"] else None),
                "envcorr_16_20k": (round(float(np.mean(ec["16-20k"])), 4)
                                   if ec["16-20k"] else None),
                "envcorr_20_24k": (round(float(np.mean(ec["20-24k"])), 4)
                                   if ec["20-24k"] else None),
            })

    summarize("natural", [("源素材48k(真人)", SRC)])
    summarize("v5", [("v5_e8", FAIR / "v5_e8"),
                     ("v5_e4", FAIR / "v5_e4"),
                     ("v5_e7", FAIR / "v5_e7")])
    summarize("base", [("v5dev底模", V5 / "out" / "v5_base")])
    summarize("v2pro", [("v2Pro", FAIR / "v2pro")])

    def cell(v, w=8, fmt="{:8.2f}"):
        return fmt.format(v) if v is not None else "--".rjust(w)

    hdr = (f"{'来源':18s} {'n':>3s} {'sr':>6s} {'8-12k':>8s} {'12-16k':>8s} "
           f"{'16-20k':>8s} {'20-24k':>8s} {'平坦16-20k':>11s} {'包络相关16-20k':>15s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        L = r["levels"]
        srs = ",".join(str(s) for s in r["sr"])
        print(f"{r['tag']:18s} {r['n']:3d} {srs:>6s} "
              f"{cell(L['8-12k'])} {cell(L['12-16k'])} "
              f"{cell(L['16-20k'])} {cell(L['20-24k'])} "
              f"{cell(r['flatness_16_20k'], 11, '{:11.4f}')} "
              f"{cell(r['envcorr_16_20k'], 15, '{:+15.4f}')}")

    print("\n判读：")
    print("  · 平坦度 « 1 → 有谐波结构（是信号）；≈1 → 像白噪")
    print("  · 包络相关 > 0.5 → 高频随语音起落（真实谐波）；≈0 → 静态噪声")
    print("  · '--' = 该频带超奈奎斯特（v2Pro 32kHz 无 16k 以上）")

    (OUT / "hf_check.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果: {OUT/'hf_check.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
