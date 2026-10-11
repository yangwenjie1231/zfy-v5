# -*- coding: utf-8 -*-
"""验证假设：用户偏好 v5，是因为**带宽**（48kHz），而不是音色相似度。

## 背景

用户听感：**v5 最佳的效果不错**。
我的客观指标：v2Pro 的 SIM 显著更高（0.8888 vs 0.8236）。

指标与听感冲突。最可能的解释：
  v5 输出 48kHz（有效带宽 22.8kHz），v2Pro 只有 32kHz（硬上限 16kHz）。
  16-24kHz 是**真实谐波**（包络相关 +0.99），不是噪声。
  高频细节让听感更清晰自然 —— 而 SIM（说话人嵌入余弦）**测不到这一点**。

## 本测试

把**同一份 v5 输出**做两版：
  · 全带宽版（48kHz，保留 16-24kHz）
  · 限带版（低通到 16kHz，模拟 v2Pro 的带宽上限）
然后 A/B 拼接。

判读：
  · 若用户偏好全带宽版 → 确认带宽是主因，**全量微调值得做**
    （既保住带宽又补上音色）
  · 若听不出差别 → 带宽不是主因，我的 SIM 指标漏掉了别的东西

用法：
    python test_bandwidth_hypothesis.py --epoch 8
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
SRC = V5 / "out" / "r32_vs_r128"          # 已合成的 r32_e8 / v2pro
OUT = V5 / "out" / "bandwidth_test"
SR = 48000
TARGET_RMS = 0.06


def lowpass_fft(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    """FFT 硬低通（用于模拟带宽上限）。

    用频域直接截断，比 IIR 更干净、可复现，且不会引入相位失真。
    """
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / sr)
    X[f > cutoff] = 0.0
    return np.fft.irfft(X, n).astype(np.float32)


def load(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        k = int(round(len(x) * SR / sr))
        x = np.interp(np.linspace(0, len(x) - 1, k), np.arange(len(x)), x)
    x = x.astype(np.float32)
    r = float(np.sqrt(np.mean(x ** 2))) + 1e-9
    return x * (TARGET_RMS / r)


def concat(parts, gap=0.7):
    g = np.zeros(int(gap * SR), dtype=np.float32)
    out = [np.zeros(int(0.5 * SR), dtype=np.float32)]
    for p in parts:
        out += [p, g]
    full = np.concatenate(out)
    pk = float(np.max(np.abs(full)))
    return full * (0.95 / pk) if pk > 0.95 else full


def band_energy(p: Path):
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    nfft, hop = 4096, 1024
    nf = max(1, 1 + (len(x) - nfft) // hop)
    idx = np.arange(nfft)[None, :] + hop * np.arange(nf)[:, None]
    S = (np.abs(np.fft.rfft(x[idx] * np.hanning(nfft), axis=1)) ** 2).mean(axis=0)
    f = np.fft.rfftfreq(nfft, 1 / sr)
    ref = float(np.mean(S[(f >= 200) & (f < 1000)])) + 1e-20
    out = {}
    for n, (a, b) in {"8-16k": (8000, 16000), "16-24k": (16000, 24000)}.items():
        b = min(b, sr / 2)
        m = (f >= a) & (f < b)
        out[n] = (round(10 * np.log10(float(np.mean(S[m])) / ref + 1e-20), 1)
                  if m.sum() >= 8 else None)
    return sr, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="r32_e02")
    ap.add_argument("--src-root", default=None,
                    help="源目录根，默认自动在 out/ 下找")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # 自动定位源目录（不同实验的命名不同：r32_e8 / r32_e08）
    root = Path(args.src_root) if args.src_root else None
    src_dir = None
    cands = [SRC / args.src, V5 / "out" / "r32_sweep" / args.src]
    if root:
        cands.insert(0, root / args.src)
    for c in cands:
        if c.exists():
            src_dir = c
            break
    if src_dir is None:
        print(f"找不到源目录，试过: {[str(c) for c in cands]}")
        return 1
    files = sorted(src_dir.glob("*.wav"))[: args.n]
    print(f"源: {src_dir}  {len(files)} 个文件")

    full_dir = OUT / "full"      # 全带宽
    lim_dir = OUT / "limited"    # 限带到 16kHz
    full_dir.mkdir(exist_ok=True)
    lim_dir.mkdir(exist_ok=True)

    print("生成全带宽 / 限带两版 …")
    for i, p in enumerate(files):
        x = load(p)
        sf.write(str(full_dir / p.name), x, SR, subtype="PCM_16")
        sf.write(str(lim_dir / p.name), lowpass_fft(x, SR, 16000.0), SR,
                 subtype="PCM_16")

    # 校验两版确实不同
    print("\n=== 频带校验（应只有 16-24k 有差异）===")
    for name, d in (("全带宽", full_dir), ("限带16k", lim_dir)):
        ps = sorted(d.glob("*.wav"))[:3]
        for p in ps:
            sr, b = band_energy(p)
            hf = f"{b['16-24k']:6.1f}" if b["16-24k"] is not None else "  --  "
            print(f"  {name:8s} {p.name}  8-16k={b['8-16k']:6.1f}dB  16-24k={hf}dB")

    # A/B 拼接：先全带宽，后限带
    parts = []
    for i in range(len(files)):
        parts.append(load(full_dir / files[i].name))
        parts.append(load(lim_dir / files[i].name))
    full = concat(parts)
    out = OUT / "带宽对比_先全带宽后限带.wav"
    sf.write(str(out), full, SR, subtype="PCM_16")
    print(f"\n生成 {out}  ({len(full)/SR:.1f}s)")

    (OUT / "说明.txt").write_text(
        f"""v5 带宽对比测试（源: {args.src}）
============================================================

同一份 v5 输出，两个版本交替播放：
  A = 全带宽（48kHz，保留 16-24kHz 真实谐波）
  B = 限带（低通到 16kHz，模拟 v2Pro 的带宽上限）

顺序：每句先 A 后 B，句间 0.7s 静音。响度已对齐。

判读：
  · 若明显偏好 A → 确认**带宽**是 v5 听感好的主因
      => 全量微调值得做（既保带宽又补音色）
  · 若听不出差别 → 带宽不是主因，
      说明我的 SIM 指标漏掉了别的东西，需要重新找指标

技术说明：低通用 FFT 频域截断（硬低通），可复现且无相位失真。
两版唯一的差异就是 16kHz 以上内容。
""", encoding="utf-8")
    print(f"说明: {OUT/'说明.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
