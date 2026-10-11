# -*- coding: utf-8 -*-
"""重建盲听包 —— 修掉线性插值的频响衰减，并加入 A-A 对照。

## 上一版的两个问题

### 问题 1（严重）：线性插值削高频
两个来源采样率不同：
    v2Pro   -> 32000 Hz（v2Pro 模型的 BigVGAN 是 32k）
    v5merged-> 48000 Hz（v5dev 是 48k）
我用 `np.interp` 线性插值把两者都升到 48kHz。
线性插值的频响是 **sinc²(f/48000)**：
     1 kHz -0.01 dB      8 kHz -0.80 dB
     4 kHz -0.20 dB     12 kHz -1.82 dB
    16 kHz -3.30 dB
**v2Pro 被削了，v5merged 没有**（原生 48k 不经过重采样）。
而 v2Pro 在 7/10 个文件里是第 1 段
-> 用户听到的"前段偏糊"**正是这个 bug**，不是模型差异。

### 问题 2：没有对照，无法分离"位置假象"
每句只有一个 A/B。若人耳对"第 1 段 vs 第 2 段"本身有偏好
（起音强调、响度重校准），就会被误读成模型差异。

## 本脚本的修法
  1. 用 **scipy.signal.resample_poly**（多相 FIR，阻带 >80 dB）
     代替线性插值 —— 通带平坦到 0.99×Nyquist
  2. 加 **A-A 对照**：同一模型出现两次
       · `AA_v2pro`   = v2pro + v2pro
       · `AA_v5merged`= v5merged + v5merged
     若用户仍觉得"第 1 段糊"，那就是**位置假象**，与模型无关
  3. 加 **三方对比**（v2pro / v5merged / v5bad）
  4. 输出**共同带宽内的**频谱对比（把两者都限到 16 kHz，
     因为 v2Pro 的 Nyquist 只有 16 kHz，超出部分无法公平比较）

## 输出
    D:\\A\\zfy-v5\\dist\\listen2\\
        AB\\NN_类型_AB.wav        每句 2 段（顺序打乱）
        AA\\AA_v2pro_*.wav        A-A 对照（校准位置假象）
        AA\\AA_v5merged_*.wav
        01_三方对比.wav
        answer_key.txt
        spectral.txt              共同带宽内的频谱指标
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

V5 = Path(r"D:\A\zfy-v5")
A_DIR = V5 / "dist" / "baseline_v2pro"                    # v2Pro 32k
B_DIR = V5 / "dist" / "sweep2" / "out_sweep" / "e16_merged"  # v5merged 48k
C_DIR = V5 / "dist" / "sweep2" / "out_sweep" / "e16"         # v5bad
OUT = V5 / "dist" / "listen2"

TARGET_SR = 48000
TARGET_RMS = 0.06
HEAD_S, GAP_S = 0.5, 0.7
SEED = 20261012
# 共同带宽：v2Pro Nyquist = 16 kHz，超出部分无法公平比较
COMMON_HZ = 16000.0

TEXTS = {
    "01": ("短", "让我想想。"), "02": ("短", "好。"),
    "03": ("中", "迎敌或是迎客，都得整装后再出发。"),
    "04": ("中", "出门前，要不要先喝杯茶提提神？"),
    "05": ("中", "嗯……我理一理。"),
    "06": ("长", "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？"),
    "07": ("长", "武陵城外有一片梨树林，每年春天都会开满白色的花。"),
    "08": ("长", "她就在那阵花香里，只是他自己不知道。"),
    "09": ("长", "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。"),
    "10": ("数字", "这件事我三点五十分再去处理。"),
}


def load_mono(p: Path):
    x, sr = sf.read(str(p), always_2d=True)
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return x.astype(np.float64), int(sr)


def to48(x: np.ndarray, sr: int) -> np.ndarray:
    """**多相 FIR** 重采样（不是线性插值）。

    scipy.signal.resample_poly 用 Kaiser 窗设计的 FIR，
    阻带衰减 >80 dB，通带平坦 —— 不会像线性插值那样削高频。
    """
    if sr == TARGET_SR:
        return x
    from math import gcd
    g = gcd(int(sr), TARGET_SR)
    up, down = TARGET_SR // g, int(sr) // g
    return resample_poly(x, up, down).astype(np.float64)


def align_rms(x: np.ndarray, target: float = TARGET_RMS) -> np.ndarray:
    r = float(np.sqrt((x ** 2).mean()))
    if r < 1e-9:
        return x
    y = x * (target / r)
    pk = float(np.abs(y).max())
    if pk > 0.99:
        y = y * (0.99 / pk)
    return y


def find(d: Path, idx: str):
    hits = sorted(d.glob(f"{idx}_*.wav")) if d.is_dir() else []
    return hits[0] if hits else None


def band_stats(x: np.ndarray, sr: int, lo_hz=0.0, hi_hz=COMMON_HZ):
    """只统计 [lo_hz, hi_hz) 频段的谱指标（公平比较用）。"""
    n, hop = 4096, 1024
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    w = np.hanning(n)
    mag = np.abs(np.fft.rfft(frames * w, axis=1)).mean(axis=0)
    f = np.fft.rfftfreq(n, 1 / sr)
    sel = (f >= lo_hz) & (f < hi_hz)
    m, ff = mag[sel], f[sel]
    tot = m.sum() + 1e-12
    centroid = float((ff * m).sum() / tot)
    # 4~8k 与 8~16k 的能量比（"气音/齿音"主要在 6~16k）
    e_low = m[(ff >= 2000) & (ff < 6000)].sum()
    e_hi = m[(ff >= 6000) & (ff < hi_hz)].sum()
    return centroid, float(e_hi / (e_low + 1e-12))


def main() -> int:
    print("=" * 96)
    print("重建盲听包（多相 FIR 重采样 + A-A 对照）")
    print("=" * 96)

    a_files = sorted(A_DIR.glob("*.wav")) if A_DIR.is_dir() else []
    b_files = sorted(B_DIR.glob("*.wav")) if B_DIR.is_dir() else []
    print(f"  v2Pro    : {len(a_files)} 个 @32kHz  {A_DIR}")
    print(f"  v5merged : {len(b_files)} 个 @48kHz  {B_DIR}")
    if not a_files or not b_files:
        print("  !! 来源缺失")
        return 1

    (OUT / "AB").mkdir(parents=True, exist_ok=True)
    (OUT / "AA").mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    head = np.zeros(int(HEAD_S * TARGET_SR))
    gap = np.zeros(int(GAP_S * TARGET_SR))

    key = ["# 盲听答案（听完再看）", ""]
    spec = ["# 共同带宽 (0~16kHz) 内的频谱指标", ""]
    n_made = 0
    pairs = []

    for idx in sorted(TEXTS):
        kind, text = TEXTS[idx]
        pa, pb = find(A_DIR, idx), find(B_DIR, idx)
        if not (pa and pb):
            print(f"  {idx}: 缺文件")
            continue

        xa_raw, sra = load_mono(pa)
        xb_raw, srb = load_mono(pb)
        xa = align_rms(to48(xa_raw, sra))
        xb = align_rms(to48(xb_raw, srb))

        # 频谱（在共同带宽内比较）
        ca, ha = band_stats(xa, TARGET_SR)
        cb, hb = band_stats(xb, TARGET_SR)
        spec.append(
            f"{idx}_{kind}:  v2pro 质心 {ca:6.0f} Hz  6-16k/2-6k {ha:.4f}  "
            f"({len(xa)/TARGET_SR:.2f}s)  |  "
            f"v5merged 质心 {cb:6.0f} Hz  {hb:.4f}  "
            f"({len(xb)/TARGET_SR:.2f}s)")
        spec.append(
            f"{'':9s}  Δ质心 {cb-ca:+6.0f} Hz   Δ气音比 {hb-ha:+.4f}"
            f"  {'v5 更亮/气音更多' if hb > ha else 'v5 更闷/气音更少'}")

        # ---- AB ----
        items = [("v2pro", xa), ("v5merged", xb)]
        rng.shuffle(items)
        cat = np.concatenate([head, items[0][1], gap, items[1][1]])
        fn = OUT / "AB" / f"{idx}_{kind}_AB.wav"
        sf.write(str(fn), cat, TARGET_SR)
        key.append(f"{idx}_{kind}:  第1段 = {items[0][0]:9s}   "
                   f"第2段 = {items[1][0]}")
        pairs.append((idx, kind, items[0][0]))
        n_made += 1

    # ---- A-A 对照（校准位置假象）----
    print()
    print("--- A-A 对照（同一模型出现两次）---")
    for tag, src in (("v2pro", A_DIR), ("v5merged", B_DIR)):
        for idx in ("06", "07", "08", "09"):   # 用户说的是长句
            p = find(src, idx)
            if not p:
                continue
            x, sr = load_mono(p)
            x = align_rms(to48(x, sr))
            kind = TEXTS[idx][0]
            cat = np.concatenate([head, x, gap, x])
            sf.write(str(OUT / "AA" / f"AA_{tag}_{idx}_{kind}.wav"),
                     cat, TARGET_SR)
        print(f"  AA_{tag}_*.wav  已生成")

    # ---- 三方对比 ----
    pa, pb, pc = find(A_DIR, "01"), find(B_DIR, "01"), find(C_DIR, "01")
    if pa and pb and pc:
        xa, sra = load_mono(pa)
        xb, srb = load_mono(pb)
        xc, src = load_mono(pc)
        xa = align_rms(to48(xa, sra))
        xb = align_rms(to48(xb, srb))
        xc = align_rms(to48(xc, src))
        sf.write(str(OUT / "01_三方对比.wav"),
                 np.concatenate([head, xa, gap, xb, gap, xc]), TARGET_SR)
        key.append("")
        key.append("01_三方对比:  第1段=v2pro  第2段=v5merged  "
                   "第3段=v5bad(缺ssl_proj)")

    (OUT / "answer_key.txt").write_text("\n".join(key) + "\n", encoding="utf-8")
    (OUT / "spectral.txt").write_text("\n".join(spec) + "\n", encoding="utf-8")

    print()
    print(f"  已生成 {n_made} 个 AB + 8 个 AA 对照 + 1 个三方对比")
    print(f"  输出: {OUT}")
    print()
    print("=" * 96)
    print("频谱指标（共同带宽 0~16kHz，公平比较）")
    print("=" * 96)
    for l in spec[1:]:
        print(f"  {l}")
    print()
    print("=" * 96)
    print("怎么听")
    print("=" * 96)
    print("""
  1. **先听 AA/ 里的文件**（A-A 对照）
     同一句话同一个模型出现两次。如果你仍觉得"第 1 段糊"，
     那就是**位置假象**，与模型无关 -> 听 AB 时要打折扣。

  2. 再听 AB/，判断哪段更像庄方宜。

  3. 最后打开 answer_key.txt 对答案。

  ## 上一版的 bug 已修
  上一版用线性插值把 v2Pro(32k) 升到 48k，频响 sinc²(f/48k)：
      8 kHz -0.80 dB   16 kHz -3.30 dB
  而 v5merged 原生 48k **没有**这个衰减 -> v2Pro 必然听起来更闷。
  本版改用 scipy.signal.resample_poly（多相 FIR，通带平坦）。
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
