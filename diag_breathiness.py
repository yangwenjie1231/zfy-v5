# -*- coding: utf-8 -*-
"""诊断听感：v5merged 真的"偏糊"吗？v2pro 真的"气音轻"吗？

## 用户报告（长句）
    "后一段的气音会轻很多，前一段会偏糊一点"

## 首要问题：模型差异 vs **位置假象**
我的 AB 文件结构：[头静音 0.5s] + 段1 + [间隔 0.7s] + 段2
答案表：
    06_长  段1=v5merged   段2=v2pro
    07_长  段1=v5merged   段2=v2pro
    08_长  段1=v2pro      段2=v5merged
    09_长  段1=v2pro      段2=v5merged

  · 若"段1糊、段2气音轻"在**所有**文件里都成立 -> **位置假象**（我的构造问题）
  · 若结论随模型而变                              -> **真实模型差异**

## 客观代理指标
  · 谱质心 (spectral centroid)   越低越"糊/闷"
  · 高频比 (>4kHz / <4kHz)       越高越"气音重/亮"
  · 波峰因数 (peak/RMS)          影响响度对齐后的实际听感

## 本脚本
  1. 两个来源的原始参数（**未经我处理**，最可信）
  2. 逐句对比 v5merged vs v2pro
  3. 验证我的响度对齐是否真的对齐（RMS 应都 ≈ 0.06）
  4. 检查峰值限制有没有触发（触发会让那一段变轻）
  5. 拆解 AB 文件，验证段1/段2 的实际电平
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
A_DIR = V5 / "dist" / "baseline_v2pro"
B_DIR = V5 / "dist" / "sweep2" / "out_sweep" / "e16_merged"
AB_DIR = V5 / "dist" / "listen_v5_vs_v2pro"

TARGET_SR = 48000
TARGET_RMS = 0.06
HEAD_S, GAP_S = 0.5, 0.7

# 答案表：哪个模型在段1
KEY = {
    "01": "v2pro", "02": "v2pro", "03": "v2pro", "04": "v2pro",
    "05": "v2pro", "06": "v5merged", "07": "v5merged", "08": "v2pro",
    "09": "v2pro", "10": "v5merged",
}
TEXTS = {
    "01": "让我想想。", "02": "好。", "03": "迎敌或是迎客，都得整装后再出发。",
    "04": "出门前，要不要先喝杯茶提提神？", "05": "嗯……我理一理。",
    "06": "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？",
    "07": "武陵城外有一片梨树林，每年春天都会开满白色的花。",
    "08": "她就在那阵花香里，只是他自己不知道。",
    "09": "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。",
    "10": "这件事我三点五十分再去处理。",
}


def load(p: Path):
    x, sr = sf.read(str(p), always_2d=True)
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return x.astype(np.float64), sr


def resample_lin(x, sr, t):
    if sr == t:
        return x
    n = int(round(len(x) * t / sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def spec_stats(x, sr):
    n, hop = 2048, 512
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    w = np.hanning(n)
    mag = np.abs(np.fft.rfft(frames * w, axis=1)).mean(axis=0)
    f = np.fft.rfftfreq(n, 1 / sr)
    tot = mag.sum() + 1e-12
    centroid = float((f * mag).sum() / tot)
    lo = mag[f < 4000].sum()
    hi = mag[f >= 4000].sum()
    return centroid, float(hi / (lo + 1e-12))


def stats(x, sr):
    rms = float(np.sqrt((x ** 2).mean()))
    peak = float(np.abs(x).max())
    c, hf = spec_stats(x, sr)
    return dict(sr=sr, dur=len(x) / sr, rms=rms, peak=peak,
                crest=peak / (rms + 1e-12), centroid=c, hf=hf)


def find(d: Path, idx: str):
    hits = sorted(d.glob(f"{idx}_*.wav")) if d.is_dir() else []
    return hits[0] if hits else None


print("=" * 108)
print("A. 原始来源（未经任何处理，最可信）")
print("=" * 108)

rows = []
for i in range(1, 11):
    idx = f"{i:02d}"
    pa, pb = find(A_DIR, idx), find(B_DIR, idx)
    if not (pa and pb):
        print(f"  {idx}: 缺文件 v2pro={bool(pa)} merged={bool(pb)}")
        continue
    xa, sra = load(pa)
    xb, srb = load(pb)
    sa, sb = stats(xa, sra), stats(xb, srb)
    rows.append((idx, sa, sb))

print(f"  {'#':4s} {'字':>3s} │ {'v2pro':>7s} {'sr':>6s} {'峰':>6s} {'因数':>5s} {'质心':>6s} {'高频':>6s}"
      f" │ {'merged':>7s} {'sr':>6s} {'峰':>6s} {'因数':>5s} {'质心':>6s} {'高频':>6s}")
print("  " + "-" * 104)
for idx, sa, sb in rows:
    n = len(TEXTS[idx])
    print(f"  {idx:4s} {n:3d} │ {sa['dur']:6.2f}s {sa['sr']:6d} {sa['peak']:6.3f} "
          f"{sa['crest']:5.2f} {sa['centroid']:6.0f} {sa['hf']:6.3f}"
          f" │ {sb['dur']:6.2f}s {sb['sr']:6d} {sb['peak']:6.3f} "
          f"{sb['crest']:5.2f} {sb['centroid']:6.0f} {sb['hf']:6.3f}")

if rows:
    print()
    print("=" * 108)
    print("B. 汇总：v5merged 相对 v2pro 的差异（负 = 更低/更少）")
    print("=" * 108)
    dd = [r[2]["dur"] - r[1]["dur"] for r in rows]
    dc = [r[2]["centroid"] - r[1]["centroid"] for r in rows]
    dh = [r[2]["hf"] - r[1]["hf"] for r in rows]
    dcr = [r[2]["crest"] - r[1]["crest"] for r in rows]
    print(f"  时长差  : 均值 {np.mean(dd):+.3f}s   "
          f"（全部 {min(dd):+.3f} ~ {max(dd):+.3f}）")
    print(f"  谱质心差: 均值 {np.mean(dc):+.0f} Hz  "
          f"（{'merged 更糊/更闷' if np.mean(dc) < 0 else 'merged 更亮'}）")
    print(f"  高频比差: 均值 {np.mean(dh):+.4f}    "
          f"（{'merged 气音更少' if np.mean(dh) < 0 else 'merged 气音更多'}）")
    print(f"  波峰因数: 均值 {np.mean(dcr):+.3f}")
    print()
    print("  **时长差的分布**（若全是同一个常数 -> 固定偏移，不是模型差异）:")
    print(f"    {[f'{d:+.3f}' for d in dd]}")
    if max(dd) - min(dd) < 0.02:
        print("    !! 差异几乎是常数 -> **固定偏移**（如尾部静音长度不同），")
        print("       不是模型生成的时长差异")

print()
print("=" * 108)
print("C. 我的处理链：响度对齐是否真的对齐？峰值限制是否触发？")
print("=" * 108)
print(f"  {'#':4s} │ {'v2pro RMS':>10s} {'限幅':>5s} │ {'merged RMS':>11s} {'限幅':>5s}")
print("  " + "-" * 50)
limit_hits = 0
for idx, sa, sb in rows:
    pa, pb = find(A_DIR, idx), find(B_DIR, idx)
    xa, sra = load(pa)
    xb, srb = load(pb)
    out = []
    for x, sr in ((xa, sra), (xb, srb)):
        y = resample_lin(x, sr, TARGET_SR)
        r = float(np.sqrt((y ** 2).mean()))
        y = y * (TARGET_RMS / (r + 1e-12))
        pk = float(np.abs(y).max())
        lim = pk > 0.99
        if lim:
            limit_hits += 1
            y = y * (0.99 / pk)
        out.append((float(np.sqrt((y ** 2).mean())), lim))
    print(f"  {idx:4s} │ {out[0][0]:10.4f} {str(out[0][1]):>5s} │ "
          f"{out[1][0]:11.4f} {str(out[1][1]):>5s}")
print()
print(f"  峰值限制触发次数: {limit_hits}  "
      f"（>0 说明有一段被压低了 -> 会影响听感）")

print()
print("=" * 108)
print("D. 位置假象检验：拆开 AB 文件，看段1 vs 段2")
print("=" * 108)
print(f"  {'#':4s} {'段1模型':>9s} {'段1 RMS':>9s} │ {'段2模型':>9s} {'段2 RMS':>9s}  {'判读'}")
print("  " + "-" * 88)
seg1_louder = 0
for idx, sa, sb in rows:
    # ⚠️ 不要用 Path + str（会 TypeError），用 glob 直接找
    cands = sorted(AB_DIR.glob(f"{idx}_*_AB.wav"))
    if not cands:
        continue
    ab = cands[0]
    x, sr = load(ab)
    n_head = int(HEAD_S * sr)
    n_gap = int(GAP_S * sr)
    first = KEY[idx]
    d1 = (len(resample_lin(load(find(A_DIR, idx))[0],
                            load(find(A_DIR, idx))[1], sr))
          if first == "v2pro" else
          len(resample_lin(load(find(B_DIR, idx))[0],
                           load(find(B_DIR, idx))[1], sr)))
    s1 = x[n_head:n_head + d1]
    s2 = x[n_head + d1 + n_gap:]
    r1 = float(np.sqrt((s1 ** 2).mean())) if len(s1) else 0
    r2 = float(np.sqrt((s2 ** 2).mean())) if len(s2) else 0
    second = "v5merged" if first == "v2pro" else "v2pro"
    note = ""
    if r1 > r2 * 1.05:
        note = "段1更响"
        seg1_louder += 1
    elif r2 > r1 * 1.05:
        note = "段2更响"
    else:
        note = "相当"
    print(f"  {idx:4s} {first:>9s} {r1:9.4f} │ {second:>9s} {r2:9.4f}  {note}")

print()
print("=" * 108)
print("E. 判读")
print("=" * 108)
print(f"""
  段1 更响的次数: {seg1_louder} / {len(rows)}
  （我的处理对两段用**完全相同的** RMS 对齐 + 限幅逻辑，
    所以理论上段1/段2 电平应当相当。若系统性偏向某一段，
    那是**位置假象**，不是模型差异。）

  ## 若用户觉得"段1糊、段2气音轻"在**所有**文件里都成立
     -> 位置假象。可能来源：
        · 段1 紧跟在 0.5s 静音后，听觉上"起音"被强调
        · 段2 前面有 0.7s 间隔，人耳会做响度重校准
        · 线性插值重采样对首尾样本的处理略有不同
     -> 解法：换成"同一模型 A-A 对照"来校准，或改用交叉排列

  ## 若结论随模型而变（06/07 与 08/09 相反）
     -> **真实模型差异**：v5merged 谱质心更低 = 更"糊"
        这正是 `enc_p` 用底模原值导致的音质损失 —— 也就是
        ZFY_UNFREEZE 训练要解决的问题。
""")
print("=" * 108)
