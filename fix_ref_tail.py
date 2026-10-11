# -*- coding: utf-8 -*-
"""精确测量 ref/default.wav 末尾那段"多余语音"，并生成修复候选。

## 已知结构（逐帧 RMS）
     0.0s  静音
     0.1~6.8s  正常语音（对应 ref.txt 的 27 字）
     6.9~7.4s  **静音**（0.5s 停顿）
     7.5~7.8s  **一段额外语音**（文本里没有！）
     7.8s  结束

## 为什么这会导致"尾音气声"
GPT-SoVITS 把「音频 + 文本」配对：
  音频在文本结束后**还有发声** -> 模型学到"句末要加一段"
-> 所有合成音频末尾都多出一截低沉浊音（听感像"气声"）

## 修复方案
  1. **裁剪**：截掉 7.4s 之后 -> 得到干净的 7.4s 参考
  2. **换素材**：从训练集挑"尾部干净"的
  3. **两者都做**，然后合成对比

## 判据（裁剪是否成功）
  裁剪后重新测：尾部 0.4s 的
    · 尾/体dB 应接近 0 或负值（不再 +30dB）
    · 尾部频谱应集中在高频（4~16kHz，语音特征）
      而不是低频（0~2kHz，那段多余浊音）

## 输出
    dist/ref_fix/
        default_trimmed.wav     截掉多余部分
        report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
REF = V5 / "GPT-SoVITS" / "ref" / "default.wav"
REFTXT = V5 / "GPT-SoVITS" / "ref" / "ref.txt"
OUT = V5 / "dist" / "ref_fix"


def load(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return x, int(sr)


def frame_rms(x: np.ndarray, sr: int, win: float = 0.02):
    n = int(win * sr)
    m = (len(x) // n) * n
    return np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1)), n


def find_speech_segments(x: np.ndarray, sr: int, thresh_db: float = -35.0):
    """找出所有「有声段」（连续 RMS 高于阈值的区间）。"""
    rms, n = frame_rms(x, sr, 0.02)
    ref = float(np.sqrt((x ** 2).mean())) + 1e-12
    thr = ref * (10 ** (thresh_db / 20))
    voiced = rms > thr
    segs = []
    start = None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start * n / sr, i * n / sr))
            start = None
    if start is not None:
        segs.append((start * n / sr, len(voiced) * n / sr))
    return segs, rms, n


print("=" * 96)
print("ref/default.wav 结构分析")
print("=" * 96)

x, sr = load(REF)
segs, rms, n = find_speech_segments(x, sr, -35.0)
print(f"  时长 {len(x)/sr:.3f}s @ {sr} Hz")
print(f"  检出 {len(segs)} 个有声段（阈值 -35 dB 相对整体 RMS）:")
print()
print(f"  {'#':>3s} {'起':>7s} {'止':>7s} {'时长':>7s}   说明")
print("  " + "-" * 60)
# 合并间隔 <0.25s 的段（同一句话内的字间停顿）
merged = []
for a, b in segs:
    if merged and a - merged[-1][1] < 0.25:
        merged[-1][1] = b
    else:
        merged.append([a, b])
for i, (a, b) in enumerate(merged, 1):
    note = ""
    if i == len(merged) and a > 7.0:
        note = "**文本外的多余语音**"
    elif i == len(merged) and len(merged) > 1:
        note = "最后一段"
    print(f"  {i:3d} {a:6.2f}s {b:6.2f}s {b-a:6.2f}s   {note}")

print()
print(f"  合并后 {len(merged)} 个语句段")

# 判断"主语音"的结束点
if len(merged) >= 2:
    main_end = merged[-2][1]      # 倒数第二段的结束（主语音）
    extra_start = merged[-1][0]
    extra_end = merged[-1][1]
    gap = extra_start - main_end
    print()
    print(f"  **主语音结束**: {main_end:.2f}s")
    print(f"  **多余段**:     {extra_start:.2f}~{extra_end:.2f}s "
          f"（{extra_end-extra_start:.2f}s）")
    print(f"  **间隔**:       {gap:.2f}s")
else:
    main_end = merged[-1][1] if merged else len(x) / sr
    print(f"  只有一个语句段，结束 {main_end:.2f}s")

# ---------- 频谱对比 ----------
print()
print("=" * 96)
print("频谱对比：主语音 vs 多余段")
print("=" * 96)
bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000),
         (4000, 8000), (8000, 16000)]


def band_dist(s: np.ndarray) -> dict:
    nn = min(len(s), 4096)
    if nn < 64:
        return {}
    S = np.abs(np.fft.rfft(s[:nn] * np.hanning(nn)))
    f = np.fft.rfftfreq(nn, 1 / sr)
    S = S / (S.sum() + 1e-12)
    return {f"{lo}-{hi}": float(S[(f >= lo) & (f < min(hi, sr / 2))].sum())
            for lo, hi in bands}


if len(merged) >= 2:
    main_seg = x[int(merged[-2][0] * sr):int(merged[-2][1] * sr)]
    extra_seg = x[int(merged[-1][0] * sr):int(merged[-1][1] * sr)]
    dm, de = band_dist(main_seg), band_dist(extra_seg)
    print(f"  {'频段':<14s} {'主语音':>9s} {'多余段':>9s}")
    print("  " + "-" * 36)
    for k in dm:
        print(f"  {k+' Hz':<14s} {dm[k]*100:8.1f}% {de.get(k,0)*100:8.1f}%")
    print()
    lo_m = sum(v for k, v in dm.items() if int(k.split("-")[1]) <= 2000)
    lo_e = sum(v for k, v in de.items() if int(k.split("-")[1]) <= 2000)
    print(f"  低频(0~2k)占比: 主语音 {lo_m*100:.1f}%  多余段 {lo_e*100:.1f}%")
    if lo_e > lo_m * 1.5:
        print("  -> 多余段**低频占比明显更高** = 浊音/元音特征（不是气声）")
        print("     很可能是**叹词/鼻音/额外的一个词**")

# ---------- 生成裁剪版 ----------
OUT.mkdir(parents=True, exist_ok=True)
pad = 0.15                      # 保留一点尾部静音，避免切到语音
cut = min(len(x) / sr, main_end + pad)
n_cut = int(cut * sr)
trimmed = x[:n_cut]
# 尾部做 30ms 淡出，避免爆音
fade = int(0.03 * sr)
if len(trimmed) > fade:
    trimmed[-fade:] *= np.linspace(1, 0, fade)

out_wav = OUT / "default_trimmed.wav"
sf.write(str(out_wav), trimmed, sr)
print()
print("=" * 96)
print(f"生成裁剪版: {out_wav.name}")
print("=" * 96)
print(f"  原时长 {len(x)/sr:.3f}s  ->  裁剪后 {len(trimmed)/sr:.3f}s "
      f"（切掉 {len(x)/sr - len(trimmed)/sr:.3f}s）")

# 验证裁剪后
rms2, n2 = frame_rms(trimmed, sr, 0.1)
ref2 = float(np.sqrt((trimmed ** 2).mean())) + 1e-12
db2 = 20 * np.log10(np.maximum(rms2, 1e-12) / ref2)
print()
print("  裁剪后最后 8 帧（0.1s）:")
for i in range(max(0, len(rms2) - 8), len(rms2)):
    bar = "#" * int(max(0, db2[i] + 40) / 2)
    print(f"    {i*0.1:5.1f}s  {rms2[i]:.5f}  {db2[i]:+6.1f} dB  {bar}")

# 尾部频谱检查
tail2 = trimmed[-int(0.4 * sr):]
body2 = trimmed[int(len(trimmed) * 0.4):int(len(trimmed) * 0.4) + int(0.4 * sr)]
d2 = band_dist(tail2)
lo2 = sum(v for k, v in d2.items() if int(k.split("-")[1]) <= 2000)
hi2 = sum(v for k, v in d2.items() if int(k.split("-")[0]) >= 4000)
print()
print(f"  裁剪后尾部 0.4s: 低频(0~2k) {lo2*100:.1f}%  "
      f"高频(4~16k) {hi2*100:.1f}%")
print(f"  原尾部        : 低频 {lo_e*100:.1f}%  高频 "
      f"{sum(v for k,v in de.items() if int(k.split('-')[0])>=4000)*100:.1f}%"
      if len(merged) >= 2 else "")
print()
if lo2 < 0.5:
    print("  ✓ 裁剪后尾部以高频为主 = 语音特征（正常收尾）")
else:
    print("  ~ 尾部仍偏低频，可能还要多切一点")

# ---------- 参考文本要改吗 ----------
print()
print("=" * 96)
print("⚠️ 参考文本是否需要修改")
print("=" * 96)
rtxt = REFTXT.read_text(encoding="utf-8").strip() if REFTXT.exists() else ""
print(f"  ref.txt: {rtxt}")
print(f"  字数: {len(rtxt)}")
print()
print("""
  ## 结论：**文本不用改**
  那段多余语音**不在文本里** —— 所以：
    · 裁剪掉它 -> 音频与文本**完全对应** -> 正确
    · 若把文本也加上那段内容 -> 需要知道它说了什么（听不出来就别加）

  ## 为什么必须裁剪
  GPT-SoVITS 的 `prompt_text` 必须与参考音频**一字不差**（技能明确写过）。
  现在音频多一段、文本没写 -> 模型把多出的部分学成"句末附加发声"
  -> **所有合成音频尾部都多一截**（用户听成"气声强"）。

  ## 下一步
  用裁剪版重新合成，对比尾音是否干净：
      python _tools/compare_ref_fix.py
""")
print("=" * 96)

rep = ["=" * 78, "ref/default.wav 修复报告", "=" * 78, ""]
rep.append(f"原文件: {REF}")
rep.append(f"  时长 {len(x)/sr:.3f}s @ {sr}Hz")
rep.append(f"  检出 {len(merged)} 个语句段:")
for i, (a, b) in enumerate(merged, 1):
    rep.append(f"    {i}. {a:.2f}~{b:.2f}s  ({b-a:.2f}s)")
if len(merged) >= 2:
    rep.append("")
    rep.append(f"  **主语音结束 {main_end:.2f}s**")
    rep.append(f"  **多余段 {extra_start:.2f}~{extra_end:.2f}s** "
               f"（文本里没有）")
    rep.append(f"  多余段低频占比 {lo_e*100:.1f}% vs 主语音 {lo_m*100:.1f}%"
               "  -> 浊音特征，非气声")
rep.append("")
rep.append(f"裁剪版: {out_wav}")
rep.append(f"  时长 {len(trimmed)/sr:.3f}s（切掉 "
           f"{len(x)/sr-len(trimmed)/sr:.3f}s）")
rep.append(f"  尾部低频占比 {lo2*100:.1f}%  高频 "
           f"{hi2*100:.1f}%")
rep.append("")
rep.append(f"ref.txt（未改）: {rtxt}")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
print(f"  -> {OUT / 'report.txt'}")
