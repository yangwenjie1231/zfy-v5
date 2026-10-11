# -*- coding: utf-8 -*-
"""同文本 A/B：模型 vs 真人说**同一句话** —— 最直接的「浊」诊断。

## 为什么这是最干净的实验
`ref/default.wav` 的文本是：
    「再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？」

而测试文本第 6 句**一字不差就是这句**。
所以：
    A = `dist/ref_swap_train/orig/06.wav`   （模型说这句）
    B = `GPT-SoVITS/ref/default.wav`        （**真人**说这句）

**同文本、同角色** -> 任何差异都是"合成质量"而非"内容差异"。
这比之前的"模型 vs 留出集（不同文本）"可靠得多。

## 「浊」的细化指标（比之前更贴切）
    ① **尾字时长**：浊 = 尾字拖长/含混
    ② **谱通量（spectral flux）末段**：
       真人句末有**快速的频谱变化**（辅音收尾）
       浊 = 变化慢 -> 频谱"糊"在一起
    ③ **鼻音共振峰**（末字若为 n/ng，250~350Hz 有能量）
    ④ **谐噪比（HNR）**：浊 = 谐波结构不清
    ⑤ **逐帧谱质心轨迹**：看末尾是否"塌下去"

## 时间对齐
两段时长不同（模型 8.74s vs 真人 7.95s），
所以按**归一化时间轴**（0~100%）比较轨迹，而不是绝对时间。

## 输出
    dist/same_text_ab/
        report.txt
        05_model.wav / 05_human.wav  （响度对齐版，便于直接听）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
# ⚠️ 必须是 **05.wav**，不是 06.wav！
#    `compare_ref_swap.py` 的 TEXTS 是 0-based：
#      索引 4 -> 05.wav = "再好的武器…好吗？"  <- 与 ref/default.wav **同文本**
#      索引 5 -> 06.wav = "武陵城外有一片梨树林…" <- 我第一版错误地用了这个
#    结果比的是两段完全不同的语音，所有指标无效（"语速差 38%" 是假的）。
MODEL = V5 / "dist" / "ref_swap_train" / "orig" / "05.wav"
HUMAN = V5 / "GPT-SoVITS" / "ref" / "default.wav"
OUT = V5 / "dist" / "same_text_ab"


def load(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return x, int(sr)


def to16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return x
    try:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(int(sr), 16000)
        return resample_poly(x, 16000 // g, int(sr) // g).astype(np.float64)
    except Exception:  # noqa: BLE001
        n = int(round(len(x) * 16000 / sr))
        return np.interp(np.linspace(0, len(x) - 1, n),
                         np.arange(len(x)), x).astype(np.float64)


def frame_feats(x: np.ndarray, sr: int = 16000, win: float = 0.025,
                hop: float = 0.010):
    """逐帧：谱质心 / 谱通量 / HNR / 低频能量（鼻音）。"""
    n = int(win * sr)
    h = int(hop * sr)
    if len(x) < n + h:
        return None
    idx = range(0, len(x) - n, h)
    cents, flux, hnrs, lows, rms = [], [], [], [], []
    prev = None
    for i in idx:
        seg = x[i:i + n] * np.hanning(n)
        S = np.abs(np.fft.rfft(seg)) + 1e-12
        f = np.fft.rfftfreq(n, 1 / sr)
        P = S ** 2
        Pn = P / (P.sum() + 1e-18)
        cents.append(float((f * Pn).sum()))
        lows.append(float(Pn[(f >= 200) & (f <= 400)].sum()))
        rms.append(float(np.sqrt((seg ** 2).mean())))
        if prev is not None:
            # 谱通量（归一化）
            a = S / (S.sum() + 1e-12)
            b = prev / (prev.sum() + 1e-12)
            flux.append(float(np.abs(a - b).sum()))
        prev = S
        # HNR 代理：自相关峰值
        z = seg - seg.mean()
        if len(z) > 32:
            ac = np.correlate(z, z, "full")[len(z) - 1:]
            ac = ac / (ac[0] + 1e-12)
            lo, hi = int(sr / 400), min(int(sr / 80), len(ac) - 1)
            hnrs.append(float(ac[lo:hi].max()) if hi > lo else np.nan)
        else:
            hnrs.append(np.nan)
    return {"cent": np.array(cents), "flux": np.array(flux),
            "hnr": np.array(hnrs), "low": np.array(lows),
            "rms": np.array(rms), "hop": hop}


def voiced_span(f: dict, thresh_db: float = -40.0):
    r = f["rms"]
    ref = r.max() + 1e-12
    v = np.where(r > ref * (10 ** (thresh_db / 20)))[0]
    return (int(v[0]), int(v[-1])) if len(v) else (0, len(r) - 1)


print("=" * 100)
print("同文本 A/B：模型 vs 真人（同一句话）")
print("=" * 100)
print("  文本: 再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？")
print(f"  模型: {MODEL.name}   （合成）")
print(f"  真人: {HUMAN.name}   （参考音频，同一句）")

xm, srm = load(MODEL)
xh, srh = load(HUMAN)
m16, h16 = to16k(xm, srm), to16k(xh, srh)
fm, fh = frame_feats(m16), frame_feats(h16)
if fm is None or fh is None:
    print("  !! 特征提取失败")
    sys.exit(1)

sm, em = voiced_span(fm)
sh, eh = voiced_span(fh)
dur_m = (em - sm) * fm["hop"]
dur_h = (eh - sh) * fh["hop"]
print()
print(f"  模型: 总 {len(xm)/srm:.2f}s   有声 {dur_m:.2f}s   帧 {len(fm['rms'])}")
print(f"  真人: 总 {len(xh)/srh:.2f}s   有声 {dur_h:.2f}s   帧 {len(fh['rms'])}")
print(f"  **语速差**: 模型比真人 {'慢' if dur_m > dur_h else '快'} "
      f"{abs(dur_m-dur_h)/dur_h*100:.0f}%")

print()
print("=" * 100)
print("归一化时间轴对比（0% = 句首，100% = 句末）")
print("=" * 100)
print(f"  {'位置':>6s} {'谱质心(模/真)':>16s} {'谱通量(模/真)':>16s} "
      f"{'HNR(模/真)':>14s} {'低频(模/真)':>14s}")
print("  " + "-" * 74)
grid = [0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100]
for pct in grid:
    im = sm + int((em - sm) * pct / 100)
    ih = sh + int((eh - sh) * pct / 100)
    im = min(im, len(fm["cent"]) - 1)
    ih = min(ih, len(fh["cent"]) - 1)
    jm = min(im, len(fm["flux"]) - 1)
    jh = min(ih, len(fh["flux"]) - 1)
    print(f"  {pct:5d}% {fm['cent'][im]:7.0f}/{fh['cent'][ih]:<7.0f} "
          f"{fm['flux'][jm]:7.4f}/{fh['flux'][jh]:<7.4f} "
          f"{fm['hnr'][im]:6.3f}/{fh['hnr'][ih]:<6.3f} "
          f"{fm['low'][im]:6.4f}/{fh['low'][ih]:<6.4f}")

# ---------- 末段专项 ----------
print()
print("=" * 100)
print("末段专项（最后 20% 有声区）")
print("=" * 100)
m0 = sm + int((em - sm) * 0.8)
h0 = sh + int((eh - sh) * 0.8)


def seg_stat(f, a, b, key):
    v = f[key][a:b]
    v = v[~np.isnan(v)] if v.dtype.kind == "f" else v
    return float(np.mean(v)) if len(v) else float("nan")


for key, name in (("cent", "谱质心"), ("flux", "谱通量"),
                  ("hnr", "HNR"), ("low", "低频能量")):
    a = seg_stat(fm, m0, em, key)
    b = seg_stat(fh, h0, eh, key)
    d = (a - b) / (abs(b) + 1e-12) * 100
    tag = ""
    if key == "cent":
        tag = "**模型更闷（浊）**" if a < b * 0.95 else (
            "模型更亮" if a > b * 1.05 else "接近")
    elif key == "flux":
        tag = "**模型变化慢（糊）**" if a < b * 0.85 else (
            "模型变化快" if a > b * 1.15 else "接近")
    elif key == "hnr":
        tag = "**模型谐波弱（浊）**" if a < b * 0.95 else (
            "模型更清晰" if a > b * 1.05 else "接近")
    else:
        tag = "**模型低频重（闷）**" if a > b * 1.2 else (
            "模型低频少" if a < b * 0.8 else "接近")
    print(f"  {name:<10s} 模型 {a:9.4f}   真人 {b:9.4f}   "
          f"Δ {d:+7.1f}%   {tag}")

# ---------- 结论 ----------
print()
print("=" * 100)
print("判读")
print("=" * 100)
c_m, c_h = seg_stat(fm, m0, em, "cent"), seg_stat(fh, h0, eh, "cent")
f_m, f_h = seg_stat(fm, m0, em, "flux"), seg_stat(fh, h0, eh, "flux")
h_m, h_h = seg_stat(fm, m0, em, "hnr"), seg_stat(fh, h0, eh, "hnr")
issues = []
if c_m < c_h * 0.95:
    issues.append(f"谱质心低 {c_m:.0f} vs {c_h:.0f}（**闷/浊**）")
if f_m < f_h * 0.85:
    issues.append(f"谱通量低 {f_m:.4f} vs {f_h:.4f}（**频谱变化慢 = 糊**）")
if h_m < h_h * 0.95:
    issues.append(f"HNR 低 {h_m:.3f} vs {h_h:.3f}（**谐波不清**）")
if issues:
    print("  **确认尾部问题**:")
    for s in issues:
        print(f"    · {s}")
else:
    print("  末段指标与真人接近 —— 「浊」可能来自别的因素：")
    print("    · 辅音缺失（句末 n/ng 未发出）")
    print("    · 音节边界模糊")
    print("    · 韵律重音位置不同")

# ---------- 导出可听版本 ----------
OUT.mkdir(parents=True, exist_ok=True)


def norm48(x, sr, target=0.06):
    try:
        from math import gcd

        from scipy.signal import resample_poly
        if sr != 48000:
            g = gcd(int(sr), 48000)
            x = resample_poly(x, 48000 // g, int(sr) // g)
        sr = 48000
    except Exception:  # noqa: BLE001
        pass
    r = float(np.sqrt((x ** 2).mean())) + 1e-12
    y = x * (target / r)
    pk = float(np.abs(y).max())
    return (y * (0.99 / pk) if pk > 0.99 else y), sr


ym, _ = norm48(xm.copy(), srm)
yh, _ = norm48(xh.copy(), srh)
sf.write(str(OUT / "05_model.wav"), ym, 48000)
sf.write(str(OUT / "05_human.wav"), yh, 48000)
# 交替拼接：模型 -> 真人 -> 模型 -> 真人
gap = np.zeros(int(0.6 * 48000))
sf.write(str(OUT / "05_alt.wav"),
         np.concatenate([ym, gap, yh, gap, ym, gap, yh]), 48000)

print()
print("=" * 100)
print("可听文件")
print("=" * 100)
print(f"  {OUT / '05_model.wav'}   模型（合成）")
print(f"  {OUT / '05_human.wav'}   真人（参考音频，**同一句话**）")
print(f"  {OUT / '05_alt.wav'}     交替: 模型->真人->模型->真人")
print()
print("  **听法**: 只听最后一个字（『好吗』的『吗』）")
print("    真人: 应该有个清晰的鼻音收尾 m->a->？")
print("    模型: 如果是『浊』，会听起来含糊、拖沓、没有明确的收尾动作")
print("=" * 100)

rep = ["=" * 78, "同文本 A/B（模型 vs 真人，同一句）", "=" * 78, ""]
rep.append("文本: 再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？")
rep.append(f"模型 {len(xm)/srm:.2f}s（有声 {dur_m:.2f}s）")
rep.append(f"真人 {len(xh)/srh:.2f}s（有声 {dur_h:.2f}s）")
rep.append("")
rep.append("末段（最后 20% 有声区）:")
for key, name in (("cent", "谱质心"), ("flux", "谱通量"),
                  ("hnr", "HNR"), ("low", "低频能量")):
    rep.append(f"  {name:<10s} 模型 {seg_stat(fm,m0,em,key):9.4f}  "
               f"真人 {seg_stat(fh,h0,eh,key):9.4f}")
rep.append("")
rep.append("归一化轨迹:")
for pct in grid:
    im = min(sm + int((em - sm) * pct / 100), len(fm["cent"]) - 1)
    ih = min(sh + int((eh - sh) * pct / 100), len(fh["cent"]) - 1)
    rep.append(f"  {pct:3d}%  cent {fm['cent'][im]:6.0f}/{fh['cent'][ih]:6.0f}"
               f"  hnr {fm['hnr'][im]:5.3f}/{fh['hnr'][ih]:5.3f}")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
print(f"  -> {OUT / 'report.txt'}")
