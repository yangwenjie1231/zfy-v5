# -*- coding: utf-8 -*-
"""干净的「浊」诊断 —— 长时平均谱（LTAS）叠加对比。

## 前面三版的错误（必须记录）
    v1 `diag_zhuo.py`     : 模型 vs **留出集**（不同文本）-> 比的是内容差异
    v2 `diag_same_text.py`: 用了 `06.wav`（"武陵城外…"）而参考是"再好的武器…"
                            -> **两段完全不同的语音**，"语速差 38%" 是假的
    v3 `make_spectrogram.py`: 改成 `05.wav` 对了，但真人的"末 1.2s"**包含了
                             `default.wav` 那段文本外的多余语音**（7.48~7.82s）
                             -> 频谱图里 0.44~1.0s 的静音 + 1.0~1.2s 的额外语音
                             都是污染

**教训：每次比较前必须确认"两段音频说的是同一段内容"。**

## 本脚本的做法
    模型: `dist/ref_swap_train/orig/05.wav`
          -> 取**最后一个有声段**（= "好吗？"）
    真人: `GPT-SoVITS/ref/default.wav`
          -> 主语音在 **0~6.92s**（7.48~7.82s 是文本外多余段，**排除**）
          -> 取**最后一个有声段**（同样 = "好吗？"）

## 指标
    ① **LTAS**（长时平均谱）—— 叠加画图，直接看哪个频段多/少
    ② **频段能量占比**（9 个频段）—— 量化"浊"在哪个频段
    ③ **谱倾斜** dB/octave
    ④ **H1-H2**（第一、二谐波差）—— 声学里"发声紧/松"的标准指标
       **H1-H2 越小 = 声音越"紧/浊"**（挤压感）

## 输出
    dist/zhuo_clean/
        ltas_compare.png     LTAS 叠加（全句 + 末段）
        report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

V5 = Path(r"D:\A\zfy-v5")
OUT = V5 / "dist" / "zhuo_clean"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MODEL = V5 / "dist" / "ref_swap_train" / "orig" / "05.wav"
HUMAN = V5 / "GPT-SoVITS" / "ref" / "default.wav"
# 真人主语音的结束点（实测：6.92s 后是静音，7.48~7.82s 是文本外多余段）
HUMAN_MAIN_END = 6.95
SR = 16000

BANDS = [(50, 150), (150, 300), (300, 500), (500, 800), (800, 1200),
         (1200, 2000), (2000, 3200), (3200, 5000), (5000, 8000)]


def load16(p: Path, max_sec: float | None = None):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if max_sec is not None:
        x = x[:int(max_sec * sr)]
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return np.asarray(x, dtype=np.float64)


def speech_segments(x, sr, thresh_db=-40.0, min_gap=0.25, min_seg=0.15):
    """返回 [(start_s, end_s), ...] 有声段（合并 <min_gap 的间隔）。"""
    n = int(0.01 * sr)
    m = (len(x) // n) * n
    r = np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))
    ref = r.max() + 1e-12
    v = r > ref * (10 ** (thresh_db / 20))
    segs, st = [], None
    for i, b in enumerate(v):
        if b and st is None:
            st = i
        elif not b and st is not None:
            segs.append([st * 0.01, i * 0.01])
            st = None
    if st is not None:
        segs.append([st * 0.01, len(v) * 0.01])
    # 合并近邻
    merged = []
    for a, b in segs:
        if merged and a - merged[-1][1] < min_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= min_seg]


def ltas(x: np.ndarray, sr: int = SR, nfft: int = 2048):
    """长时平均谱（dB，归一化到总能量）。"""
    hop = nfft // 4
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    idx = range(0, len(x) - nfft, hop)
    acc = np.zeros(nfft // 2 + 1)
    cnt = 0
    for i in idx:
        seg = x[i:i + nfft]
        if np.sqrt((seg ** 2).mean()) < 1e-4:
            continue
        S = np.abs(np.fft.rfft(seg * np.hanning(nfft))) ** 2
        acc += S
        cnt += 1
    if cnt == 0:
        return None, None
    acc /= cnt
    f = np.fft.rfftfreq(nfft, 1 / sr)
    acc = acc / (acc.sum() + 1e-20)          # 归一化为能量占比
    db = 10 * np.log10(acc + 1e-14)
    return f, db


def band_share(x: np.ndarray, sr: int = SR) -> dict:
    f, db = ltas(x, sr)
    if f is None:
        return {}
    P = 10 ** (db / 10)
    out = {}
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        out[f"{lo}-{hi}"] = float(P[m].sum() / P.sum())
    # 谱倾斜
    sel = (f > 100) & (f < 6000)
    out["_tilt"] = float(np.polyfit(np.log2(f[sel]), db[sel], 1)[0]) \
        if sel.sum() > 8 else float("nan")
    out["_centroid"] = float((f * P).sum() / P.sum())
    return out


def h1h2(x: np.ndarray, sr: int = SR, f0_range=(70, 350)) -> float:
    """H1-H2（dB）：第一谐波减第二谐波。越小 = 声音越紧/浊（挤压感）。"""
    vals = []
    n = 2048
    hop = n // 2
    for i in range(0, len(x) - n, hop):
        seg = x[i:i + n]
        if np.sqrt((seg ** 2).mean()) < 1e-3:
            continue
        S = np.abs(np.fft.rfft(seg * np.hanning(n)))
        f = np.fft.rfftfreq(n, 1 / sr)
        # 找 F0（自相关）
        z = seg - seg.mean()
        ac = np.correlate(z, z, "full")[len(z) - 1:]
        ac /= (ac[0] + 1e-12)
        lo, hi = int(sr / f0_range[1]), min(int(sr / f0_range[0]), len(ac) - 1)
        if hi <= lo:
            continue
        pk = lo + int(np.argmax(ac[lo:hi]))
        f0 = sr / pk
        if not (f0_range[0] <= f0 <= f0_range[1]):
            continue
        # H1, H2 峰值（在 ±25% 内取最大）
        def peak_near(freq):
            m = (f > freq * 0.75) & (f < freq * 1.25)
            return S[m].max() if m.sum() else np.nan
        h1, h2 = peak_near(f0), peak_near(2 * f0)
        if np.isfinite(h1) and np.isfinite(h2) and h1 > 0 and h2 > 0:
            vals.append(20 * np.log10(h1 / h2))
    return float(np.median(vals)) if vals else float("nan")


print("=" * 100)
print("干净的「浊」诊断（同文本 + 排除多余段）")
print("=" * 100)

xm = load16(MODEL)
xh_full = load16(HUMAN)
xh = load16(HUMAN, max_sec=HUMAN_MAIN_END)      # **排除多余段**

sm = speech_segments(xm, SR)
sh = speech_segments(xh, SR)
print(f"  模型 {MODEL.name}: {len(xm)/SR:.2f}s   有声段 {len(sm)} 个")
for a, b in sm:
    print(f"      {a:6.2f}~{b:6.2f}s  ({b-a:.2f}s)")
print(f"  真人 {HUMAN.name}: 截到 {HUMAN_MAIN_END}s（排除文本外多余段）  "
      f"有声段 {len(sh)} 个")
for a, b in sh:
    print(f"      {a:6.2f}~{b:6.2f}s  ({b-a:.2f}s)")

# 最后一个有声段（= "好吗？"）
am, bm = sm[-1]
ah, bh = sh[-1]
tail_m = xm[int(am * SR):int(bm * SR)]
tail_h = xh[int(ah * SR):int(bh * SR)]
print()
print(f"  模型末段: {am:.2f}~{bm:.2f}s ({bm-am:.2f}s)")
print(f"  真人末段: {ah:.2f}~{bh:.2f}s ({bh-ah:.2f}s)")

# ---------- 全句 LTAS ----------
f, dbm = ltas(xm)
_, dbh = ltas(xh)
fm = band_share(xm)
fh = band_share(xh)

print()
print("=" * 100)
print("全句 频段能量占比（%）")
print("=" * 100)
print(f"  {'频段 Hz':<14s} {'模型':>9s} {'真人':>9s} {'Δ':>9s}   判读")
print("  " + "-" * 62)
for lo, hi in BANDS:
    k = f"{lo}-{hi}"
    a, b = fm.get(k, 0) * 100, fh.get(k, 0) * 100
    d = a - b
    tag = ""
    if lo < 500 and d > 2:
        tag = "**模型低频多（浊/闷）**"
    elif lo < 500 and d < -2:
        tag = "模型低频少"
    elif hi > 2000 and d < -2:
        tag = "**模型高频少（暗）**"
    elif hi > 2000 and d > 2:
        tag = "模型高频多（亮/虚）"
    print(f"  {k:<14s} {a:8.2f}% {b:8.2f}% {d:+8.2f}   {tag}")
print()
print(f"  谱质心   模型 {fm.get('_centroid',0):.0f} Hz   真人 "
      f"{fh.get('_centroid',0):.0f} Hz")
print(f"  谱倾斜   模型 {fm.get('_tilt',float('nan')):+.2f} dB/oct   "
      f"真人 {fh.get('_tilt',float('nan')):+.2f} dB/oct")

# ---------- 末段 LTAS ----------
tm = band_share(tail_m)
th = band_share(tail_h)
print()
print("=" * 100)
print(f"  末段（最后一句『好吗？』）频段能量占比（%）")
print("=" * 100)
print(f"  {'频段 Hz':<14s} {'模型':>9s} {'真人':>9s} {'Δ':>9s}   判读")
print("  " + "-" * 62)
for lo, hi in BANDS:
    k = f"{lo}-{hi}"
    a, b = tm.get(k, 0) * 100, th.get(k, 0) * 100
    d = a - b
    tag = ""
    if lo < 500 and d > 2:
        tag = "**模型低频多（浊）**"
    elif hi > 2000 and d < -2:
        tag = "**模型高频少（暗）**"
    print(f"  {k:<14s} {a:8.2f}% {b:8.2f}% {d:+8.2f}   {tag}")
print()
print(f"  谱质心   模型 {tm.get('_centroid',0):.0f} Hz   真人 "
      f"{th.get('_centroid',0):.0f} Hz")
print(f"  谱倾斜   模型 {tm.get('_tilt',float('nan')):+.2f} dB/oct   "
      f"真人 {th.get('_tilt',float('nan')):+.2f} dB/oct")

# ---------- H1-H2 ----------
hm, hh = h1h2(xm), h1h2(xh)
htm, hth = h1h2(tail_m), h1h2(tail_h)
print()
print("=" * 100)
print("H1-H2（第一/二谐波差，dB）—— 声学标准指标")
print("=" * 100)
print(f"  全句   模型 {hm:+.2f} dB   真人 {hh:+.2f} dB   Δ {hm-hh:+.2f}")
print(f"  末段   模型 {htm:+.2f} dB   真人 {hth:+.2f} dB   Δ {htm-hth:+.2f}")
print()
if htm - hth < -1.0:
    print("  -> **H1-H2 更小 = 发声更「紧/挤压」** —— 这就是「浊」的声学解释")
elif htm - hth > 1.0:
    print("  -> H1-H2 更大 = 发声更「松/气」")
else:
    print("  -> H1-H2 接近，发声方式不是主因")

# ---------- 画图 ----------
fig, axes = plt.subplots(2, 1, figsize=(13, 8))
for ax, (a, b, ttl) in zip(axes, ((xm, dbm, "全句"), (tail_m, None, "末段（好吗？）"))):
    if ttl == "全句":
        ax.plot(f, dbh, label="真人", lw=1.3, color="tab:blue")
        ax.plot(f, dbm, label="模型", lw=1.3, color="tab:red", alpha=0.85)
    else:
        f2, d2m = ltas(tail_m)
        _, d2h = ltas(tail_h)
        ax.plot(f2, d2h, label="真人", lw=1.3, color="tab:blue")
        ax.plot(f2, d2m, label="模型", lw=1.3, color="tab:red", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlim(50, 8000)
    ax.set_ylim(-95, -45)
    ax.set_xlabel("Hz", fontsize=9)
    ax.set_ylabel("能量 (dB)", fontsize=9)
    ax.set_title(f"LTAS 对比 — {ttl}   （红=模型，蓝=真人）", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    # 标注频段边界
    for lo, _ in BANDS[1:]:
        ax.axvline(lo, color="gray", lw=0.4, alpha=0.4)
fig.tight_layout()
fig.savefig(OUT / "ltas_compare.png", dpi=140)
plt.close(fig)
print()
print(f"  -> {OUT / 'ltas_compare.png'}")

print()
print("=" * 100)
print("判读")
print("=" * 100)
lo_m = sum(fm.get(f"{lo}-{hi}", 0) for lo, hi in BANDS if hi <= 500)
lo_h = sum(fh.get(f"{lo}-{hi}", 0) for lo, hi in BANDS if hi <= 500)
lo_tm = sum(tm.get(f"{lo}-{hi}", 0) for lo, hi in BANDS if hi <= 500)
lo_th = sum(th.get(f"{lo}-{hi}", 0) for lo, hi in BANDS if hi <= 500)
print(f"  低频(50~500Hz)占比:")
print(f"    全句  模型 {lo_m*100:.1f}%   真人 {lo_h*100:.1f}%   "
      f"Δ {(lo_m-lo_h)*100:+.1f}pp")
print(f"    末段  模型 {lo_tm*100:.1f}%   真人 {lo_th*100:.1f}%   "
      f"Δ {(lo_tm-lo_th)*100:+.1f}pp")
print()
if lo_tm > lo_th + 0.03:
    print("  **确认：模型末段低频能量明显偏多** —— 这是「浊」的直接原因")
    print()
    print("  ## 修复方向（与之前的高频衰减**相反**）")
    print("    · **低频搁架衰减**：150~400Hz 减 2~4dB（去掉轰鸣感）")
    print("    · **去隆隆**：80Hz 以下高通（去掉胸腔共鸣的浑浊）")
    print("    · **动态均衡**：只在句末施加，不碰主体")
elif lo_tm < lo_th - 0.03:
    print("  模型末段低频**反而更少** —— 「浊」可能来自：")
    print("    · 高频不足（暗）")
    print("    · 共振峰位置偏移")
else:
    print("  低频占比接近 —— 「浊」可能是谐波结构（H1-H2）或时域因素")

rep = ["=" * 80, "「浊」诊断（同文本 + 排除多余段）", "=" * 80, ""]
rep.append(f"模型: {MODEL.name}  末段 {am:.2f}~{bm:.2f}s")
rep.append(f"真人: {HUMAN.name}  末段 {ah:.2f}~{bh:.2f}s（主语音内）")
rep.append("")
rep.append("全句频段占比(%):")
rep.append(f"  {'频段':<12s}{'模型':>9s}{'真人':>9s}{'Δ':>9s}")
for lo, hi in BANDS:
    k = f"{lo}-{hi}"
    rep.append(f"  {k:<12s}{fm.get(k,0)*100:8.2f}%{fh.get(k,0)*100:8.2f}%"
               f"{(fm.get(k,0)-fh.get(k,0))*100:+8.2f}")
rep.append(f"  谱质心 模型 {fm.get('_centroid',0):.0f} 真人 {fh.get('_centroid',0):.0f}")
rep.append("")
rep.append("末段频段占比(%):")
for lo, hi in BANDS:
    k = f"{lo}-{hi}"
    rep.append(f"  {k:<12s}{tm.get(k,0)*100:8.2f}%{th.get(k,0)*100:8.2f}%"
               f"{(tm.get(k,0)-th.get(k,0))*100:+8.2f}")
rep.append("")
rep.append(f"H1-H2  全句 模型 {hm:+.2f} 真人 {hh:+.2f}")
rep.append(f"H1-H2  末段 模型 {htm:+.2f} 真人 {hth:+.2f}")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
print()
print(f"  -> {OUT / 'report.txt'}")
print("=" * 100)
