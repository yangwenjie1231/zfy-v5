# -*- coding: utf-8 -*-
"""诊断「尾音音高上翘 + 虚」—— 这是**韵律**问题，不是音色问题。

## 用户的描述（关键）
    「尾音它声音会往上高一点，然后呢，会虚一点」
      · 往上高 = **音高上翘（升调）**   <- 韵律，GPT 侧
      · 虚     = 单薄、没底气           <- 可能与升调同源

## 我的假设（强）
`ref/default.wav` 的文本以**问句**结尾：
    「再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，**好吗？**」
中文问句天然**升调**。GPT-SoVITS 会模仿参考音频的**韵律风格**
-> **所有合成句子（包括陈述句）都带升调结尾**

## 这能一次解释两个症状
    升调结尾 -> 音高往上高  ✓
    升调时声带拉紧、能量散 -> 听感"虚"  ✓

**也能解释为什么我的频谱修复"只好一点点"** —— 我修的是**音色**，
而问题是**韵律**（F0 走向）。方向错了。

## 验证方法：F0 轨迹
    ① `ref/default.wav` 结尾的 F0 走向（应该是升）
    ② 模型输出结尾的 F0 走向（若 ≈ ① -> 假设成立）
    ③ **真人陈述句**结尾的 F0 走向（应该是降/平）
    ④ 参考音频的"好吗？"段 vs 模型"好吗？"段（同文本）

## 指标
    · F0 轨迹（半音，相对句中位数）
    · **末段斜率**（最后 300ms，semitone/s）—— 正=升，负=降
    · 末段 vs 主体 的 F0 差
    · 能量与 HNR（看"虚"）

## 输出
    dist/prosody/
        f0_compare.png     F0 轨迹对比图
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
OUT = V5 / "dist" / "prosody"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MODEL_DIR = V5 / "dist" / "ref_swap_train" / "orig"
REF = V5 / "GPT-SoVITS" / "ref" / "default.wav"
REF_END = 6.95                 # 参考音频主语音结束（排除文本外多余段）
HELD = V5 / "dist" / "heldout" / "heldout_clean.json"
SR = 16000


def load16(p: Path, max_sec=None):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if max_sec:
        x = x[:int(max_sec * sr)]
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return np.asarray(x, dtype=np.float64)


def f0_track(x: np.ndarray, sr: int = SR, win=0.04, hop=0.01,
             fmin=70.0, fmax=400.0):
    """自相关 F0 轨迹。返回 (times, f0, voiced_mask, energy)。"""
    n, h = int(win * sr), int(hop * sr)
    if len(x) < n:
        return None, None, None, None
    ts, f0s, vs, es = [], [], [], []
    lag_min, lag_max = int(sr / fmax), int(sr / fmin)
    for i in range(0, len(x) - n, h):
        seg = x[i:i + n]
        e = float(np.sqrt((seg ** 2).mean()))
        ts.append(i / sr)
        es.append(e)
        if e < 1e-4:
            f0s.append(np.nan)
            vs.append(False)
            continue
        z = seg - seg.mean()
        z = z * np.hanning(len(z))
        ac = np.correlate(z, z, "full")[len(z) - 1:]
        if ac[0] <= 0:
            f0s.append(np.nan)
            vs.append(False)
            continue
        ac = ac / ac[0]
        seg_ac = ac[lag_min:min(lag_max, len(ac))]
        if len(seg_ac) < 4:
            f0s.append(np.nan)
            vs.append(False)
            continue
        k = int(np.argmax(seg_ac))
        peak = seg_ac[k]
        # 清音判据：自相关峰值太低
        if peak < 0.3:
            f0s.append(np.nan)
            vs.append(False)
            continue
        lag = lag_min + k
        f0s.append(sr / lag if lag > 0 else np.nan)
        vs.append(True)
    return (np.array(ts), np.array(f0s), np.array(vs), np.array(es))


def to_semi(f0, ref=None):
    """转半音（相对 ref 或中位数）。"""
    v = f0[~np.isnan(f0)]
    if len(v) < 3:
        return f0 * np.nan
    base = ref if ref else float(np.median(v))
    return 12 * np.log2(np.maximum(f0, 1e-6) / base)


def tail_slope(f0, ts, tail_sec=0.3):
    """末段 F0 斜率（semitone/s）。"""
    v = ~np.isnan(f0)
    if v.sum() < 6:
        return float("nan")
    t_end = ts[v][-1]
    m = v & (ts >= t_end - tail_sec)
    if m.sum() < 4:
        return float("nan")
    s = to_semi(f0[m])
    ok = ~np.isnan(s)
    if ok.sum() < 4:
        return float("nan")
    return float(np.polyfit(ts[m][ok] - ts[m][ok][0], s[ok], 1)[0])


def voiced_span(x, sr=SR, t=-45.0):
    n = int(0.01 * sr)
    m = (len(x) // n) * n
    if m < n * 4:
        return 0, len(x)
    r = np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))
    ref = r.max() + 1e-12
    v = np.where(r > ref * (10 ** (t / 20)))[0]
    return (int(v[0]) * n, int(v[-1] + 1) * n) if len(v) else (0, len(x))


print("=" * 100)
print("韵律诊断：尾音音高走向")
print("=" * 100)
print("""
  假设：参考音频以**问句**结尾（「…好吗？」）-> 升调
        GPT-SoVITS 模仿其韵律风格 -> 所有句子都升调结尾
""")

# ---------- 1) 参考音频 ----------
xr = load16(REF, max_sec=REF_END)
tr, fr, vr, er = f0_track(xr)
print("=" * 100)
print("① 参考音频 ref/default.wav（文本以「好吗？」结尾 = 问句）")
print("=" * 100)
print(f"  时长 {len(xr)/SR:.2f}s")
if tr is not None and vr.sum() > 10:
    sem = to_semi(fr)
    # 逐段斜率
    print(f"  {'区间':<18s} {'F0中位':>8s} {'斜率(semi/s)':>13s}  走向")
    print("  " + "-" * 54)
    for a, b, nm in ((0, 1.0, "句首 0~1s"), (1.0, 3.0, "1~3s"),
                     (3.0, 5.0, "3~5s"),
                     (max(0, tr[-1] - 1.5), tr[-1], "**末 1.5s**"),
                     (max(0, tr[-1] - 0.5), tr[-1], "**末 0.5s**")):
        m = (tr >= a) & (tr <= b) & vr
        if m.sum() < 4:
            continue
        s = to_semi(fr[m])
        ok = ~np.isnan(s)
        if ok.sum() < 4:
            continue
        slope = np.polyfit(tr[m][ok] - tr[m][ok][0], s[ok], 1)[0]
        med = float(np.nanmedian(fr[m]))
        arrow = "**升** ↑" if slope > 0.5 else ("降 ↓" if slope < -0.5 else "平 →")
        print(f"  {nm:<18s} {med:7.1f}Hz {slope:+12.2f}  {arrow}")
    print()
    print(f"  末段整体斜率: {tail_slope(fr, tr, 0.5):+.2f} semi/s")

# ---------- 2) 模型输出 ----------
print()
print("=" * 100)
print("② 模型输出（10 句，全部是陈述句）")
print("=" * 100)
print(f"  {'句':<4s} {'时长':>6s} {'末0.5s斜率':>12s} {'末1.0s斜率':>12s}  走向")
print("  " + "-" * 58)
slopes_m = []
for w in sorted(MODEL_DIR.glob("*.wav")):
    x = load16(w)
    a, b = voiced_span(x)
    xv = x[a:b]
    t, f, v, e = f0_track(xv)
    if t is None or v.sum() < 8:
        continue
    s05 = tail_slope(f, t, 0.5)
    s10 = tail_slope(f, t, 1.0)
    slopes_m.append(s05)
    arrow = "**升** ↑" if s05 > 0.5 else ("降 ↓" if s05 < -0.5 else "平 →")
    print(f"  {w.stem:<4s} {len(xv)/SR:5.2f}s {s05:+12.2f} {s10:+12.2f}  {arrow}")
print()
print(f"  平均末 0.5s 斜率: **{np.nanmean(slopes_m):+.2f} semi/s**")

# ---------- 3) 真人陈述句 ----------
print()
print("=" * 100)
print("③ 真人录音（留出集，陈述句）—— 对照组")
print("=" * 100)
import json  # noqa: E402
slopes_h = []
if HELD.exists():
    hd = json.loads(HELD.read_text(encoding="utf-8"))
    for it in hd.get("all", [])[:40]:
        p = Path(it["path"])
        if not p.exists():
            continue
        x = load16(p)
        a, b = voiced_span(x)
        xv = x[a:b]
        if len(xv) / SR < 1.5:
            continue
        t, f, v, e = f0_track(xv)
        if t is None or v.sum() < 8:
            continue
        s = tail_slope(f, t, 0.5)
        if not np.isnan(s):
            slopes_h.append(s)
print(f"  样本 {len(slopes_h)} 条")
if slopes_h:
    arr = np.array(slopes_h)
    print(f"  平均末 0.5s 斜率: **{arr.mean():+.2f} semi/s**  "
          f"（中位 {np.median(arr):+.2f}）")
    print(f"  升调占比: {(arr > 0.5).mean()*100:.0f}%   "
          f"降调占比: {(arr < -0.5).mean()*100:.0f}%")

# ---------- 4) 对比结论 ----------
print()
print("=" * 100)
print("判读")
print("=" * 100)
mr = float(np.nanmean(slopes_m)) if slopes_m else float("nan")
hr = float(np.mean(slopes_h)) if slopes_h else float("nan")
print(f"  模型 末0.5s 斜率 : {mr:+.2f} semi/s")
print(f"  真人 末0.5s 斜率 : {hr:+.2f} semi/s")
print(f"  差              : {mr-hr:+.2f} semi/s")
print()
if mr > 0.5 and (np.isnan(hr) or hr < 0.3):
    print("  ✅ **假设成立！**")
    print("     · 模型**升调结尾**，真人是**降调/平调**")
    print("     · 这就是「尾音声音往上高」的直接原因")
    print()
    print("  ## 为什么「虚」")
    print("     升调时：")
    print("       · 声带拉紧、基频升高 -> 谐波结构变薄")
    print("       · 句末能量本应衰减，但为了维持升调 -> 能量分布异常")
    print("       · 听感 = 单薄、没底气 = 「虚」")
    print()
    print("  ## 根因")
    print("     `ref/default.wav` 的文本以**问句**结尾（「…好吗？」）")
    print("     GPT-SoVITS 模仿参考的韵律风格 -> 所有句子都升调")
    print()
    print("  ## 修复（不需要重训！）")
    print("     换一个**以陈述句结尾**的参考音频")
    print("     `pick_ref_from_train.py` 筛出的候选里，挑文本结尾是句号的")
elif mr < -0.5 and (np.isnan(hr) or hr > -0.3):
    print("  模型**降调**而真人升调 —— 与用户描述相反，需复查")
else:
    print("  两者走向接近 —— 升调不是主因，需另找")
    print(f"  （模型 {mr:+.2f}，真人 {hr:+.2f}）")

# ---------- 画图 ----------
fig, axes = plt.subplots(3, 1, figsize=(13, 10))
# 参考音频
if tr is not None:
    sem = to_semi(fr)
    axes[0].plot(tr, sem, ".", ms=2, color="tab:green")
    axes[0].axhline(0, color="gray", lw=0.6)
    axes[0].set_title("① 参考音频 ref/default.wav（结尾是「好吗？」= 问句）",
                      fontsize=10)
    axes[0].set_ylabel("半音（相对中位）", fontsize=9)
# 模型（叠加多条）
for w in sorted(MODEL_DIR.glob("*.wav"))[:6]:
    x = load16(w)
    a, b = voiced_span(x)
    t, f, v, e = f0_track(x[a:b])
    if t is None:
        continue
    axes[1].plot(t, to_semi(f), ".", ms=1.6, alpha=0.7,
                 label=w.stem)
axes[1].axhline(0, color="gray", lw=0.6)
axes[1].set_title("② 模型输出（陈述句，6 条叠加）—— 看末尾是否上翘",
                  fontsize=10)
axes[1].set_ylabel("半音", fontsize=9)
axes[1].legend(fontsize=7, ncol=6)
# 真人
if HELD.exists():
    for it in hd.get("all", [])[:6]:
        p = Path(it["path"])
        if not p.exists():
            continue
        x = load16(p)
        a, b = voiced_span(x)
        t, f, v, e = f0_track(x[a:b])
        if t is None:
            continue
        axes[2].plot(t, to_semi(f), ".", ms=1.6, alpha=0.7,
                     label=p.stem[:14])
axes[2].axhline(0, color="gray", lw=0.6)
axes[2].set_title("③ 真人录音（陈述句）—— 看末尾是否下压", fontsize=10)
axes[2].set_ylabel("半音", fontsize=9)
axes[2].set_xlabel("秒", fontsize=9)
axes[2].legend(fontsize=7, ncol=6)
for ax in axes:
    ax.grid(alpha=0.3)
    ax.set_ylim(-8, 8)
fig.tight_layout()
fig.savefig(OUT / "f0_compare.png", dpi=130)
plt.close(fig)
print()
print(f"  -> {OUT / 'f0_compare.png'}")

rep = ["=" * 80, "韵律诊断：尾音音高走向", "=" * 80, ""]
rep.append(f"模型 末0.5s 斜率: {mr:+.2f} semi/s")
rep.append(f"真人 末0.5s 斜率: {hr:+.2f} semi/s")
if slopes_h:
    arr = np.array(slopes_h)
    rep.append(f"真人升调占比 {(arr>0.5).mean()*100:.0f}%  "
               f"降调占比 {(arr<-0.5).mean()*100:.0f}%")
rep.append("")
rep.append("假设: ref/default.wav 以问句结尾 -> 模型模仿升调")
rep.append(f"结论: {'成立' if (mr>0.5 and (np.isnan(hr) or hr<0.3)) else '需复查'}")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
print(f"  -> {OUT / 'report.txt'}")
print("=" * 100)
