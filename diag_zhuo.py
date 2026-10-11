# -*- coding: utf-8 -*-
"""诊断「尾音浊」—— 重新定义指标（用户的描述比"气声"精确得多）。

## 用户的澄清
    「尾音说得比较浊」= **发闷、发糊、字咬不清**

这与「气声」几乎相反：
    气声 = 高频噪声多、发虚发飘
    浊   = **低频重、高频少、发闷发糊**

## 我之前的错误
`postprocess_tail.py` 做的是 **高频衰减（lowpass）**：
    tilt-db -10  ->  6kHz 以上衰减 10dB
**这正好把声音弄得更浊！** 方向完全反了。

## 「浊」的声学指标（本脚本）
    ① **谱质心**（spectral centroid）
       越低越闷/浊。这是"明亮度"的直接度量。
    ② **谱倾斜**（spectral tilt, dB/octave）
       负得越多越暗/浊。用 log 幅度对 log 频率做线性回归。
    ③ **高频占比** HF(4~16k) / 总能量
       越低越浊。
    ④ **共振峰清晰度**（formant prominence）
       0~4kHz 内 峰值/谷值 比。**越低 = 共振峰糊 = 咬字不清**。
       这是"浊"最贴切的度量 —— 元音靠共振峰区分，糊了就听不清。
    ⑤ **逐频段噪声性**（band-wise flatness）
       找出"噪声集中在哪个频段"：
         低频噪声 -> 浊（闷）
         高频噪声 -> 气声（虚）

## 对比
    · 模型输出尾部
    · **真人录音尾部**（留出集）
    · 参考音频尾部

## 输出
    dist/diag_zhuo.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
RAW = V5 / "dist" / "ref_swap_train" / "orig"
HELD = V5 / "dist" / "heldout" / "heldout_clean.json"
REF = V5 / "GPT-SoVITS" / "ref" / "default.wav"

BANDS = [(0, 250), (250, 500), (500, 1000), (1000, 2000),
         (2000, 4000), (4000, 6000), (6000, 8000), (8000, 12000),
         (12000, 16000)]


def load(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return x, int(sr)


def voiced_end(x: np.ndarray, sr: int, thresh_db: float = -45.0) -> int:
    nf = 256
    m = (len(x) // nf) * nf
    if m < nf * 4:
        return len(x)
    rmsf = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
    r = float(np.sqrt((x ** 2).mean())) + 1e-12
    v = np.where(rmsf > r * (10 ** (thresh_db / 20)))[0]
    return int((v[-1] + 1) * nf) if len(v) else len(x)


def analyze_seg(s: np.ndarray, sr: int) -> dict:
    """对一段音频算「浊度」指标。"""
    if len(s) < 256:
        return {}
    n = min(len(s), 4096)
    S = np.abs(np.fft.rfft(s[:n] * np.hanning(n))) + 1e-12
    f = np.fft.rfftfreq(n, 1 / sr)
    P = S ** 2
    Pn = P / (P.sum() + 1e-18)

    # ① 谱质心
    centroid = float((f * Pn).sum())
    # ② 谱倾斜（dB/octave）：log 幅度 对 log 频率回归
    sel = (f > 80) & (f < min(8000, sr / 2))
    if sel.sum() > 8:
        lf = np.log10(f[sel])
        lm = 20 * np.log10(S[sel])
        tilt = float(np.polyfit(lf, lm, 1)[0])   # dB per decade
        tilt_oct = tilt * np.log10(2)            # dB per octave
    else:
        tilt_oct = float("nan")
    # ③ 高频占比
    hf = float(Pn[(f >= 4000) & (f < min(16000, sr / 2))].sum())
    lf_ = float(Pn[f < 2000].sum())
    # ④ 共振峰清晰度：0~4kHz 内 峰值/均值
    fs_ = (f >= 200) & (f < min(4000, sr / 2))
    if fs_.sum() > 8:
        sp = S[fs_]
        # 平滑后取峰谷比
        k = max(3, len(sp) // 40)
        ker = np.ones(k) / k
        sm = np.convolve(sp, ker, "same")
        prom = float(sm.max() / (sm.mean() + 1e-12))
    else:
        prom = float("nan")
    # ⑤ 逐频段平坦度（找噪声在哪个频段）
    flats = {}
    for lo, hi in BANDS:
        if lo >= sr / 2:
            continue
        m_ = (f >= lo) & (f < min(hi, sr / 2))
        if m_.sum() < 4:
            continue
        seg = S[m_]
        flats[f"{lo}-{hi}"] = float(np.exp(np.log(seg).mean()) / seg.mean())
    return {"centroid": centroid, "tilt_oct": tilt_oct, "hf": hf, "lf": lf_,
            "prom": prom, "flats": flats}


def tail_body(p: Path, tail_sec: float = 0.35):
    x, sr = load(p)
    if len(x) < sr * 0.3:
        return None, None, sr
    end = voiced_end(x, sr)
    n = int(tail_sec * sr)
    tail = x[max(0, end - n):end]
    mid = end // 2
    body = x[max(0, mid - sr // 3):mid + sr // 3]
    return tail, body, sr


print("=" * 100)
print("「尾音浊」诊断")
print("=" * 100)
print("""
  指标含义：
    谱质心    越低越闷/浊（明亮度）
    谱倾斜    负得越多越暗（dB/octave）
    高频占比  4~16kHz 能量占比，越低越浊
    共振峰清晰度  0~4kHz 峰值/均值，**越低 = 咬字越糊**（最贴切"浊"）
    逐频段平坦度  噪声集中在哪个频段
""")

groups: dict[str, list] = {}

# 模型输出
for w in sorted(RAW.glob("*.wav"))[:10]:
    t, b, sr = tail_body(w)
    if t is None:
        continue
    at, ab = analyze_seg(t, sr), analyze_seg(b, sr)
    if at and ab:
        groups.setdefault("模型输出(生产配置)", []).append(
            {"tail": at, "body": ab, "sr": sr})

# 真人录音
if HELD.exists():
    hd = json.loads(HELD.read_text(encoding="utf-8"))
    for it in hd.get("all", [])[:40]:
        p = Path(it["path"])
        if not p.exists():
            continue
        t, b, sr = tail_body(p)
        if t is None:
            continue
        at, ab = analyze_seg(t, sr), analyze_seg(b, sr)
        if at and ab:
            groups.setdefault("真人录音(留出集)", []).append(
                {"tail": at, "body": ab, "sr": sr})

# 参考音频
if REF.exists():
    t, b, sr = tail_body(REF)
    if t is not None:
        at, ab = analyze_seg(t, sr), analyze_seg(b, sr)
        if at and ab:
            groups.setdefault("参考音频 ref/default", []).append(
                {"tail": at, "body": ab, "sr": sr})


def m(rows, where, key):
    xs = [r[where][key] for r in rows if r[where].get(key) is not None
          and not (isinstance(r[where][key], float)
                   and np.isnan(r[where][key]))]
    return float(np.mean(xs)) if xs else float("nan")


print("=" * 100)
print("尾部指标对比（tail_sec = 0.35s）")
print("=" * 100)
print(f"  {'来源':<26s} {'n':>4s} {'谱质心':>8s} {'谱倾斜':>9s} "
      f"{'高频占比':>9s} {'共振峰清晰':>11s}")
print("  " + "-" * 78)
for k, v in groups.items():
    print(f"  {k:<26s} {len(v):4d} {m(v,'tail','centroid'):8.0f} "
          f"{m(v,'tail','tilt_oct'):+9.2f} {m(v,'tail','hf'):9.4f} "
          f"{m(v,'tail','prom'):11.3f}")

print()
print("=" * 100)
print("主体 vs 尾部（同一来源内对比，看尾部是否更浊）")
print("=" * 100)
print(f"  {'来源':<26s} {'谱质心(体->尾)':>18s} {'高频(体->尾)':>16s} "
      f"{'共振峰(体->尾)':>18s}")
print("  " + "-" * 84)
for k, v in groups.items():
    cb, ct = m(v, "body", "centroid"), m(v, "tail", "centroid")
    hb, ht = m(v, "body", "hf"), m(v, "tail", "hf")
    pb, pt = m(v, "body", "prom"), m(v, "tail", "prom")
    print(f"  {k:<26s} {cb:7.0f} -> {ct:7.0f} {hb:7.4f} -> {ht:7.4f} "
          f"{pb:8.3f} -> {pt:8.3f}")

print()
print("=" * 100)
print("逐频段平坦度（找噪声在哪 —— 低频噪声=浊，高频噪声=气声）")
print("=" * 100)
bands = sorted({b for v in groups.values() for r in v
                for b in r["tail"]["flats"]},
               key=lambda s: int(s.split("-")[0]))
hdr = f"  {'来源':<26s}" + "".join(f"{b.split('-')[0]:>7s}" for b in bands)
print(hdr)
print("  " + "-" * (26 + 7 * len(bands)))
for k, v in groups.items():
    row = f"  {k:<26s}"
    for b in bands:
        xs = [r["tail"]["flats"][b] for r in v if b in r["tail"]["flats"]]
        row += f"{np.mean(xs):7.3f}" if xs else f"{'—':>7s}"
    print(row)
print(f"  （列标题为频段下限 Hz）")

print()
print("=" * 100)
print("判读")
print("=" * 100)
mod = groups.get("模型输出(生产配置)", [])
real = groups.get("真人录音(留出集)", [])
if mod and real:
    ct_m, ct_r = m(mod, "tail", "centroid"), m(real, "tail", "centroid")
    hf_m, hf_r = m(mod, "tail", "hf"), m(real, "tail", "hf")
    pr_m, pr_r = m(mod, "tail", "prom"), m(real, "tail", "prom")
    ti_m, ti_r = m(mod, "tail", "tilt_oct"), m(real, "tail", "tilt_oct")
    print(f"  模型 vs 真人（尾部）:")
    print(f"    谱质心    {ct_m:.0f} vs {ct_r:.0f} Hz   "
          f"{'✓ 模型更亮' if ct_m > ct_r*1.02 else ('**✗ 模型更闷（浊）**' if ct_m < ct_r*0.98 else '~ 接近')}")
    print(f"    高频占比  {hf_m:.4f} vs {hf_r:.4f}   "
          f"{'✓ 模型更高' if hf_m > hf_r*1.05 else ('**✗ 模型更低（浊）**' if hf_m < hf_r*0.95 else '~ 接近')}")
    print(f"    共振峰清晰 {pr_m:.3f} vs {pr_r:.3f}   "
          f"{'✓ 模型更清晰' if pr_m > pr_r*1.02 else ('**✗ 模型更糊（咬字不清）**' if pr_m < pr_r*0.98 else '~ 接近')}")
    print(f"    谱倾斜    {ti_m:+.2f} vs {ti_r:+.2f} dB/oct   "
          f"{'✓ 模型更亮' if ti_m > ti_r+0.5 else ('**✗ 模型更暗（浊）**' if ti_m < ti_r-0.5 else '~ 接近')}")
    print()
    # 主因判断
    issues = []
    if ct_m < ct_r * 0.98:
        issues.append("谱质心偏低（闷）")
    if hf_m < hf_r * 0.95:
        issues.append("高频不足（暗）")
    if pr_m < pr_r * 0.98:
        issues.append("**共振峰糊（咬字不清）**")
    if ti_m < ti_r - 0.5:
        issues.append("谱倾斜偏暗")
    if issues:
        print(f"  确认的问题: {', '.join(issues)}")
        print()
        print("  ## 修复方向（**与之前相反！**）")
        print("    之前我做了**高频衰减**（lowpass tilt -10dB）—— 那是把声音弄得更浊。")
        print("    正确方向应该是 **尾部高频提升（high-shelf boost）**：")
        print("      · 提升 3~8kHz，+3~6dB")
        print("      · 或者做 **尾部去噪**（浊 = 低频噪声，用谱减）")
        print("      · 或者 **尾部谐波增强**（提高共振峰清晰度）")
    else:
        print("  尾部指标与真人接近 —— 可能是别的因素（如辅音缺失）")
else:
    print("  数据不足")

print()
print("=" * 100)
out = V5 / "dist" / "diag_zhuo.txt"
lines = ["=" * 78, "「尾音浊」诊断", "=" * 78, ""]
lines.append(f"{'来源':<28s}{'谱质心':>10s}{'谱倾斜':>10s}{'高频':>10s}{'共振峰':>10s}")
for k, v in groups.items():
    lines.append(f"{k:<28s}{m(v,'tail','centroid'):10.0f}"
                 f"{m(v,'tail','tilt_oct'):+10.2f}{m(v,'tail','hf'):10.4f}"
                 f"{m(v,'tail','prom'):10.3f}")
lines.append("")
lines.append("逐频段平坦度（尾部）:")
for k, v in groups.items():
    row = f"{k:<28s}"
    for b in bands:
        xs = [r["tail"]["flats"][b] for r in v if b in r["tail"]["flats"]]
        row += f"{np.mean(xs):8.3f}" if xs else f"{'—':>8s}"
    lines.append(row)
lines.append("")
lines.append("频段: " + ", ".join(bands))
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"  -> {out}")
print("=" * 100)
