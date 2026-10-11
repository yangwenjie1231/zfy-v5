# -*- coding: utf-8 -*-
"""分析 nofreeze 40 个 wav —— 三组关键对比。

## 三组对比
  A) **e08 vs e32**        SoVITS 是否已饱和（决定还要不要训）
  B) **e32 vs v2Pro**      解冻后能否突破 0.79 平台期
  C) **e32 vs e16_merged** **解冻 vs 补底模值，哪个更好** ← 核心问题

## 为什么 C 最关键
之前的 e16_merged 是"用底模原值补 ssl_proj/enc_p"的应急方案。
现在是"训练时就让 enc_p/ssl_proj 参与"的真全量微调。
**C 直接回答"解冻值不值得"** —— 这是我们折腾这么多轮的核心。

## 物理指标（不用 GPU，秒出）
  · 时长 / 语速（正常 3~5 字/秒）
  · 谱质心（越低越闷）
  · 高频比（4k+/4k-，越高越亮）
  · 尾音 RMS（**先裁尾部静音再量**，否则被静音填充污染）
  · 时长比值（<0.6 = 崩了）

## 输出
    dist/nofreeze/compare_report.txt
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
NEW = V5 / "dist" / "nofreeze" / "out_sweep_nofreeze"
OLD_MERGED = V5 / "dist" / "sweep2" / "out_sweep" / "e16_merged"
V2PRO = V5 / "dist" / "baseline_v2pro"
COMMON_SR = 16000
COMMON_HZ = 7600.0

TEXTS = {
    "01": "让我想想。", "02": "好。",
    "03": "迎敌或是迎客，都得整装后再出发。",
    "04": "出门前，要不要先喝杯茶提提神？",
    "05": "嗯……我理一理。",
    "06": "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？",
    "07": "武陵城外有一片梨树林，每年春天都会开满白色的花。",
    "08": "她就在那阵花香里，只是他自己不知道。",
    "09": "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。",
    "10": "这件事我三点五十分再去处理。",
}


def to_common(x: np.ndarray, sr: int) -> np.ndarray:
    """统一重采样到 16kHz（**多相 FIR**，不是线性插值）。

    线性插值的频响是 sinc²(f/sr_out)，16kHz 处 -3.3 dB ——
    上一轮我用它做盲听包，v2Pro(32k) 被削高频而 v5(48k) 没有，
    用户听出"前段糊"，差点误判成模型差异。
    """
    if sr == COMMON_SR:
        return x
    try:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(int(sr), COMMON_SR)
        return resample_poly(x, COMMON_SR // g, int(sr) // g).astype(np.float64)
    except Exception:  # noqa: BLE001
        n = int(round(len(x) * COMMON_SR / sr))
        return np.interp(np.linspace(0, len(x) - 1, n),
                         np.arange(len(x)), x).astype(np.float64)


def load(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return x, int(sr)


def spectral16(x: np.ndarray):
    n, hop = 2048, 512
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    mag = np.abs(np.fft.rfft(fr * np.hanning(n), axis=1)).mean(axis=0)
    f = np.fft.rfftfreq(n, 1 / COMMON_SR)
    sel = f < COMMON_HZ
    m, ff = mag[sel], f[sel]
    tot = m.sum() + 1e-12
    centroid = float((ff * m).sum() / tot)
    return centroid, float(m[ff >= 4000].sum() / (m[ff < 4000].sum() + 1e-12))


def tail_rms(x: np.ndarray, sr: int, sec: float = 0.4) -> float:
    """尾音 RMS —— **先裁尾部静音再量**。

    ⚠️ 直接用 `x[-0.4s:]` 会被**末尾静音填充长度**污染。
    实测：v2Pro 静音尾 0.34s、v5merged 0.23s ->
    同一指标在不同窗长下比值从 0.0x 跳到 23.4x 再降到 1.1x。
    裁掉静音后：0.06634 vs 0.06671（只差 0.6%）。
    """
    nf = 256
    if len(x) < nf * 4:
        return float("nan")
    m = (len(x) // nf) * nf
    rmsf = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
    ref = float(np.sqrt((x ** 2).mean())) + 1e-12
    voiced = np.where(rmsf > ref * (10 ** (-45.0 / 20)))[0]
    if len(voiced) == 0:
        return float("nan")
    end = int((voiced[-1] + 1) * nf)
    n = int(sec * sr)
    seg = x[max(0, end - n):end]
    return float(np.sqrt((seg ** 2).mean())) if len(seg) >= n // 2 else float("nan")


def collect(d: Path, pat: str = "*.wav"):
    """返回 {idx: {dur, speed, cent, hf, tail, sr}}。"""
    out = {}
    if not d.is_dir():
        return out
    for w in sorted(d.glob(pat)):
        m = re.match(r"(\d+)_", w.stem)
        if not m:
            continue
        idx = m.group(1)
        text = TEXTS.get(idx)
        if text is None:
            continue
        try:
            x, sr = load(w)
        except Exception:  # noqa: BLE001
            continue
        if len(x) < sr * 0.2:
            continue
        dur = len(x) / sr
        xc = to_common(x, sr)
        c, h = spectral16(xc)
        out[idx] = {
            "dur": dur, "sr": sr,
            "speed": len(re.sub(r"[，。？！、…,.?!\s]", "", text)) / dur,
            "cent": c, "hf": h, "tail": tail_rms(x, sr),
            "file": w.name,
        }
    return out


def summ(d: dict) -> dict:
    if not d:
        return {}
    ks = list(d.values())
    return {
        "n": len(ks),
        "dur": float(np.mean([k["dur"] for k in ks])),
        "speed": float(np.mean([k["speed"] for k in ks])),
        "cent": float(np.mean([k["cent"] for k in ks])),
        "hf": float(np.mean([k["hf"] for k in ks])),
        "tail": float(np.nanmean([k["tail"] for k in ks])),
    }


print("=" * 96)
print("nofreeze 分析（e08 / e16 / e24 / e32 + 对照）")
print("=" * 96)

data = {}
for e in ("e08", "e16", "e24", "e32"):
    data[e] = collect(NEW / e)
    print(f"  {e:6s} {len(data[e]):2d} 句  "
          f"{NEW / e if (NEW/e).is_dir() else '**目录不存在**'}")

data["e16_merged"] = collect(OLD_MERGED)
data["v2pro"] = collect(V2PRO)
print(f"  {'e16m':6s} {len(data['e16_merged']):2d} 句  （旧：补底模值）")
print(f"  {'v2pro':6s} {len(data['v2pro']):2d} 句  （生产对照）")

# ---------- 逐句时长表 ----------
print()
print("=" * 96)
print("逐句时长（秒）")
print("=" * 96)
hdr = f"  {'句':4s} {'字数':>4s}"
for k in ("e08", "e16", "e24", "e32", "e16m", "v2pro"):
    hdr += f" {k:>7s}"
print(hdr)
print("  " + "-" * 74)
for idx in sorted(TEXTS):
    n = len(re.sub(r"[，。？！、…,.?!\s]", "", TEXTS[idx]))
    row = f"  {idx:4s} {n:4d}"
    for k in ("e08", "e16", "e24", "e32", "e16_merged", "v2pro"):
        v = data[k].get(idx)
        row += f" {v['dur']:7.2f}" if v else f" {'-':>7s}"
    print(row)

# ---------- 汇总表 ----------
print()
print("=" * 96)
print("汇总")
print("=" * 96)
print(f"  {'模型':10s} {'句数':>4s} {'平均时长':>9s} {'语速':>6s} "
      f"{'谱质心':>8s} {'高频比':>8s} {'尾音RMS':>9s}")
print("  " + "-" * 68)
sums = {}
for k, label in (("e08", "e08"), ("e16", "e16"), ("e24", "e24"),
                 ("e32", "**e32**"), ("e16_merged", "e16_merged"),
                 ("v2pro", "v2pro")):
    s = summ(data[k])
    sums[k] = s
    if not s:
        print(f"  {label:10s} {'无数据':>4s}")
        continue
    print(f"  {label:10s} {s['n']:4d} {s['dur']:8.2f}s {s['speed']:6.2f} "
          f"{s['cent']:8.0f} {s['hf']:8.4f} {s['tail']:9.5f}")

# ---------- 三组对比 ----------
print()
print("=" * 96)
print("A) e08 vs e32 —— SoVITS 是否已饱和")
print("=" * 96)
if sums.get("e08") and sums.get("e32"):
    a, b = sums["e08"], sums["e32"]
    for key, name, fmt, better in (
        ("dur", "平均时长", "{:.3f}", "接近"),
        ("speed", "语速", "{:.3f}", "接近 3~5"),
        ("cent", "谱质心", "{:.1f}", "越高越亮"),
        ("hf", "高频比", "{:.4f}", "越高越亮"),
        ("tail", "尾音RMS", "{:.5f}", "越高越有力"),
    ):
        d = b[key] - a[key]
        rel = d / a[key] * 100 if a[key] else 0
        print(f"  {name:10s} e08={fmt.format(a[key]):>10s}  "
              f"e32={fmt.format(b[key]):>10s}  Δ={d:+.4f} ({rel:+.1f}%)")
    print()
    print("  判读：")
    for key, name in (("cent", "谱质心"), ("hf", "高频比"), ("tail", "尾音")):
        rel = (b[key] - a[key]) / a[key] * 100 if a[key] else 0
        tag = "**仍在小幅提升**" if abs(rel) > 3 else "**已饱和**（差异 <3%）"
        print(f"    {name:8s} 变化 {rel:+.1f}%  -> {tag}")

print()
print("=" * 96)
print("B) e32 vs v2Pro（生产）")
print("=" * 96)
if sums.get("e32") and sums.get("v2pro"):
    a, b = sums["v2pro"], sums["e32"]
    for key, name, fmt in (("dur", "平均时长", "{:.3f}"),
                           ("speed", "语速", "{:.3f}"),
                           ("cent", "谱质心", "{:.1f}"),
                           ("hf", "高频比", "{:.4f}"),
                           ("tail", "尾音RMS", "{:.5f}")):
        print(f"  {name:10s} v2pro={fmt.format(a[key]):>10s}  "
              f"e32={fmt.format(b[key]):>10s}  Δ={b[key]-a[key]:+.4f}")

print()
print("=" * 96)
print("C) e32 vs e16_merged —— **解冻 vs 补底模值（核心问题）**")
print("=" * 96)
if sums.get("e32") and sums.get("e16_merged"):
    a, b = sums["e16_merged"], sums["e32"]
    for key, name, fmt in (("dur", "平均时长", "{:.3f}"),
                           ("speed", "语速", "{:.3f}"),
                           ("cent", "谱质心", "{:.1f}"),
                           ("hf", "高频比", "{:.4f}"),
                           ("tail", "尾音RMS", "{:.5f}")):
        d = b[key] - a[key]
        rel = d / a[key] * 100 if a[key] else 0
        print(f"  {name:10s} e16m={fmt.format(a[key]):>10s}  "
              f"e32={fmt.format(b[key]):>10s}  Δ={d:+.4f} ({rel:+.1f}%)")
    print()
    print("  判读：")
    dh = (b["hf"] - a["hf"]) / a["hf"] * 100 if a["hf"] else 0
    dc = (b["cent"] - a["cent"]) / a["cent"] * 100 if a["cent"] else 0
    if dh > 3 or dc > 3:
        print(f"    ✓ **解冻有效** —— 谱质心 {dc:+.1f}%、高频比 {dh:+.1f}%")
        print("      说明训练时让 enc_p/ssl_proj 参与，比事后补底模值更好")
    elif dh < -3 or dc < -3:
        print(f"    ✗ **解冻反而更闷** —— 谱质心 {dc:+.1f}%、高频比 {dh:+.1f}%")
        print("      说明底模原值比训练后的更好（可能过拟合）")
    else:
        print(f"    ~ **无明显差异**（谱质心 {dc:+.1f}%、高频比 {dh:+.1f}%）")
        print("      说明 enc_p/ssl_proj 不是瓶颈 -> 该维度触顶")

# ---------- 崩坏检测 ----------
print()
print("=" * 96)
print("崩坏检测（时长比值 < 0.6 = 提前终止）")
print("=" * 96)
bad_any = False
for k in ("e08", "e16", "e24", "e32", "e16_merged", "v2pro"):
    bad = []
    for idx, v in data[k].items():
        n = len(re.sub(r"[，。？！、…,.?!\s]", "", TEXTS[idx]))
        ratio = v["dur"] / (n / 4.0)      # 期望 4 字/秒
        if ratio < 0.6 or v["speed"] > 7:
            bad.append(f"{idx}({v['dur']:.2f}s,{v['speed']:.1f}字/秒)")
    if bad:
        bad_any = True
        print(f"  !! {k:10s} {len(bad)} 句异常: {', '.join(bad[:5])}")
if not bad_any:
    print("  ✓ 全部正常（无语速异常/提前终止）")

# ---------- 写报告 ----------
rep = ["=" * 78, "nofreeze 客观对比报告", "=" * 78, ""]
rep.append("三组对比：")
rep.append("  A) e08 vs e32       SoVITS 是否饱和")
rep.append("  B) e32 vs v2Pro     能否突破 0.79 平台期")
rep.append("  C) e32 vs e16_merged  解冻 vs 补底模值（核心）")
rep.append("")
rep.append(f"{'模型':12s}{'句数':>5s}{'平均时长':>10s}{'语速':>8s}"
           f"{'谱质心':>9s}{'高频比':>9s}{'尾音RMS':>10s}")
rep.append("-" * 78)
for k in ("e08", "e16", "e24", "e32", "e16_merged", "v2pro"):
    s = sums.get(k)
    if not s:
        continue
    rep.append(f"{k:12s}{s['n']:5d}{s['dur']:9.2f}s{s['speed']:8.2f}"
               f"{s['cent']:9.0f}{s['hf']:9.4f}{s['tail']:10.5f}")
rep.append("")
rep.append("判读：")
rep.append("  · 谱质心/高频比 越高越亮（越低越闷）")
rep.append("  · 语速正常 3~5 字/秒")
rep.append("  · 尾音 RMS 已裁掉尾部静音后再量")
rep.append("  · 相似度需另跑 eval_from_wav.py（要 GPU + 留出集）")
out = V5 / "dist" / "nofreeze" / "compare_report.txt"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(rep) + "\n", encoding="utf-8")
print()
print(f"  -> {out}")
print("=" * 96)
