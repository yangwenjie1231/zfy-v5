# -*- coding: utf-8 -*-
"""决定性对比：v2Pro 输出的尾部 vs 真实录音 vs 参考音频 vs 我的处理。

## 前面的假设被推翻
裁掉参考音频末尾那段（0.34s）后：
    尾部低频 57.5% -> 58.7%   （+1.2pp，无变化）
-> **"参考音频多一段"不是"尾音气声"的原因**

## 那还剩什么可能
    A. **v2Pro 模型本身的输出特性**（生产也有，用户以前没注意）
    B. **我的听包处理引入的**（to48 重采样 / RMS 对齐 / 拼接）
    C. 真实录音本来就这样（那就是数据特性，不是 bug）

## 本脚本：同口径测四种来源
    1. `dist/gpt_ab/base/NN.wav`      v2Pro 原始输出（32kHz，未处理）
    2. 同样音频经 to48 + align_rms    我的处理（48kHz）
    3. `dist/heldout` 真实录音         真人（留出集）
    4. `ref/default.wav`               参考音频

**判据**：
    · 若 1 ≈ 2  -> 不是我的处理
    · 若 1 ≈ 3  -> 真实录音也这样，是数据特性
    · 若 1 明显比 3 "低频重" -> **模型在句末加了低频能量**（真问题）
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
RAW = V5 / "dist" / "gpt_ab" / "base"
HELD = V5 / "dist" / "heldout" / "heldout_clean.json"
REF = V5 / "GPT-SoVITS" / "ref" / "default.wav"
BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000),
         (4000, 8000), (8000, 16000)]


def load(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return x, int(sr)


def to48(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == 48000:
        return x
    try:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(int(sr), 48000)
        return resample_poly(x, 48000 // g, int(sr) // g).astype(np.float64)
    except Exception:  # noqa: BLE001
        n = int(round(len(x) * 48000 / sr))
        return np.interp(np.linspace(0, len(x) - 1, n),
                         np.arange(len(x)), x).astype(np.float64)


def align_rms(x: np.ndarray, target: float = 0.06) -> np.ndarray:
    r = float(np.sqrt((x ** 2).mean()))
    if r < 1e-9:
        return x
    y = x * (target / r)
    pk = float(np.abs(y).max())
    return y * (0.99 / pk) if pk > 0.99 else y


def band_dist(s: np.ndarray, sr: int) -> dict:
    n = min(len(s), 4096)
    if n < 64:
        return {}
    S = np.abs(np.fft.rfft(s[:n] * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1 / sr)
    S = S / (S.sum() + 1e-12)
    return {f"{lo}-{hi}": float(S[(f >= lo) & (f < min(hi, sr / 2))].sum())
            for lo, hi in BANDS}


def tail_metrics(x: np.ndarray, sr: int, sec: float = 0.4) -> dict:
    """尾部（裁掉静音后的最后 sec 秒）的指标。"""
    if len(x) < sr * 0.3:
        return {}
    nf = 256
    m = (len(x) // nf) * nf
    rmsf = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
    ref = float(np.sqrt((x ** 2).mean())) + 1e-12
    voiced = np.where(rmsf > ref * (10 ** (-45 / 20)))[0]
    if len(voiced) == 0:
        return {}
    end = int((voiced[-1] + 1) * nf)
    n = int(sec * sr)
    tail = x[max(0, end - n):end]
    body = x[max(0, end // 2 - sr // 2):end // 2 + sr // 2]
    bd = band_dist(tail, sr)
    lo = sum(v for k, v in bd.items() if int(k.split("-")[1]) <= 2000)
    hi = sum(v for k, v in bd.items() if int(k.split("-")[0]) >= 4000)
    tr = float(np.sqrt((tail ** 2).mean())) if len(tail) else 0.0
    br = float(np.sqrt((body ** 2).mean())) + 1e-12
    # 谱平坦度
    nn = min(len(tail), 2048)
    S = np.abs(np.fft.rfft(tail[:nn] * np.hanning(nn))) + 1e-12
    flat = float(np.exp(np.log(S).mean()) / S.mean()) if nn >= 64 else float("nan")
    return {"lo": lo, "hi": hi, "tb_db": 20 * np.log10(tr / br + 1e-12),
            "flat": flat, "dur": len(x) / sr,
            "sil_tail": (len(x) - end) / sr}


print("=" * 100)
print("尾部特征 · 同口径四方对比")
print("=" * 100)

groups: dict[str, list] = {}

# 1) v2Pro 原始输出
for w in sorted(RAW.glob("*.wav"))[:20]:
    x, sr = load(w)
    t = tail_metrics(x, sr)
    if t:
        t["_sr"] = sr
        groups.setdefault(f"v2Pro 原始输出 ({sr//1000}kHz)", []).append(t)

# 2) 我的处理（同音频 to48 + align_rms）
for w in sorted(RAW.glob("*.wav"))[:20]:
    x, sr = load(w)
    y = align_rms(to48(x, sr))
    t = tail_metrics(y, 48000)
    if t:
        t["_sr"] = 48000
        groups.setdefault("我的处理 (48kHz+RMS对齐)", []).append(t)

# 3) 真实录音（留出集）
if HELD.exists():
    hd = json.loads(HELD.read_text(encoding="utf-8"))
    for f in hd.get("all", [])[:30]:
        p = Path(f["path"])
        if not p.exists():
            continue
        x, sr = load(p)
        t = tail_metrics(x, sr)
        if t:
            t["_sr"] = sr
            groups.setdefault("真实录音（留出集）", []).append(t)

# 4) 参考音频
if REF.exists():
    x, sr = load(REF)
    t = tail_metrics(x, sr)
    if t:
        t["_sr"] = sr
        groups.setdefault("参考音频 ref/default", []).append(t)

print(f"  {'来源':<30s} {'n':>4s} {'尾部低频':>9s} {'尾部高频':>9s} "
      f"{'尾/体dB':>8s} {'尾平坦':>8s} {'静音尾':>8s}")
print("  " + "-" * 88)
for k, v in groups.items():
    lo = np.mean([x["lo"] for x in v])
    hi = np.mean([x["hi"] for x in v])
    tb = np.mean([x["tb_db"] for x in v])
    fl = np.nanmean([x["flat"] for x in v])
    st = np.mean([x["sil_tail"] for x in v])
    print(f"  {k:<30s} {len(v):4d} {lo*100:8.1f}% {hi*100:8.1f}% "
          f"{tb:+8.2f} {fl:8.4f} {st:7.2f}s")

print()
print("=" * 100)
print("判读")
print("=" * 100)
keys = list(groups)
g = {k: groups[k] for k in keys}


def m(k, f):
    xs = [x[f] for x in g[k] if x.get(f) is not None]
    return float(np.nanmean(xs)) if xs else float("nan")


raw_k = next((k for k in keys if "原始输出" in k), None)
proc_k = next((k for k in keys if "我的处理" in k), None)
real_k = next((k for k in keys if "真实录音" in k), None)
ref_k = next((k for k in keys if "参考音频" in k), None)

if raw_k and proc_k:
    d = (m(proc_k, "lo") - m(raw_k, "lo")) * 100
    print(f"  ① v2Pro原始 vs 我的处理:  尾部低频 {d:+.1f}pp")
    print(f"     -> {'**我的处理改变了尾部**' if abs(d) > 3 else '✓ 我的处理无影响'}")
if raw_k and real_k:
    d = (m(raw_k, "lo") - m(real_k, "lo")) * 100
    print(f"  ② v2Pro原始 vs 真实录音:  尾部低频 {d:+.1f}pp")
    if d > 8:
        print("     -> **模型句末低频明显多于真人** = 真问题")
    elif d < -8:
        print("     -> 模型句末低频**少于**真人（反而更清）")
    else:
        print("     -> 与真人接近，**属正常范围**")
if raw_k and ref_k:
    d = (m(raw_k, "lo") - m(ref_k, "lo")) * 100
    print(f"  ③ v2Pro原始 vs 参考音频:  尾部低频 {d:+.1f}pp")
    print(f"     -> {'参考音频尾部更低沉' if d < 0 else '参考音频尾部更高'}")

print()
print("""
  ## 关键：如果 ① 无影响、② 与真人接近
  那"尾音气声"就是**真实语音的正常特征**，只是：
    · 三方对比包把 3 段拼在一起 -> **尾音出现 3 次**，更显眼
    · 用户以前听单句时不会注意

  ## 如果 ② 显示模型低频明显更多
  那是 v2Pro 的模型特性（生产也有），需要：
    · 换参考音频（挑尾音干净的）
    · 或做后处理（2~8kHz 轻提，抵消低频感）
""")
print("=" * 100)