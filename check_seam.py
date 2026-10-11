# -*- coding: utf-8 -*-
"""接缝检查的**对照实验** —— 排除"测量假象"。

## 疑点
`verify_zhuofix.py` 报"边界 RMS 跳变 最大 42 dB"。
但**边界点 `e - 1.0s` 可能落在句中的静音处** ——
那样 RMS 从"静音"跳到"语音"本来就是几十 dB，**与我的处理无关**。

## 判据
对**同一位置**同时测：
    ① 原版在该边界的跳变
    ② v2 在该边界的跳变
若 ① ≈ ② -> **跳变来自音频本身（静音/语音交界），不是处理引入**
若 ② >> ① -> 确实是处理接缝

## 更直接的检查：处理区边界处的**样本级不连续**
    · 一阶差分在边界处的突变（|x[n]-x[n-1]| 的局部峰值）
    · 与相邻区域的差分分布比较（z-score）
    · **谱通量**在边界处是否异常

## 输出
    dist/seam_check/report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
ORIG = V5 / "dist" / "ref_swap_train" / "orig"
V1 = V5 / "dist" / "zhuo_sweep" / "best"
V2 = V5 / "dist" / "zhuo_sweep2" / "best"
OUT = V5 / "dist" / "seam_check"
OUT.mkdir(parents=True, exist_ok=True)
TAIL_SEC = 1.0


def load(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return np.asarray(x, dtype=np.float64), int(sr)


def voiced_end(x, sr, t=-45.0):
    nf = 256
    m = (len(x) // nf) * nf
    if m < nf * 4:
        return len(x)
    r = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
    ref = float(np.sqrt((x ** 2).mean())) + 1e-12
    v = np.where(r > ref * (10 ** (t / 20)))[0]
    return int((v[-1] + 1) * nf) if len(v) else len(x)


def rms_at(x, c, half):
    a, b = max(0, c - half), min(len(x), c + half)
    seg = x[a:b]
    return float(np.sqrt((seg ** 2).mean())) if len(seg) > 8 else 0.0


print("=" * 96)
print("接缝检查（带原版对照）")
print("=" * 96)

files = sorted(ORIG.glob("*.wav"))
res = {"orig": [], "v1": [], "v2": []}
diff_res = {"orig": [], "v1": [], "v2": []}

for w in files:
    xo, sr = load(w)
    e = voiced_end(xo, sr)
    a = e - int(TAIL_SEC * sr)
    if a < int(0.15 * sr):
        continue
    w50 = int(0.05 * sr)

    for tag, d in (("orig", ORIG), ("v1", V1), ("v2", V2)):
        p = d / w.name
        if not p.exists():
            continue
        y = load(p)[0]
        if len(y) < e:
            continue
        # ① 边界 RMS 跳变（50ms 窗）
        pre = rms_at(y, a - w50, w50)
        post = rms_at(y, a + w50, w50)
        if pre > 1e-7 and post > 1e-7:
            res[tag].append(20 * np.log10(post / pre))
        # ② 一阶差分在边界的 z-score
        d1 = np.abs(np.diff(y))
        lo, hi = max(1, a - 20), min(len(d1) - 1, a + 20)
        if hi > lo:
            local = d1[lo:hi].max()
            # 参考分布：整个处理区之外的差分
            ref_region = np.concatenate([d1[:max(1, a - 200)],
                                         d1[min(len(d1) - 1, e + 200):]])
            if len(ref_region) > 100:
                mu, sd = ref_region.mean(), ref_region.std() + 1e-12
                diff_res[tag].append(float((local - mu) / sd))

print()
print(f"  {'版本':<8s} {'n':>4s} {'RMS跳变均值':>12s} {'RMS跳变最大':>12s} "
      f"{'差分z均值':>10s} {'差分z最大':>10s}")
print("  " + "-" * 62)
for tag in ("orig", "v1", "v2"):
    if not res[tag]:
        continue
    r = np.array(res[tag])
    z = np.array(diff_res[tag]) if diff_res[tag] else np.array([np.nan])
    print(f"  {tag:<8s} {len(r):4d} {r.mean():+11.2f}dB "
          f"{np.abs(r).max():11.2f}dB {np.nanmean(z):10.2f} "
          f"{np.nanmax(np.abs(z)):10.2f}")

print()
print("=" * 96)
print("判读")
print("=" * 96)
if res["orig"] and res["v2"]:
    ro, r2 = np.array(res["orig"]), np.array(res["v2"])
    zo = np.array(diff_res["orig"]) if diff_res["orig"] else np.array([0.0])
    z2 = np.array(diff_res["v2"]) if diff_res["v2"] else np.array([0.0])
    print(f"  RMS 跳变:  原版 {np.abs(ro).max():.2f}dB  ->  "
          f"v2 {np.abs(r2).max():.2f}dB")
    print(f"  差分 z:    原版 {np.nanmax(np.abs(zo)):.2f}  ->  "
          f"v2 {np.nanmax(np.abs(z2)):.2f}")
    print()
    if np.abs(r2).max() <= np.abs(ro).max() + 1.0:
        print("  ✓ **跳变来自音频本身**（静音/语音交界），不是处理引入")
        print(f"    原版在同一位置的跳变更大或相当（{np.abs(ro).max():.1f} vs "
              f"{np.abs(r2).max():.1f} dB）")
    else:
        print("  ⚠️ v2 的跳变明显大于原版，可能确有接缝问题")
    print()
    if np.nanmax(np.abs(z2)) < 30:
        print("  ✓ 差分 z-score 正常（<30）—— 边界无样本级不连续")
    else:
        print(f"  ⚠️ 差分 z-score 偏大（{np.nanmax(np.abs(z2)):.1f}）"
              "，边界可能有突变")

print()
print("  ## 为什么原版也有大跳变")
print("    处理区起点 = `voiced_end - 1.0s`。")
print("    若该点落在**句中停顿**（静音），前后 50ms 的 RMS 差自然很大。")
print("    这不是接缝，是**语音本身的停顿**。")

rep = ["=" * 78, "接缝检查（带原版对照）", "=" * 78, ""]
rep.append(f"{'版本':<8s}{'n':>5s}{'RMS跳变均值':>14s}{'RMS最大':>12s}"
           f"{'差分z最大':>12s}")
for tag in ("orig", "v1", "v2"):
    if not res[tag]:
        continue
    r = np.array(res[tag])
    z = np.array(diff_res[tag]) if diff_res[tag] else np.array([np.nan])
    rep.append(f"{tag:<8s}{len(r):5d}{r.mean():+13.2f}dB"
               f"{np.abs(r).max():11.2f}dB{np.nanmax(np.abs(z)):12.2f}")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
print()
print(f"  -> {OUT / 'report.txt'}")
print("=" * 96)
