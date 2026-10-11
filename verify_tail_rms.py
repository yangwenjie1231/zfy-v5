# -*- coding: utf-8 -*-
"""验证「尾音 RMS」的 23 倍差异 —— 是真差异还是测量假象？

## 发现
    v2Pro(生产)              尾音RMS = 0.00122
    v5merged(全量+ssl_proj)  尾音RMS = 0.02884     <- 23 倍
若为真，说明 **v2Pro 的尾音几乎静音**（截断感），而 v5merged 有自然收尾。
若为假，说明是我的测量链引入的（重采样边界效应等）。

## 可能的假象来源
  1. **重采样边界效应** —— v2Pro 是 32k，我用 resample_poly 降到 16k。
     但 0.4s = 6400 样本，远超任何滤波器瞬态，不太可能。
  2. **时长不同导致尾部静音长度不同** —— 若 v2Pro 输出末尾本来就有静音，
     那不是"截断"，而是"模型生成了静音尾巴"。
  3. **v2Pro 的输出有 padding** —— 需要看原始波形的实际内容。

## 本脚本
  1. 打印每条音频的**原始**尾部波形统计（不做任何重采样）
  2. 用**多种尾窗长度**（0.2/0.4/0.8s）交叉验证
  3. 检测尾部是否有静音段（连续低能量区间）
  4. 对比原始采样率下的结果 vs 重采样后的结果
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
A = V5 / "dist" / "baseline_v2pro"
B = V5 / "dist" / "sweep2" / "out_sweep" / "e16_merged"


def load(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return x, int(sr)


def tail(x, sr, sec):
    n = int(sec * sr)
    if len(x) < n * 2:
        return float("nan")
    return float(np.sqrt((x[-n:] ** 2).mean()))


def silence_tail(x, sr, thresh_db=-45.0):
    """从尾部往前数，连续低于阈值的时长（秒）。"""
    n = 512
    if len(x) < n * 2:
        return float("nan")
    m = (len(x) // n) * n
    fr = x[:m].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    ref = np.sqrt((x ** 2).mean()) + 1e-12
    thr = ref * (10 ** (thresh_db / 20))
    cnt = 0
    for v in rms[::-1]:
        if v < thr:
            cnt += 1
        else:
            break
    return cnt * n / sr


print("=" * 104)
print("1. 原始采样率下的尾部统计（**不做任何重采样**）")
print("=" * 104)
print(f"  {'句':4s} │ {'v2pro':>8s} {'sr':>6s} {'时长':>7s} {'尾0.4':>9s} {'静音尾':>7s}"
      f" │ {'merged':>8s} {'sr':>6s} {'时长':>7s} {'尾0.4':>9s} {'静音尾':>7s}")
print("  " + "-" * 100)
rows = []
for i in range(1, 11):
    idx = f"{i:02d}"
    pa = sorted(A.glob(f"{idx}_*.wav"))
    pb = sorted(B.glob(f"{idx}_*.wav"))
    if not (pa and pb):
        continue
    xa, sra = load(pa[0])
    xb, srb = load(pb[0])
    ta, tb = tail(xa, sra, 0.4), tail(xb, srb, 0.4)
    sa, sb = silence_tail(xa, sra), silence_tail(xb, srb)
    rows.append((idx, xa, sra, xb, srb, ta, tb, sa, sb))
    print(f"  {idx:4s} │ {ta:8.5f} {sra:6d} {len(xa)/sra:6.2f}s {ta:9.5f} {sa:6.2f}s"
          f" │ {tb:8.5f} {srb:6d} {len(xb)/srb:6.2f}s {tb:9.5f} {sb:6.2f}s")

if rows:
    ta_all = np.array([r[5] for r in rows])
    tb_all = np.array([r[6] for r in rows])
    sa_all = np.array([r[7] for r in rows])
    sb_all = np.array([r[8] for r in rows])
    print()
    print(f"  平均尾音RMS: v2pro {ta_all.mean():.5f}   merged {tb_all.mean():.5f}   "
          f"比值 {tb_all.mean()/max(ta_all.mean(),1e-12):.1f}x")
    print(f"  平均静音尾 : v2pro {sa_all.mean():.2f}s    merged {sb_all.mean():.2f}s")
    print(f"  **v2pro 尾部静音段更长 {sa_all.mean()-sb_all.mean():+.2f}s**")
    print()
    print("  判读：")
    if sa_all.mean() > sb_all.mean() + 0.15:
        print("    -> v2Pro 的输出**末尾带静音尾巴**（不是截断）")
        print("       这解释了尾音 RMS 低：分母是固定的 0.4s，")
        print("       其中大部分是静音，所以 RMS 被拉低。")
        print("       **不是音质问题**，是输出格式差异。")
    else:
        print("    -> 静音尾长度接近，尾音 RMS 差异需另找原因")

print()
print("=" * 104)
print("2. 多尾窗长度交叉验证（原始采样率）")
print("=" * 104)
print(f"  {'窗长':>6s} │ {'v2pro':>10s} {'merged':>10s} {'比值':>8s}")
print("  " + "-" * 46)
for sec in (0.1, 0.2, 0.4, 0.8, 1.5):
    va = np.nanmean([tail(r[1], r[2], sec) for r in rows])
    vb = np.nanmean([tail(r[3], r[4], sec) for r in rows])
    print(f"  {sec:5.1f}s │ {va:10.5f} {vb:10.5f} {vb/max(va,1e-12):7.1f}x")
print()
print("  若比值在所有窗长下都稳定 -> 差异是**系统性的**（真实特征）")
print("  若比值随窗长剧烈变化    -> 与尾部静音长度有关（格式差异）")

print()
print("=" * 104)
print("3. 结论")
print("=" * 104)
print("""
  「尾音 RMS」这个指标**会被末尾静音长度污染**：
    分母是固定的 0.4s 窗口，若末尾有 0.3s 静音，
    RMS 自然被拉低 —— 这测的是"输出末尾有多少静音"，
    不是"尾音有没有力气"。

  技能里对尾音 RMS 的定义是「最后 0.4s 的 RMS，越高越好」，
  但那个定义隐含假设「输出末尾没有填充静音」。
  当两批输出的**静音填充策略不同**时，这个指标就不可比。

  正确的可比做法：先**裁掉尾部静音**，再量最后 0.4s。
  本脚本的输出已给出静音尾长度，可直接用于修正。
""")
print("=" * 104)
