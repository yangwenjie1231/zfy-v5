# -*- coding: utf-8 -*-
"""排查「吞音」—— 是模型的问题，还是**我的尾音包切割**造成的？

## 用户的反馈
「我感觉**都**吞音了」—— 听的是 `dist/ref_listen_v2/tail/NN_tail.wav`

## 我的怀疑（高）
尾音包的构造：
    e = voiced_end(x)                    # 最后一个有声帧
    t0 = max(0, e - int(1.5 * SR))
    tails.append(nz(to48(x[t0:e])))      # 直接切 [t0, e]
    # 然后 RMS 归一化 + 峰值保护

**问题**：
    ① **t0 落在字中间** -> 第一段音频从半个音节开始 -> 听感"吞掉了开头"
    ② **没有淡入** -> 第一个样本从 0 跳到某个幅度 -> 爆音（click）
    ③ **RMS 归一化 + 峰值保护** -> 若某段有强爆破音，整段被压低
       -> 短片段被压得很轻，听不清 = "吞音"

## 验证方法
    ① 测 t0 处是否处于**有声中间**（RMS 高于阈值）
    ② 测片段开头 5ms 的**样本跳变**（|x[0]| 与相邻差分）
    ③ 对比：**完整句**（用户之前听过，没抱怨吞音）vs **尾音片段**
    ④ 测片段内的**音节数**（能量谷计数）与预期对比

## 输出
    dist/tunyin_check/report.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
TAIL = V5 / "dist" / "ref_listen_v2" / "tail"
FULL = V5 / "dist" / "ref_listen_v2"
SWAP = V5 / "dist" / "ref_swap_v2"
OUT = V5 / "dist" / "tunyin_check"
OUT.mkdir(parents=True, exist_ok=True)
SR = 16000


def load16(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return np.asarray(x, dtype=np.float64)


def voiced_end(x, sr=SR, t=-45.0):
    n = int(0.01 * sr)
    m = (len(x) // n) * n
    if m < n * 4:
        return len(x)
    r = np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))
    ref = r.max() + 1e-12
    v = np.where(r > ref * (10 ** (t / 20)))[0]
    return int((v[-1] + 1) * n) if len(v) else len(x)


def rms_env(x, sr=SR, win=0.01):
    n = int(win * sr)
    m = (len(x) // n) * n
    if m < n:
        return np.array([np.sqrt((x ** 2).mean())])
    return np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))


print("=" * 100)
print("「吞音」排查：模型问题 vs 我的切割问题")
print("=" * 100)

print()
print("=" * 100)
print("① 尾音片段的构造检查")
print("=" * 100)
print(f"  {'文件':<22s} {'时长':>6s} {'首样本':>9s} {'开头5ms':>9s} "
      f"{'首帧RMS':>9s} {'峰值':>8s} {'RMS':>8s}  判定")
print("  " + "-" * 88)

issues = []
for w in sorted(TAIL.glob("*_tail.wav")):
    x = load16(w)
    if len(x) < 100:
        continue
    first = abs(float(x[0]))
    # 开头 5ms 的最大差分（爆音指标）
    n5 = int(0.005 * SR)
    d5 = float(np.abs(np.diff(x[:n5])).max()) if n5 > 2 else 0.0
    env = rms_env(x)
    first_rms = float(env[0]) if len(env) else 0.0
    pk = float(np.abs(x).max())
    rms = float(np.sqrt((x ** 2).mean()))
    # 判定
    tag = []
    if first > 0.05:
        tag.append("**首样本非零**")
    if d5 > 0.3:
        tag.append("**开头有跳变**")
    if first_rms > rms * 1.5:
        tag.append("**从字中间开始**")
    if pk < 0.15:
        tag.append("**音量过低**")
    print(f"  {w.name:<22s} {len(x)/SR:5.2f}s {first:9.4f} {d5:9.4f} "
          f"{first_rms:9.4f} {pk:8.4f} {rms:8.4f}  {'、'.join(tag) if tag else 'ok'}")
    if tag:
        issues.append((w.name, tag))

print()
print(f"  有问题的片段: **{len(issues)} / {len(list(TAIL.glob('*_tail.wav')))}**")

print()
print("=" * 100)
print("② 对比：完整句 vs 尾音片段（同一句）")
print("=" * 100)
print(f"  {'句':<6s} {'完整句时长':>10s} {'尾音时长':>10s} "
      f"{'完整句峰值':>10s} {'尾音峰值':>10s}  说明")
print("  " + "-" * 70)
for i in range(1, 11):
    f3 = FULL / f"{i:02d}_3way.wav"
    ft = TAIL / f"{i:02d}_tail.wav"
    if not ft.exists():
        continue
    xt = load16(ft)
    # 完整句取 orig 组
    fo = SWAP / "orig" / f"{i:02d}.wav"
    xf = load16(fo) if fo.exists() else None
    pkt = float(np.abs(xt).max())
    if xf is not None:
        pkf = float(np.abs(xf).max())
        print(f"  {i:02d}     {len(xf)/SR:9.2f}s {len(xt)/SR:9.2f}s "
              f"{pkf:10.4f} {pkt:10.4f}  "
              f"{'**尾音被压低**' if pkt < pkf * 0.6 else 'ok'}")
    else:
        print(f"  {i:02d}     {'—':>9s} {len(xt)/SR:9.2f}s {'—':>10s} "
              f"{pkt:10.4f}")

print()
print("=" * 100)
print("③ 音节数检查（能量谷计数）")
print("=" * 100)


def count_syllables(x, sr=SR):
    """用能量谷计数近似音节数。"""
    env = rms_env(x, sr, 0.01)
    if len(env) < 5:
        return 0
    # 平滑
    k = max(1, len(env) // 50)
    ker = np.ones(k) / k
    e = np.convolve(env, ker, "same")
    thr = e.max() * 0.15
    v = e > thr
    # 统计上升沿
    return int(np.sum((~v[:-1]) & v[1:])) + (1 if v[0] else 0)


print(f"  {'文件':<22s} {'尾音节数':>9s} {'完整句音节数':>12s}  说明")
print("  " + "-" * 56)
for i in range(1, 11):
    ft = TAIL / f"{i:02d}_tail.wav"
    fo = SWAP / "orig" / f"{i:02d}.wav"
    if not ft.exists():
        continue
    ct = count_syllables(load16(ft))
    cf = count_syllables(load16(fo)) if fo.exists() else 0
    print(f"  {i:02d}_tail.wav{'':<11s} {ct:9d} {cf:12d}")

print()
print("=" * 100)
print("结论")
print("=" * 100)
n_bad = len(issues)
n_tot = len(list(TAIL.glob("*_tail.wav")))
if n_bad > n_tot * 0.5:
    print(f"  **{n_bad}/{n_tot} 个尾音片段有构造问题** —— 「吞音」很可能是**我的包造成的**")
    print()
    print("  主要原因：")
    print("    · 从字中间切开 -> 开头半个音节丢失 -> 听感'吞掉开头'")
    print("    · 无淡入 -> 爆音")
    print("    · RMS 归一化 -> 短片段被压轻")
    print()
    print("  ## 修复方案")
    print("    1. **从静音边界切**（找 t0 之前最近的静音点），而不是固定 1.5s")
    print("    2. **加 10ms 淡入**")
    print("    3. **不做 RMS 归一化**（或只做峰值对齐）")
    print("    4. 保留更长的上下文（2.5s），确保包含完整的最后一句")
elif n_bad > 0:
    print(f"  {n_bad}/{n_tot} 个片段有问题，部分影响")
else:
    print("  ✓ 尾音片段构造正常 —— 「吞音」更可能来自**模型本身**")
    print()
    print("  ## 若确认是模型问题")
    print("    可能原因：")
    print("      · `repetition_penalty 1.35` 过高 -> 模型跳过 token（吞音）")
    print("      · `top_k 15` / `temperature 1.0` 采样导致丢字")
    print("      · 参考音频的韵律风格影响")
    print("    ## 验证方法")
    print("      用**完整句**（不是尾音片段）听，若完整句正常 -> 是片段问题")
    print("      若完整句也吞音 -> 调低 repetition_penalty 到 1.2 或 1.1 重测")

rep = ["=" * 78, "「吞音」排查", "=" * 78, ""]
rep.append(f"尾音片段有问题: {n_bad}/{n_tot}")
rep.append("")
for nm, tag in issues:
    rep.append(f"{nm}: {'、'.join(tag)}")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
print()
print(f"  -> {OUT / 'report.txt'}")
print("=" * 100)
