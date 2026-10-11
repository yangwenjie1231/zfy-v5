# -*- coding: utf-8 -*-
"""「吞音」客观验证：**数音节**。

## 原理
中文**一个字 = 一个音节**。合成音频的能量包络上，每个音节对应一个"峰"。
    · 若模型吞字 -> 峰数 < 字数
    · 若正常     -> 峰数 ≈ 字数

## 为什么这个方法可靠
不依赖 ASR（本地没有可靠的中文 ASR）。
能量谷计数对"漏字"很敏感：漏一个音节就少一个峰。

## 同时检查两件事
    ① **完整句**是否吞音（模型问题）
    ② **尾音包**是否因"从字中间切"而制造了假吞音（我的问题）

## 判据
    完整句: 峰数 / 字数 应 >= 0.85
    尾音包: 只含最后 1.5s，本来就不含全部字数 -> 看**开头是否从静音起**

## 输出
    dist/syllable_check/report.txt
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
SWAP = V5 / "dist" / "ref_swap_v2"
TAIL = V5 / "dist" / "ref_listen_v2" / "tail"
OUT = V5 / "dist" / "syllable_check"
OUT.mkdir(parents=True, exist_ok=True)
SR = 16000

# compare_ref_swap.py 的 TEXTS（0-based -> 01.wav 是索引 0）
TEXTS = [
    "让我想想。",
    "好。",
    "迎敌或是迎客，都得整装后再出发。",
    "出门前，要不要先喝杯茶提提神？",
    "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？",
    "武陵城外有一片梨树林，每年春天都会开满白色的花。",
    "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。",
    "管理员，今天的实验数据我已经整理好了。",
    "若是要谈，就该拿出诚意来。",
    "或许吧。毕竟这座城，承载了太多了。",
]
STRIP = re.compile(r"[，。？！、…,.?!\s]")


def load16(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return np.asarray(x, dtype=np.float64)


def count_syllables(x, sr=SR, win=0.01, smooth_ms=30, thr_ratio=0.12):
    """能量峰计数（近似音节数）。

    · 10ms 帧 RMS
    · 30ms 平滑（去掉抖动）
    · 阈值 = 峰值 × thr_ratio
    · 统计"从低于阈值升到高于阈值"的次数 = 音节数
    """
    n = int(win * sr)
    m = (len(x) // n) * n
    if m < n * 3:
        return 0, None
    env = np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))
    k = max(1, int(smooth_ms / (win * 1000)))
    if k > 1:
        env = np.convolve(env, np.ones(k) / k, "same")
    if env.max() <= 0:
        return 0, env
    thr = env.max() * thr_ratio
    v = env > thr
    # 上升沿计数（首个为真的也算一次）
    rises = int(np.sum((~v[:-1]) & v[1:])) + (1 if v[0] else 0)
    return rises, env


def leading_silence(x, sr=SR, thr_db=-45.0):
    """开头静音时长（秒）。"""
    n = int(0.01 * sr)
    m = (len(x) // n) * n
    if m < n * 2:
        return 0.0
    env = np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))
    ref = env.max() + 1e-12
    v = env > ref * (10 ** (thr_db / 20))
    if not v.any():
        return len(x) / sr
    return float(np.argmax(v) * 0.01)


print("=" * 100)
print("「吞音」客观验证：数音节")
print("=" * 100)

print()
print("=" * 100)
print("① 完整句（模型是否吞字）")
print("=" * 100)
print(f"  {'句':<4s} {'字数':>4s} | {'orig':>14s} {'r908':>14s} {'r815':>14s}")
print(f"  {'':<4s} {'':>4s} | {'峰/率':>14s} {'峰/率':>14s} {'峰/率':>14s}")
print("  " + "-" * 62)

bad_full = []
for i, text in enumerate(TEXTS, 1):
    n_ch = len(STRIP.sub("", text))
    line = f"  {i:02d}   {n_ch:4d} |"
    for tag in ("orig", "r908", "r815"):
        p = SWAP / tag / f"{i:02d}.wav"
        if not p.exists():
            line += f" {'—':>14s}"
            continue
        x = load16(p)
        peaks, _ = count_syllables(x)
        ratio = peaks / n_ch if n_ch else 0
        flag = "" if ratio >= 0.85 else "!"
        line += f" {peaks:6d}/{ratio*100:5.0f}%{flag}"
        if ratio < 0.85:
            bad_full.append((i, tag, n_ch, peaks, ratio))
    print(line)

print()
if bad_full:
    print(f"  ⚠️ 疑似吞字的: {len(bad_full)} 处")
    for i, tag, n, pk, r in bad_full[:10]:
        print(f"      {i:02d}.wav {tag}: 字数 {n} 峰 {pk} ({r*100:.0f}%)")
else:
    print("  ✓ 完整句峰数/字数 >= 85% —— **模型没有大面积吞字**")

print()
print("=" * 100)
print("② 尾音包（我的切割是否制造假吞音）")
print("=" * 100)
print(f"  {'文件':<20s} {'时长':>6s} {'开头静音':>9s} {'峰数':>6s} "
      f"{'首样本':>9s}  判定")
print("  " + "-" * 74)
bad_tail = 0
n_tail = 0
for w in sorted(TAIL.glob("*_tail.wav")):
    x = load16(w)
    n_tail += 1
    ls = leading_silence(x)
    peaks, _ = count_syllables(x)
    first = abs(float(x[0]))
    tag = []
    if ls < 0.02:
        tag.append("**无开头静音**")
    if first > 0.03:
        tag.append("**首样本非零**")
    if tag:
        bad_tail += 1
    print(f"  {w.name:<20s} {len(x)/SR:5.2f}s {ls:8.3f}s {peaks:6d} "
          f"{first:9.4f}  {'、'.join(tag) if tag else 'ok'}")

print()
print(f"  有问题的尾音片段: **{bad_tail} / {n_tail}**")

print()
print("=" * 100)
print("③ 关键对比：尾音片段的第一段 vs 源文件同位置")
print("=" * 100)
print("""
  尾音包 = [最后1.5s] + 0.6s + [最后1.5s] + 0.6s + [最后1.5s] + 0.6s
  每段都从 `voiced_end - 1.5s` 开始 —— **很可能落在字中间**。

  听感：连续三段都"从半个字开始" -> **像吞掉了开头** -> 用户说「都吞音了」
""")

# 检查每段起点是否落在语音中间
print(f"  {'文件':<20s} {'起点处RMS':>10s} {'片段RMS':>9s} {'比值':>7s}  判定")
print("  " + "-" * 60)
mid_start = 0
for i in range(1, 11):
    p = SWAP / "orig" / f"{i:02d}.wav"
    if not p.exists():
        continue
    x = load16(p)
    # 复现 tail 的切割
    n = int(0.01 * SR)
    m = (len(x) // n) * n
    env = np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))
    ref = env.max() + 1e-12
    v = np.where(env > ref * (10 ** (-45 / 20)))[0]
    e = int((v[-1] + 1) * n) if len(v) else len(x)
    t0 = max(0, e - int(1.5 * SR))
    seg = x[t0:e]
    if len(seg) < 100:
        continue
    r_start = float(np.sqrt((seg[:n] ** 2).mean()))
    r_all = float(np.sqrt((seg ** 2).mean()))
    ratio = r_start / (r_all + 1e-12)
    is_mid = ratio > 0.5
    if is_mid:
        mid_start += 1
    print(f"  {i:02d}.wav{'':<13s} {r_start:10.4f} {r_all:9.4f} "
          f"{ratio:7.2f}  {'**从字中间开始**' if is_mid else '从静音/弱音开始'}")
print()
print(f"  **{mid_start} / 10 个片段从字中间开始**")

print()
print("=" * 100)
print("结论")
print("=" * 100)
if not bad_full:
    print("  ✓ **完整句没有吞字**（峰数/字数 >= 85%）")
    print("     -> 模型本身是好的")
print()
if mid_start > 5:
    print(f"  ✗ **{mid_start}/10 个尾音片段从字中间开始**")
    print("     -> 用户听到的「吞音」是**我的尾音包造成的**")
    print()
    print("  ## 为什么这听起来像吞音")
    print("     尾音包每段只取最后 1.5s，起点落在句子中间。")
    print("     连续三段都是「从半个字开始」-> 每个开头都像被吞掉了。")
    print()
    print("  ## 修复")
    print("     1. 起点对齐到**最近的静音边界**（而不是固定 1.5s）")
    print("     2. 加 10ms 淡入（避免爆音）")
    print("     3. 保留完整最后一句（从标点处切）")
    print()
    print("  ## 或者更简单")
    print("     **直接听完整句**（`dist/ref_listen_v2/NN_3way.wav`）")
    print("     那里的三段都是完整的，不存在切割问题")
else:
    print("  尾音片段起点基本正常 —— 需另找原因")

rep = ["=" * 78, "「吞音」验证", "=" * 78, ""]
rep.append(f"完整句疑似吞字: {len(bad_full)} 处")
rep.append(f"尾音片段从字中间开始: {mid_start}/10")
rep.append(f"尾音片段无开头静音: {bad_tail}/{n_tail}")
rep.append("")
for i, tag, n, pk, r in bad_full[:20]:
    rep.append(f"{i:02d}.wav {tag}: 字数 {n} 峰 {pk} ({r*100:.0f}%)")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
print()
print(f"  -> {OUT / 'report.txt'}")
print("=" * 100)
