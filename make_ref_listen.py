# -*- coding: utf-8 -*-
"""参考音频 A/B 盲听包 —— 判断 cand3 是否真的更好。

## 待验证
    orig  : 尾平坦 0.3025  尾/体 +0.89dB  相似度 0.8489
    cand3 : 尾平坦 0.1959  尾/体 +6.14dB  相似度 0.8397
    -> 尾音噪声降 35%，但尾部能量 +5.3dB

**指标一好一坏，必须靠耳朵。**

## 设计
· 三组：orig / cand1 / cand3
· 每句拼接：orig -> cand3 -> cand1（**固定顺序**）
· 响度统一 RMS 0.06 + 峰值保护
· 多相 FIR 重采样 48kHz（不是线性插值）
· 句间 0.8s 静音
· 另出盲测包（打乱顺序）

## 输出
    dist/ref_listen/
        NN_*_3way.wav      orig -> cand3 -> cand1
        blind/NN_*_AB.wav  打乱
        answer_key.txt
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
SRC = V5 / "dist" / "ref_swap_ab"
OUT = V5 / "dist" / "ref_listen"
TARGET_SR = 48000
TARGET_RMS = 0.06
GAP_S = 0.8
SEED = 20261011

GROUPS = [("orig", "当前 ref/default"), ("cand3", "剧情_e7m1_002"),
          ("cand1", "剧情_e9m4_010")]


def load_mono(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return x, int(sr)


def to48(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return x
    try:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(int(sr), TARGET_SR)
        return resample_poly(x, TARGET_SR // g, int(sr) // g).astype(np.float64)
    except Exception:  # noqa: BLE001
        n = int(round(len(x) * TARGET_SR / sr))
        return np.interp(np.linspace(0, len(x) - 1, n),
                         np.arange(len(x)), x).astype(np.float64)


def align_rms(x: np.ndarray) -> np.ndarray:
    r = float(np.sqrt((x ** 2).mean()))
    if r < 1e-9:
        return x
    y = x * (TARGET_RMS / r)
    pk = float(np.abs(y).max())
    return y * (0.99 / pk) if pk > 0.99 else y


print("=" * 92)
print("参考音频盲听包")
print("=" * 92)
for g, why in GROUPS:
    d = SRC / g
    n = len(list(d.glob("*.wav"))) if d.is_dir() else 0
    print(f"  {g:8s} {n:3d} 个  ({why})")

(OUT / "blind").mkdir(parents=True, exist_ok=True)
gap = np.zeros(int(GAP_S * TARGET_SR))
rng = random.Random(SEED)
key = ["# 盲听答案（听完再看）", "",
       "# 固定顺序: 第1段=orig(当前)  第2段=cand3  第3段=cand1", ""]
made = 0

for i in range(1, 11):
    clips, ok = [], True
    for g, _ in GROUPS:
        p = SRC / g / f"{i:02d}.wav"
        if not p.exists():
            print(f"  !! 缺 {p}")
            ok = False
            break
        x, sr = load_mono(p)
        clips.append(align_rms(to48(x, sr)))
    if not ok:
        continue
    cat = np.concatenate([c for c in clips for c in (c, gap)])
    sf.write(str(OUT / f"{i:02d}_3way.wav"), cat, TARGET_SR)

    order = list(range(len(GROUPS)))
    rng.shuffle(order)
    cat2 = np.concatenate([
        clips[idx] if j == 0 else np.concatenate([gap, clips[idx]])
        for j, idx in enumerate(order)])
    sf.write(str(OUT / "blind" / f"{i:02d}_AB.wav"), cat2, TARGET_SR)
    key.append(f"{i:02d}:  " + "  ".join(
        f"第{j+1}段={GROUPS[idx][0]}" for j, idx in enumerate(order)))
    made += 1

(OUT / "answer_key.txt").write_text("\n".join(key) + "\n", encoding="utf-8")
print()
print(f"  生成 {made} 个三方对比 + {made} 个盲测")
print(f"  -> {OUT}")
print()
print("=" * 92)
print("怎么听（重点听**句末**）")
print("=" * 92)
print("""
  ## 三方对比 `dist/ref_listen/NN_3way.wav`
      第1段 = orig  （当前 ref/default.wav）
      第2段 = cand3 （剧情_e7m1_002，尾音最干净）
      第3段 = cand1 （剧情_e9m4_010）

  ## 重点判断（只关注句末最后一个字之后）
    · 哪个**结尾收得干净**？（没有多余的"嘶"/"呼"声）
    · 哪个**结尾拖泥带水**？
    · 哪个**音色最像庄方宜**？（不只是尾音）

  ## 预期
    cand3 的尾音噪声应该**明显更少**（指标 -35%）
    但尾部能量更突出（+5.3dB）—— **可能听起来"结尾更重"**

  ## 关键取舍
    | | 尾音干净 | 相似度 |
    |---|---|---|
    | orig  | 0.3025 | **0.8489** |
    | cand3 | **0.1959** | 0.8397 |

    **相似度只差 0.0092（噪声内）** —— 所以如果 cand3 尾音明显更好，
    **换它是划算的**。

  ## 盲测
    `blind/` 里顺序打乱，听完对 `answer_key.txt`。
""")
print("=" * 92)
