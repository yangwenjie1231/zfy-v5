# -*- coding: utf-8 -*-
"""GPT 盲听包 —— 三方对比（base / A_e4 / B_e4）。

## 为什么要听
指标**互相矛盾**：
    · 时长CV : base 最优（40.8%）
    · ratioσ : A_e4 最优（0.183）
两者对"韵律好坏"的方向判断相反 —— 因为
    · 时长CV 被**句长差异**主导（"好。"2 字 vs 35 字长句）
    · ratioσ 扣掉句长，但**低也可能=语调单一**（社区警告的过拟合症状）
**指标无法定论，只能靠耳朵。**

## 为什么选 e4
社区经验：「**GPT e2~e4 就到顶**，再训反而过拟合/不稳」。
本次数据也支持：**A_e4 / B_e4 的 ratioσ 最小**（0.183 / 0.208）。

## 设计
· 三组同句拼接：**base / A_e4 / B_e4**（固定顺序，便于对比）
· 响度统一对齐到 RMS 0.06（否则"响的听起来更好"）
· 句间 0.7s 静音
· 用**多相 FIR** 重采样到 48kHz（不是线性插值 —— 那会削高频 3.3dB）
· 另做一组**盲测包**（顺序打乱 + 答案单独存）

## 输出
    dist/gpt_listen/
        01_短_3way.wav    base -> A_e4 -> B_e4
        ...
        blind/01_短_AB.wav  打乱顺序
        answer_key.txt
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
AB = V5 / "dist" / "gpt_ab"
OUT = V5 / "dist" / "gpt_listen"
TARGET_SR = 48000
TARGET_RMS = 0.06
GAP_S = 0.7
SEED = 20261011

GROUPS = [("base", "生产 zfh-e4"), ("A_e4", "s1v3底模 e4"),
          ("B_e4", "v2底模 e4")]

TEXTS = {
    "01": "让我想想。", "02": "好。",
    "03": "迎敌或是迎客，都得整装后再出发。",
    "04": "出门前，要不要先喝杯茶提提神？", "05": "嗯……我理一理。",
    "06": "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？",
    "07": "武陵城外有一片梨树林，每年春天都会开满白色的花。",
    "08": "她就在那阵花香里，只是他自己不知道。",
    "09": "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。",
    "10": "这件事我三点五十分再去处理。",
    "11": "管理员，今天的实验数据我已经整理好了。",
    "12": "清波城那边的情况，我一直在关注。",
    "13": "若是要谈，就该拿出诚意来。",
    "14": "我答应你，会好好照顾自己。",
    "15": "这些话，我本想留到以后再说的。",
    "16": "你来了。坐吧，茶还热着。",
    "17": "有些事，不是努力就能改变的。",
    "18": "但不去试，就永远不知道结果。",
    "19": "汤汤说，我是个念旧的人。",
    "20": "或许吧。毕竟这座城，承载了太多。",
}


def load_mono(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return x, int(sr)


def to48(x: np.ndarray, sr: int) -> np.ndarray:
    """多相 FIR 重采样（**不是线性插值**）。

    线性插值的频响是 sinc²(f/48000)，16kHz 处 -3.3 dB ——
    会让 32k 源听起来更闷，制造假的"模型差异"。
    """
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
print("GPT 盲听包")
print("=" * 92)
for g, why in GROUPS:
    d = AB / g
    n = len(list(d.glob("*.wav"))) if d.is_dir() else 0
    print(f"  {g:8s} {n:3d} 个  ({why})")

(OUT / "blind").mkdir(parents=True, exist_ok=True)
gap = np.zeros(int(GAP_S * TARGET_SR))
rng = random.Random(SEED)
key = ["# 盲听答案（听完再看）", "", "# 三方对比固定顺序: base -> A_e4 -> B_e4", ""]
made = 0

for idx in sorted(TEXTS):
    clips = []
    ok = True
    for g, _ in GROUPS:
        d = AB / g
        # ⚠️ compare_gpt.py 存的是 **`01.wav`~`20.wav`**（两位补零，无后缀）。
        #    我第一版先试 `{idx}_*.wav`（匹配不到），
        #    兜底又写成 `{int(idx)}.wav` -> `1.wav`（也不存在）
        #    -> 只生成了 10~20，前 9 句全丢。
        #    正确：`{idx}.wav`（idx 已补零）。
        cands = [d / f"{idx}.wav"] if (d / f"{idx}.wav").exists() else []
        if not cands:
            cands = sorted(d.glob(f"{idx}_*.wav")) or \
                sorted(d.glob(f"{idx.zfill(2)}.wav")) or \
                sorted(d.glob(f"{int(idx)}.wav"))
        if not cands:
            print(f"  !! {g} 缺 {idx}")
            ok = False
            break
        x, sr = load_mono(cands[0])
        clips.append(align_rms(to48(x, sr)))
    if not ok:
        continue
    # 三方固定顺序
    cat = np.concatenate([c for c in clips for c in (c, gap)])
    kind = "短" if idx in ("01", "02") else (
        "中" if idx in ("03", "04", "05") else (
            "数字" if idx == "10" else "长"))
    sf.write(str(OUT / f"{idx}_{kind}_3way.wav"), cat, TARGET_SR)

    # 盲测（打乱）
    order = list(range(len(GROUPS)))
    rng.shuffle(order)
    cat2 = np.concatenate([clips[i] if j == 0 else np.concatenate([gap, clips[i]])
                           for j, i in enumerate(order)])
    sf.write(str(OUT / "blind" / f"{idx}_{kind}_AB.wav"), cat2, TARGET_SR)
    key.append(f"{idx}_{kind}:  " + "  ".join(
        f"第{j+1}段={GROUPS[i][0]}" for j, i in enumerate(order)))
    made += 1

(OUT / "answer_key.txt").write_text("\n".join(key) + "\n", encoding="utf-8")
print()
print(f"  生成 {made} 个三方对比 + {made} 个盲测")
print(f"  -> {OUT}")
print()
print("=" * 92)
print("怎么听")
print("=" * 92)
print("""
  ## 第一步：听三方对比（`dist/gpt_listen/*_3way.wav`）
  每个文件是**固定顺序**：
      第1段 = base（生产 zfh-e4）
      第2段 = A_e4（s1v3 底模微调）
      第3段 = B_e4（v2 底模微调）

  **重点判断**（用长句 06~09、11~20，最能听出韵律）：
    · 哪一段的**停顿位置**最自然？
    · 哪一段的**语气**最像庄方宜（不平、不呆）？
    · 哪一段**速度**最舒服？（base 明显更慢）
    · 有没有**念错字**、**吞字**？

  ## 第二步：盲测（`dist/gpt_listen/blind/`）
  顺序打乱，听完再对 `answer_key.txt`。

  ## 关键：base 的语速明显更慢（3.04 vs 3.77~3.95 字/秒）
  所以你会听到 base 更"慢条斯理"。**这算好还是坏，只有你能判断。**

  ## 指标矛盾提醒
    · 时长CV 说 base 好
    · ratioσ 说 A_e4 好
  **所以别信指标，信耳朵。**
""")
print("=" * 92)
