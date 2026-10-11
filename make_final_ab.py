# -*- coding: utf-8 -*-
"""最终对照：原版 vs 后处理 vs 换参考 —— 一次性回答"问题解决了没"。

## 三个方案的实测数据
    A. orig      参考 ref/default.wav（生产配置）
                 尾/体 +0.89dB  尾平坦 0.3025  语速 3.13  相似度 0.8488
    B. tailfix   orig + 后处理（f350/d-20/t-10）
                 尾/体 -2.01dB  尾平坦 0.3109  语速 3.13  相似度 0.8488（不变）
    C. t908      参考换 剧情_e6m4_026
                 尾/体 +1.80dB  尾平坦 0.3525  语速 3.34  相似度 0.8589
    真人参考     尾/体 -4.94dB  尾平坦 0.1352  语速 —      —

## 关键问题
    · 后处理修好了"不衰减"（+0.89 -> -2.01），但**没修噪声**（0.3025 -> 0.3109）
    · 换参考（t908）相似度 +0.0100，但尾音**没改善**
    · **没有全赢方案**

## ⚠️ 必须验证的混淆
t816/t700 的"尾音 -78%"可能是**测量假象**：
    它们的输出语速只有 2.33 字/秒（正常 3~5）
    -> 句末字拖长、自然衰减 -> "尾部 0.4s" 窗口里内容不同
    -> 尾平坦读数偏低，但**不是真的更干净**
本脚本会同时报**语速**，便于识别这个混淆。

## 输出
    dist/final_ab/
        NN_3way.wav     A(orig) -> B(tailfix) -> C(t908)
        blind/NN_AB.wav 打乱
        answer_key.txt
        report.txt
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
A_DIR = V5 / "dist" / "ref_swap_train" / "orig"
B_DIR = V5 / "dist" / "tailfix_sweep" / "best"
C_DIR = V5 / "dist" / "ref_swap_train" / "t908"
OUT = V5 / "dist" / "final_ab"
TARGET_SR = 48000
TARGET_RMS = 0.06
GAP_S = 0.8
SEED = 20261011

GROUPS = [("A_orig", A_DIR, "生产配置（当前）"),
          ("B_tailfix", B_DIR, "orig + 尾音后处理"),
          ("C_t908", C_DIR, "换参考 剧情_e6m4_026")]


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


print("=" * 96)
print("最终对照包")
print("=" * 96)
for tag, d, why in GROUPS:
    n = len(list(d.glob("*.wav"))) if d.is_dir() else 0
    print(f"  {tag:10s} {n:3d} 个  ({why})")

(OUT / "blind").mkdir(parents=True, exist_ok=True)
gap = np.zeros(int(GAP_S * TARGET_SR))
rng = random.Random(SEED)
key = ["# 盲听答案", "", "# 固定顺序: 第1段=A_orig  第2段=B_tailfix  第3段=C_t908", ""]
made = 0

for i in range(1, 11):
    clips, ok = [], True
    for tag, d, _ in GROUPS:
        p = d / f"{i:02d}.wav"
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

# ---------- 指标表 ----------
print()
print("=" * 96)
print("指标汇总")
print("=" * 96)


def stats(d: Path) -> dict:
    tbs, fls, sps = [], [], []
    for w in sorted(d.glob("*.wav")):
        x, sr = sf.read(str(w), always_2d=True, dtype="float64")
        x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
        if len(x) < sr * 0.3:
            continue
        nf = 256
        m = (len(x) // nf) * nf
        rmsf = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
        r = float(np.sqrt((x ** 2).mean())) + 1e-12
        v = np.where(rmsf > r * (10 ** (-45 / 20)))[0]
        if len(v) == 0:
            continue
        end = int((v[-1] + 1) * nf)
        n = int(0.4 * sr)
        tail = x[max(0, end - n):end]
        body = x[max(0, end // 2 - sr // 2):end // 2 + sr // 2]
        if len(tail) < n // 2:
            continue
        tr = float(np.sqrt((tail ** 2).mean()))
        br = float(np.sqrt((body ** 2).mean())) + 1e-12
        tbs.append(20 * np.log10(tr / br + 1e-12))
        nn = min(len(tail), 2048)
        S = np.abs(np.fft.rfft(tail[:nn] * np.hanning(nn))) + 1e-12
        fls.append(float(np.exp(np.log(S).mean()) / S.mean()))
        sps.append(len(x) / sr)
    return {"tb": float(np.mean(tbs)), "flat": float(np.mean(fls)),
            "dur": float(np.mean(sps)), "n": len(tbs)}


print(f"  {'方案':<22s} {'尾/体dB':>9s} {'尾平坦':>8s} {'时长':>7s}  说明")
print("  " + "-" * 84)
REF_TB, REF_FLAT = -4.94, 0.1352
print(f"  {'真人参考':<22s} {REF_TB:+9.2f} {REF_FLAT:8.4f} {'—':>7s}  "
      f"目标")
rows = {}
for tag, d, why in GROUPS:
    if not d.is_dir() or not list(d.glob("*.wav")):
        continue
    s = stats(d)
    rows[tag] = s
    print(f"  {tag:<22s} {s['tb']:+9.2f} {s['flat']:8.4f} {s['dur']:6.2f}s  {why}")

print()
print("=" * 96)
print("判读")
print("=" * 96)
if "A_orig" in rows:
    a = rows["A_orig"]
    for tag in ("B_tailfix", "C_t908"):
        if tag not in rows:
            continue
        s = rows[tag]
        dtb = s["tb"] - a["tb"]
        dfl = s["flat"] - a["flat"]
        print(f"\n  {tag} vs A_orig:")
        print(f"    尾/体   {a['tb']:+.2f} -> {s['tb']:+.2f} dB  "
              f"({dtb:+.2f})  {'✓ 更接近真人' if abs(s['tb']-REF_TB) < abs(a['tb']-REF_TB) else '✗ 更远'}")
        print(f"    尾平坦  {a['flat']:.4f} -> {s['flat']:.4f}  "
              f"({dfl:+.4f})  {'✓ 更干净' if dfl < -0.02 else ('✗ 更脏' if dfl > 0.02 else '~ 无变化')}")
        print(f"    时长    {a['dur']:.2f}s -> {s['dur']:.2f}s")

print()
print("=" * 96)
print("结论")
print("=" * 96)
print("""
  ## 直接回答「问题解决了没」
    · **"句末不衰减"** -> 后处理**能修**（+0.89 -> -2.01 dB）
    · **"句末噪声高"** -> **没修好**
        · 后处理无效（0.3025 -> 0.3109）
        · 换参考（t908）也无效（0.3525）
        · 只有 t816/t700 显示"降低 78%"，但它们**语速掉到 2.33**
          -> 很可能是**测量混淆**（句末拖长导致窗口内容不同），不是真改善

  ## 所以
    **没有完全解决。** 但也不是"没进展"：
      · 根因已确认（参考音频尾部脏 + 含文本外语音）
      · "不衰减"有零风险修复（后处理）
      · "噪声"这一维目前无解（不牺牲其他指标的前提下）

  ## 听这三个文件判断
    dist/final_ab/NN_3way.wav
      第1段 = A_orig（生产配置）
      第2段 = B_tailfix（后处理版）
      第3段 = C_t908（换参考版）

    **重点：第2段比第1段，句末是否"收得干净些"？**
    如果听不出差别 -> 后处理不值得加
    如果听得出 -> 加上它（零风险）
""")
print("=" * 96)

rep = ["=" * 78, "最终对照", "=" * 78, ""]
rep.append(f"{'方案':<24s}{'尾/体dB':>10s}{'尾平坦':>10s}{'时长':>9s}")
rep.append(f"{'真人参考':<24s}{REF_TB:+10.2f}{REF_FLAT:10.4f}{'—':>9s}")
for tag, s in rows.items():
    rep.append(f"{tag:<24s}{s['tb']:+10.2f}{s['flat']:10.4f}{s['dur']:8.2f}s")
rep.append("")
rep.append("结论: '不衰减'可修（后处理），'噪声'未修好")
rep.append("t816/t700 的 -78% 疑似测量混淆（语速 2.33 导致窗口内容不同）")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
print(f"  -> {OUT / 'report.txt'}")
print(f"  -> 三方对比 {made} 个 + 盲测 {made} 个")
