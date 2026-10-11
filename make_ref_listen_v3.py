# -*- coding: utf-8 -*-
"""重做尾音包 —— **静音边界对齐 + 淡入**，修掉"吞音"。

## 上一版的 bug（用户反馈「都吞音了」）
    t0 = max(0, e - int(1.5 * SR))     # 固定往前 1.5s
    seg = x[t0:e]                       # 直接切 -> **落在字中间**
    # 无淡入、无边界对齐
实测：**6/10 个片段首样本非零、无开头静音** -> 听感"开头被吞掉"
而尾音包是**三段连续**，每段都这样 -> 用户说「**都**吞音了」

## 修法
    ① **向前找最近的静音边界**作为起点（最多回退到 2.5s）
       —— 若找不到静音，就**从句子开头**取（宁可长，不可切字）
    ② **加 10ms 淡入 + 10ms 淡出**（消除爆音）
    ③ **不做 RMS 归一化**，只做**统一峰值对齐**（避免短片段被压轻）
    ④ 起点前保留 50ms 静音（更自然）

## 同时产出
    · `tail2/NN_tail.wav`  —— 修好的尾音包（三段：orig / r908 / r815）
    · `full/NN_full.wav`   —— **完整句**三段（最保险，无切割问题）
    · `blind/NN.wav`       —— 打乱

## 用法
    python _tools/make_ref_listen_v3.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
OUT = V5 / "dist" / "ref_listen_v3"
for sub in ("tail", "full", "blind"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

GROUPS = [("orig", V5 / "dist" / "ref_swap_v2" / "orig", "当前 ref/default"),
          ("r908", V5 / "dist" / "ref_swap_v2" / "r908", "相似度最高"),
          ("r815", V5 / "dist" / "ref_swap_v2" / "r815", "最不浊")]
SR = 16000
TARGET_SR = 48000
GAP = 0.6
FADE = 0.01              # 10ms
LEAD_SIL = 0.05          # 起点前保留静音
MAX_BACK = 2.5           # 最多向前回退秒数
SEED = 20261011


def load16(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return np.asarray(x, dtype=np.float64)


def env10(x, sr=SR):
    n = int(0.01 * sr)
    m = (len(x) // n) * n
    if m < n:
        return np.array([np.sqrt((x ** 2).mean())]), n
    return np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1)), n


def voiced_end(x, sr=SR, t=-45.0):
    env, n = env10(x, sr)
    ref = env.max() + 1e-12
    v = np.where(env > ref * (10 ** (t / 20)))[0]
    return int((v[-1] + 1) * n) if len(v) else len(x)


def silence_aware_start(x, end_idx, want_sec=1.5, sr=SR,
                        thr_db=-40.0, max_back=MAX_BACK):
    """**向前找最近的静音边界**作为起点。

    ## 为什么
    固定 `end - 1.5s` 会落在字中间 -> 听感"吞掉开头"。
    改成：从 `end - want_sec` 出发**向前找**第一个静音帧（RMS 低于阈值），
    以它作为起点。若在 `max_back` 内找不到，就**从句子开头取**。

    ## 返回
        起点样本索引
    """
    env, n = env10(x, sr)
    ref = env.max() + 1e-12
    thr = ref * (10 ** (thr_db / 20))
    sil = env < thr
    want = max(0, end_idx - int(want_sec * sr))
    lo = max(0, end_idx - int(max_back * sr))
    # 从 want 往前找静音帧
    idx = want // n
    for j in range(idx, max(0, lo // n) - 1, -1):
        if j < len(sil) and sil[j]:
            # 取该静音帧的起点（再往前留一点）
            start = max(0, j * n - int(LEAD_SIL * sr))
            return start
    # 找不到静音 -> 从句子开头取（宁可长，不可切字）
    return 0


def to48(x):
    try:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(SR, TARGET_SR)
        return resample_poly(x, TARGET_SR // g, SR // g)
    except Exception:  # noqa: BLE001
        return x


def apply_fade(x, sr=TARGET_SR, fade=FADE):
    n = min(int(fade * sr), len(x) // 4)
    if n < 4:
        return x
    y = x.copy()
    y[:n] *= np.linspace(0, 1, n)
    y[-n:] *= np.linspace(1, 0, n)
    return y


def peak_align(x, target=0.90):
    """**峰值对齐**（不是 RMS）—— 避免短片段被压轻。"""
    pk = float(np.abs(x).max())
    if pk < 1e-9:
        return x
    return x * (target / pk)


print("=" * 100)
print("重做尾音包（静音边界对齐 + 淡入）")
print("=" * 100)
print(f"  修的问题: 固定 1.5s 切割 -> 落在字中间 -> 听感「吞音」")
print(f"  新策略:   向前找最近静音边界（最多回退 {MAX_BACK}s）")
print(f"            找不到就从句子开头取（宁可长，不可切字）")
print(f"            10ms 淡入淡出 + 峰值对齐（不做 RMS 归一化）")

rng = random.Random(SEED)
key = ["# 盲听答案", "",
       "# 固定顺序: 第1段=orig(当前)  第2段=r908(相似度高)  第3段=r815(最不浊)",
       ""]
stats = []
made = 0

for i in range(1, 11):
    full, tails = [], []
    ok = True
    for tag, d, _ in GROUPS:
        p = d / f"{i:02d}.wav"
        if not p.exists():
            ok = False
            break
        x = load16(p)
        e = voiced_end(x)
        # ---- 完整句（从开头到 voiced_end）----
        full.append(peak_align(to48(apply_fade(x[:e]))))
        # ---- 尾音：静音对齐 ----
        s = silence_aware_start(x, e, want_sec=1.5)
        seg = x[s:e]
        if len(seg) < int(0.2 * SR):
            seg = x[:e]
        stats.append({"i": i, "tag": tag, "start": s / SR,
                      "end": e / SR, "dur": len(seg) / SR,
                      "first": abs(float(seg[0]))})
        tails.append(peak_align(to48(apply_fade(seg))))
    if not ok:
        continue

    gap = np.zeros(int(GAP * TARGET_SR))
    sf.write(str(OUT / "tail" / f"{i:02d}_tail.wav"),
             np.concatenate([c for c in tails for c in (c, gap)]), TARGET_SR)
    sf.write(str(OUT / "full" / f"{i:02d}_full.wav"),
             np.concatenate([c for c in full for c in (c, gap)]), TARGET_SR)

    order = [0, 1, 2]
    rng.shuffle(order)
    seq = []
    for j, idx in enumerate(order):
        if j:
            seq.append(gap)
        seq.append(tails[idx])
    sf.write(str(OUT / "blind" / f"{i:02d}_tail_blind.wav"),
             np.concatenate(seq), TARGET_SR)
    key.append(f"{i:02d}: " + "  ".join(
        f"第{j+1}段={GROUPS[idx][0]}" for j, idx in enumerate(order)))
    made += 1

(OUT / "answer_key.txt").write_text("\n".join(key) + "\n", encoding="utf-8")

# ---------- 验证 ----------
print()
print("=" * 100)
print("验证：新的尾音片段是否从静音开始")
print("=" * 100)
print(f"  {'句':<4s} {'标签':<6s} {'起点':>7s} {'终点':>7s} {'时长':>7s} "
      f"{'首样本':>9s}  判定")
print("  " + "-" * 62)
bad = 0
for st in stats:
    tag = "ok" if st["first"] < 0.02 else "**首样本非零**"
    if st["first"] >= 0.02:
        bad += 1
    print(f"  {st['i']:02d}   {st['tag']:<6s} {st['start']:6.2f}s "
          f"{st['end']:6.2f}s {st['dur']:6.2f}s {st['first']:9.4f}  {tag}")

print()
print(f"  首样本非零: **{bad} / {len(stats)}**"
      f"  {'✓ 已修好' if bad == 0 else '⚠️ 仍有问题'}")

durs = [s["dur"] for s in stats]
print(f"  片段时长: {min(durs):.2f} ~ {max(durs):.2f}s "
      f"（均值 {np.mean(durs):.2f}s）")

print()
print("=" * 100)
print("产出")
print("=" * 100)
print(f"  {OUT / 'tail'}   尾音包（{made} 个，**已修**）")
print(f"  {OUT / 'full'}   **完整句**（{made} 个，最保险，无切割）")
print(f"  {OUT / 'blind'}  盲测（{made} 个）")
print()
print("=" * 100)
print("怎么听")
print("=" * 100)
print("""
  ## 先听 `full/NN_full.wav`（完整句，无切割）
     第1段 = orig（当前 ref/default）
     第2段 = r908
     第3段 = r815

    **完整句不会有"吞音"** —— 如果这里听着正常，说明模型没问题，
    之前的「吞音」纯粹是尾音包的切割 bug。

  ## 再听 `tail/NN_tail.wav`（已修：静音对齐 + 淡入）
     三段都是**完整的最后一个短语**（从静音边界开始）

  ## 对比要点
     · 尾音是否"往上扬"（orig 应该扬，r908/r815 应该不扬）
     · 音色是否像庄方宜
     · 是否还有"吞音"（应该没有了）

  ## ⚠️ 我的错误
    上一版尾音包用固定 `voiced_end - 1.5s` 切割，落在字中间，
     6/10 个片段首样本非零、无开头静音 -> 听起来"吞掉了开头"。
     而且尾音包是**三段连续**，每段都这样 -> 你说「都吞音了」。
     **这是我的 bug，不是模型的问题。**
""")
print("=" * 100)

rep = ["=" * 78, "重做尾音包（静音对齐）", "=" * 78, ""]
rep.append(f"首样本非零: {bad}/{len(stats)}")
rep.append(f"片段时长 {min(durs):.2f}~{max(durs):.2f}s 均值 {np.mean(durs):.2f}s")
rep.append("")
rep.append(f"{'句':<4s}{'标签':<7s}{'起点':>8s}{'终点':>8s}{'时长':>8s}{'首样本':>10s}")
for st in stats:
    rep.append(f"{st['i']:02d}   {st['tag']:<7s}{st['start']:7.2f}s"
               f"{st['end']:7.2f}s{st['dur']:7.2f}s{st['first']:10.4f}")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
print(f"  -> {OUT / 'report.txt'}")
print("=" * 100)
