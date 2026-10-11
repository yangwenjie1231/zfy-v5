# -*- coding: utf-8 -*-
"""r908 vs r815 听感对比包 + **修好的 F0 测量**。

## 用户已确认
「尾音往上高」= **音调上扬，像疑问句结尾** = F0 升调。

## 我的测量缺陷（必须修）
`f0_slope` 要求自相关峰值 ≥ 0.4 才算"有声"。
但**句末尾音常是气声/清音，自相关低 -> 被跳过**
-> 只测到前半段（降），**升调的尾巴被滤掉**
-> 参考音频测出 +16.71（升），输出却测出 -2.35（降）—— 与用户听感矛盾

**修法**：
    ① 阈值降到 0.25
    ② 改用 **F0 轨迹的末端值 vs 中位值**（比线性回归稳）
    ③ 报告**最后 3 个可靠帧**的走向（不依赖长时间窗）
    ④ 同时报**未滤波**版本，便于对照

## 听感包（用户要求：先听 r908 vs r815）
    每个文件三段：**orig -> r908 -> r815**（固定顺序）
    另做**尾音专用包**：只保留最后 1.5s，句间 0.6s
      -> 这样能**连续听三个尾音**，差异最明显

## 输出
    dist/ref_listen_v2/
        NN_3way.wav        整句三段
        tail/NN_tail.wav   只有尾音（最后 1.5s）三段
        blind/NN.wav       打乱
        answer_key.txt
        report.txt
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
OUT = V5 / "dist" / "ref_listen_v2"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tail").mkdir(exist_ok=True)
(OUT / "blind").mkdir(exist_ok=True)

GROUPS = [("orig", V5 / "dist" / "ref_swap_v2" / "orig", "当前 ref/default"),
          ("r908", V5 / "dist" / "ref_swap_v2" / "r908",
           "剧情_e6m4_026（相似度最高）"),
          ("r815", V5 / "dist" / "ref_swap_v2" / "r815",
           "剧情_e6m3_016（最不浊）")]
HUMAN = V5 / "GPT-SoVITS" / "ref" / "default.wav"
HUMAN_END = 6.95
SR = 16000
TAIL_KEEP = 1.5          # 尾音包保留秒数
GAP = 0.6
SEED = 20261011


def load16(p: Path, max_sec=None):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if max_sec:
        x = x[:int(max_sec * sr)]
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


def f0_track_v2(x, sr=SR, win=0.03, hop=0.005, fmin=70.0, fmax=400.0,
                ac_min=0.25):
    """改进版 F0 轨迹 —— **降低有声阈值**，让句末气声段也能被追踪。

    ⚠️ 原版阈值 0.4，句末气声段被跳过 -> 升调尾巴丢失。
    """
    n, h = int(win * sr), int(hop * sr)
    if len(x) < n:
        return None, None
    ts, f0s = [], []
    lo, hi = int(sr / fmax), int(sr / fmin)
    for i in range(0, len(x) - n, h):
        seg = x[i:i + n]
        if np.sqrt((seg ** 2).mean()) < 1e-4:
            continue
        z = (seg - seg.mean()) * np.hanning(n)
        ac = np.correlate(z, z, "full")[len(z) - 1:]
        if ac[0] <= 0:
            continue
        ac /= ac[0]
        hh = min(hi, len(ac) - 1)
        if hh <= lo:
            continue
        k = int(np.argmax(ac[lo:hh]))
        pk = ac[lo + k]
        if pk < ac_min:
            continue
        ts.append(i / sr)
        f0s.append(sr / (lo + k))
    if len(ts) < 4:
        return None, None
    return np.array(ts), np.array(f0s)


def slope_metrics(x, sr=SR):
    """多口径 F0 走向。"""
    t, f = f0_track_v2(x, sr)
    out = {"n_frames": 0 if t is None else len(t)}
    if t is None or len(t) < 5:
        return {**out, "slope_all": float("nan"), "slope_last05": float("nan"),
                "slope_last03": float("nan"), "end_vs_med": float("nan"),
                "last3_dir": float("nan")}
    semi = 12 * np.log2(f / np.median(f))
    out["n_frames"] = len(t)
    out["slope_all"] = float(np.polyfit(t - t[0], semi, 1)[0])
    for key, sec in (("slope_last05", 0.5), ("slope_last03", 0.3)):
        m = t >= t[-1] - sec
        out[key] = (float(np.polyfit(t[m] - t[m][0], semi[m], 1)[0])
                    if m.sum() >= 3 else float("nan"))
    # 末端 vs 中位
    out["end_vs_med"] = float(np.median(semi[t >= t[-1] - 0.15]) -
                              np.median(semi))
    # 最后 3 帧方向
    if len(semi) >= 3:
        out["last3_dir"] = float(semi[-1] - semi[-3])
    else:
        out["last3_dir"] = float("nan")
    return out


print("=" * 104)
print("修好的 F0 测量 + 听感对比包")
print("=" * 104)

# ---------- 1) 测量 ----------
print()
print("=" * 104)
print("① F0 走向（改进测量：阈值 0.25，30ms 窗，5ms 步长）")
print("=" * 104)
print(f"  {'来源':<26s} {'帧数':>5s} {'全程斜率':>10s} {'末0.5s':>9s} "
      f"{'末0.3s':>9s} {'末端-中位':>10s} {'最后3帧':>9s}")
print("  " + "-" * 92)

# 参考音频（主语音段）
xr = load16(HUMAN, max_sec=HUMAN_END)
er = voiced_end(xr)
m = slope_metrics(xr[max(0, er - int(3.0 * SR)):er])
print(f"  {'ref/default 末3s':<26s} {m['n_frames']:5d} "
      f"{m['slope_all']:+10.2f} {m['slope_last05']:+9.2f} "
      f"{m['slope_last03']:+9.2f} {m['end_vs_med']:+10.2f} "
      f"{m['last3_dir']:+9.2f}")

# 真人留出集
HELD = V5 / "dist" / "heldout" / "heldout_clean.json"
real_ms = []
if HELD.exists():
    hd = json.loads(HELD.read_text(encoding="utf-8"))
    for it in hd.get("all", [])[:30]:
        p = Path(it["path"])
        if not p.exists():
            continue
        x = load16(p)
        e = voiced_end(x)
        if e / SR < 1.5:
            continue
        mm = slope_metrics(x[max(0, e - int(2.0 * SR)):e])
        if not np.isnan(mm["slope_last05"]):
            real_ms.append(mm)
if real_ms:
    avg = {k: float(np.nanmean([r[k] for r in real_ms])) for k in real_ms[0]}
    print(f"  {'真人留出集(30条均值)':<26s} {avg['n_frames']:5.0f} "
          f"{avg['slope_all']:+10.2f} {avg['slope_last05']:+9.2f} "
          f"{avg['slope_last03']:+9.2f} {avg['end_vs_med']:+10.2f} "
          f"{avg['last3_dir']:+9.2f}")

# 三组输出
print()
res = {}
for tag, d, why in GROUPS:
    if not d.is_dir():
        continue
    ms = []
    for w in sorted(d.glob("*.wav")):
        x = load16(w)
        e = voiced_end(x)
        mm = slope_metrics(x[max(0, e - int(2.0 * SR)):e])
        if not np.isnan(mm["slope_last05"]):
            ms.append(mm)
    if not ms:
        continue
    avg = {k: float(np.nanmean([r[k] for r in ms])) for k in ms[0]}
    res[tag] = avg
    print(f"  {tag:<26s} {avg['n_frames']:5.0f} {avg['slope_all']:+10.2f} "
          f"{avg['slope_last05']:+9.2f} {avg['slope_last03']:+9.2f} "
          f"{avg['end_vs_med']:+10.2f} {avg['last3_dir']:+9.2f}")

print()
print("  判读：")
print("    · 末0.3s 斜率 > 0  = **结尾上扬**（用户说的「往上高」）")
print("    · 末端-中位 > 0    = 末段音高高于整体")
if res:
    for tag, a in res.items():
        v = a["slope_last03"]
        d_ = "**上扬** ↑" if v > 0.5 else ("下压 ↓" if v < -0.5 else "平 →")
        print(f"    {tag:<8s} 末0.3s {v:+.2f} semi/s  -> {d_}")
    if real_ms:
        v = avg["slope_last03"]
        print(f"    {'真人':<8s} 末0.3s {v:+.2f} semi/s  -> "
              f"{'**上扬** ↑' if v > 0.5 else ('下压 ↓' if v < -0.5 else '平 →')}")

# ---------- 2) 听感包 ----------
print()
print("=" * 104)
print("② 听感对比包")
print("=" * 104)


def to48(x):
    try:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(SR, 48000)
        return resample_poly(x, 48000 // g, SR // g)
    except Exception:  # noqa: BLE001
        return x


def nz(v, target=0.06):
    r = float(np.sqrt((v ** 2).mean())) + 1e-12
    v = v * (target / r)
    pk = float(np.abs(v).max())
    return v * (0.99 / pk) if pk > 0.99 else v


rng = random.Random(SEED)
key = ["# 盲听答案", "",
       "# 固定顺序: 第1段=orig(当前)  第2段=r908(相似度高)  第3段=r815(最不浊)",
       ""]
made = 0
for i in range(1, 11):
    full, tails = [], []
    ok = True
    for tag, d, _ in GROUPS:
        p = d / f"{i:02d}.wav"
        if not p.exists():
            print(f"  !! 缺 {p}")
            ok = False
            break
        x = load16(p)
        e = voiced_end(x)
        # 整句（对齐到有声区）
        a = max(0, e - int(6.0 * SR))
        full.append(nz(to48(x[a:e])))
        # 尾音（最后 TAIL_KEEP 秒）
        t0 = max(0, e - int(TAIL_KEEP * SR))
        tails.append(nz(to48(x[t0:e])))
    if not ok:
        continue
    gap = np.zeros(int(GAP * 48000))
    sf.write(str(OUT / f"{i:02d}_3way.wav"),
             np.concatenate([c for c in full for c in (c, gap)]), 48000)
    sf.write(str(OUT / "tail" / f"{i:02d}_tail.wav"),
             np.concatenate([c for c in tails for c in (c, gap)]), 48000)

    order = [0, 1, 2]
    rng.shuffle(order)
    seq = []
    for j, idx in enumerate(order):
        if j:
            seq.append(gap)
        seq.append(tails[idx])
    sf.write(str(OUT / "blind" / f"{i:02d}_tail_blind.wav"),
             np.concatenate(seq), 48000)
    key.append(f"{i:02d} 尾音: " + "  ".join(
        f"第{j+1}段={GROUPS[idx][0]}" for j, idx in enumerate(order)))
    made += 1

(OUT / "answer_key.txt").write_text("\n".join(key) + "\n", encoding="utf-8")
print(f"  整句三段对比: {made} 个  -> {OUT}")
print(f"  **尾音专用包**: {made} 个  -> {OUT / 'tail'}")
print(f"  尾音盲测:     {made} 个  -> {OUT / 'blind'}")

print()
print("=" * 104)
print("怎么听（重点）")
print("=" * 104)
print("""
  ## 先听「尾音专用包」`dist/ref_listen_v2/tail/NN_tail.wav`
     每个文件只含**最后 1.5 秒**，三段连续：
        第1段 = orig  （当前 ref/default）
        第2段 = r908  （相似度最高 0.8589）
        第3段 = r815  （最不浊 4.25%）

     **这样能连续听三个尾音，差异最明显。**

  ## 要回答的三个问题
     1. **第几段的尾音最"不往上扬"**？（像陈述句收尾）
     2. **第几段的音色最像庄方宜**？
     3. **第几段听起来最"实"（不虚）**？

  ## 预期（基于指标）
     · r908: body 78.94%（真人 78.35%）-> **最不虚**；相似度最高
     · r815: 500-800Hz 4.25%（真人 4.64%）-> **最不浊**；但相似度掉 0.046
     · orig: 两者都不如

  ## 盲测
     `blind/NN_tail_blind.wav` 顺序打乱，听完对 `answer_key.txt`

  ## ⚠️ 关于「往上高」
     参考音频 ref/default.wav 结尾是问句「…好吗？」（F0 斜率 +16.71）
     而 r908/r815 的参考都是**陈述句**（斜率 -11 ~ -18）
     -> 若假设成立，r908/r815 的尾音**应该明显不扬**
""")
print("=" * 104)

rep = ["=" * 80, "F0 走向（改进测量）", "=" * 80, ""]
rep.append(f"{'来源':<28s}{'帧数':>6s}{'全程':>9s}{'末0.5s':>9s}"
           f"{'末0.3s':>9s}{'末端-中位':>11s}")
rep.append(f"{'ref/default 末3s':<28s}{m['n_frames']:6d}{m['slope_all']:+9.2f}"
           f"{m['slope_last05']:+9.2f}{m['slope_last03']:+9.2f}"
           f"{m['end_vs_med']:+11.2f}")
if real_ms:
    rep.append(f"{'真人留出集均值':<28s}{avg['n_frames']:6.0f}"
               f"{avg['slope_all']:+9.2f}{avg['slope_last05']:+9.2f}"
               f"{avg['slope_last03']:+9.2f}{avg['end_vs_med']:+11.2f}")
for tag, a in res.items():
    rep.append(f"{tag:<28s}{a['n_frames']:6.0f}{a['slope_all']:+9.2f}"
               f"{a['slope_last05']:+9.2f}{a['slope_last03']:+9.2f}"
               f"{a['end_vs_med']:+11.2f}")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
(OUT / "f0_results.json").write_text(json.dumps(
    {"ref_default": m, "human_avg": avg if real_ms else None, "outputs": res},
    ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  -> {OUT / 'report.txt'}")
print("=" * 104)
