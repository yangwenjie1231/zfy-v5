# -*- coding: utf-8 -*-
"""决定性评估：换参考后，**用户听感对应的指标**是否改善。

## 为什么不用 compare_ref_swap.py 的指标
它报的是「尾平坦 / 尾体dB」—— 那是"气声/衰减"，**不是用户的听感**。
用户说的是「往上高」（F0 升调）+「虚」（body 不足）+「浊」（500-800Hz 多）。

## 本脚本测的四项（已确认与听感对应）
    ① **末段 F0 斜率**（semi/s）—— 正=往上高
    ② **body 150-800Hz** %     —— 低=虚
    ③ **500-800Hz** %          —— 高=浊
    ④ **相似度**               —— 音色是否保住
    另附 谱质心 / H1-H2 参考

## 对照
    真人（ref/default.wav 主语音段 0~6.95s）
    orig（当前参考）
    r908（剧情_e6m4_026，F0 斜率 -11.20）
    r815（剧情_e6m3_016，F0 斜率 -18.07，最不浊）

## 输出
    dist/ref_eval_v2/report.txt
    dist/ref_eval_v2/summary.png
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
OUT = V5 / "dist" / "ref_eval_v2"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DIRS = [("orig", V5 / "dist" / "ref_swap_v2" / "orig", "当前 ref/default"),
        ("r908", V5 / "dist" / "ref_swap_v2" / "r908",
         "剧情_e6m4_026（F0斜率-11.2）"),
        ("r815", V5 / "dist" / "ref_swap_v2" / "r815",
         "剧情_e6m3_016（F0斜率-18.1，最不浊）")]
HUMAN = GSV / "ref" / "default.wav"
HUMAN_END = 6.95
HELD = V5 / "dist" / "heldout" / "heldout_clean.json"
SR = 16000
TAIL_SEC = 1.0


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


def f0_slope(x, sr=SR, tail=0.5):
    n, h = int(0.04 * sr), int(0.01 * sr)
    if len(x) < n * 4:
        return float("nan")
    ts, f0s = [], []
    for i in range(0, len(x) - n, h):
        seg = x[i:i + n]
        if np.sqrt((seg ** 2).mean()) < 1e-3:
            continue
        z = (seg - seg.mean()) * np.hanning(n)
        ac = np.correlate(z, z, "full")[len(z) - 1:]
        if ac[0] <= 0:
            continue
        ac /= ac[0]
        lo, hi = int(sr / 350), min(int(sr / 70), len(ac) - 1)
        if hi <= lo:
            continue
        k = int(np.argmax(ac[lo:hi]))
        if ac[lo + k] < 0.4:
            continue
        ts.append(i / sr)
        f0s.append(sr / (lo + k))
    if len(ts) < 6:
        return float("nan")
    ts, f0s = np.array(ts), np.array(f0s)
    m = ts >= ts[-1] - tail
    if m.sum() < 4:
        return float("nan")
    s = 12 * np.log2(f0s[m] / np.median(f0s))
    return float(np.polyfit(ts[m] - ts[m][0], s, 1)[0])


def feats(x, sr=SR):
    nfft = 2048
    acc = np.zeros(nfft // 2 + 1)
    cnt = 0
    for i in range(0, max(1, len(x) - nfft), nfft // 4):
        seg = x[i:i + nfft]
        if len(seg) < nfft or np.sqrt((seg ** 2).mean()) < 1e-4:
            continue
        acc += np.abs(np.fft.rfft(seg * np.hanning(nfft))) ** 2
        cnt += 1
    if cnt == 0:
        return {}
    f = np.fft.rfftfreq(nfft, 1 / sr)
    P = acc / (acc.sum() + 1e-20)
    return {"cent": float((f * P).sum()),
            "body": float(P[(f >= 150) & (f < 800)].sum()) * 100,
            "b500_800": float(P[(f >= 500) & (f < 800)].sum()) * 100,
            "air": float(P[(f >= 6000) & (f < min(9000, SR / 2))].sum()) * 100}


print("=" * 104)
print("决定性评估：换参考后，听感对应的指标是否改善")
print("=" * 104)

# 真人基准
hx = load16(HUMAN, max_sec=HUMAN_END)
he = voiced_end(hx)
hn = min(int(TAIL_SEC * SR), he)
ht = hx[max(0, he - hn):he]
HT = feats(ht)
HT["slope"] = f0_slope(ht)
print(f"  真人基准（主语音末 {TAIL_SEC}s）:")
print(f"    F0斜率 {HT['slope']:+.2f}   body {HT['body']:.2f}%   "
      f"500-800 {HT['b500_800']:.2f}%   质心 {HT['cent']:.0f}   "
      f"air {HT['air']:.2f}%")

# ---------- 相似度 ----------
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
sys.path[:0] = [str(GSV), str(GSV / "GPT-SoVITS"), str(GSV / "tools")]
os.chdir(str(GSV))
from GPT_SoVITS.sv import SV  # noqa: E402

sv = SV(DEV, is_half=False)
hd = json.loads(HELD.read_text(encoding="utf-8"))
targets = [Path(t["path"]) for t in hd.get("targets", [])
           if Path(t["path"]).exists()]


def emb(p: Path):
    try:
        import torchaudio
        w, sr = torchaudio.load(str(p))
        if w.shape[0] > 1:
            w = w.mean(0, keepdim=True)
        if sr != 16000:
            w = torchaudio.functional.resample(w, sr, 16000)
        w = w.squeeze(0)
        if w.shape[0] > 160000:
            w = w[:160000]
        e = sv.compute_embedding3(w.to(DEV).unsqueeze(0).float())
        return F.normalize(e.float(), dim=-1)
    except Exception:  # noqa: BLE001
        return None


T = [e for e in (emb(p) for p in targets) if e is not None]
T = torch.stack([t.reshape(-1) for t in T])

rows = []
for tag, d, why in DIRS:
    if not d.is_dir():
        print(f"  !! 缺 {d}")
        continue
    sl, bd, mu, ce, ai, sim = [], [], [], [], [], []
    for w in sorted(d.glob("*.wav")):
        x = load16(w)
        e = voiced_end(x)
        n = min(int(TAIL_SEC * SR), e)
        tail = x[max(0, e - n):e]
        ft = feats(tail)
        s = f0_slope(tail)
        if ft and not np.isnan(s):
            sl.append(s)
            bd.append(ft["body"])
            mu.append(ft["b500_800"])
            ce.append(ft["cent"])
            ai.append(ft["air"])
        ev = emb(w)
        if ev is not None:
            sim.append(float((T @ ev.reshape(-1)).max().item()))
    rows.append({"tag": tag, "why": why, "n": len(sl),
                 "slope": float(np.mean(sl)), "body": float(np.mean(bd)),
                 "b500_800": float(np.mean(mu)), "cent": float(np.mean(ce)),
                 "air": float(np.mean(ai)),
                 "sim": float(np.mean(sim)) if sim else float("nan")})

print()
print("=" * 104)
print("结果（10 句合成音频的末段指标）")
print("=" * 104)
print(f"  {'版本':<8s} {'F0斜率':>9s} {'body':>9s} {'500-800':>9s} "
      f"{'质心':>7s} {'air':>7s} {'相似度':>9s}  说明")
print("  " + "-" * 96)
print(f"  {'真人':<8s} {HT['slope']:+9.2f} {HT['body']:8.2f}% "
      f"{HT['b500_800']:8.2f}% {HT['cent']:7.0f} {HT['air']:6.2f}% "
      f"{'—':>9s}  目标")
for r in rows:
    print(f"  {r['tag']:<8s} {r['slope']:+9.2f} {r['body']:8.2f}% "
          f"{r['b500_800']:8.2f}% {r['cent']:7.0f} {r['air']:6.2f}% "
          f"{r['sim']:9.4f}  {r['why']}")

print()
print("=" * 104)
print("相对 orig（当前参考）的变化")
print("=" * 104)
o = next((r for r in rows if r["tag"] == "orig"), None)
if o:
    print(f"  {'版本':<8s} {'ΔF0斜率':>10s} {'Δbody':>10s} {'Δ500-800':>11s} "
          f"{'Δ质心':>9s} {'Δ相似度':>10s}  判定")
    print("  " + "-" * 82)
    for r in rows:
        if r["tag"] == "orig":
            continue
        ds = r["slope"] - o["slope"]
        db = r["body"] - o["body"]
        dm = r["b500_800"] - o["b500_800"]
        dc = r["cent"] - o["cent"]
        dsim = r["sim"] - o["sim"]
        # 判定：三项都朝真人方向 + 相似度不降
        good = 0
        if abs(r["slope"] - HT["slope"]) < abs(o["slope"] - HT["slope"]):
            good += 1
        if abs(r["b500_800"] - HT["b500_800"]) < abs(o["b500_800"] - HT["b500_800"]):
            good += 1
        if abs(r["body"] - HT["body"]) < abs(o["body"] - HT["body"]):
            good += 1
        verdict = f"{good}/3 项改善" + ("，相似度↑" if dsim > 0.005
                                    else ("，相似度↓" if dsim < -0.005 else ""))
        print(f"  {r['tag']:<8s} {ds:+10.2f} {db:+9.2f}pp {dm:+10.2f}pp "
              f"{dc:+9.0f} {dsim:+10.4f}  {verdict}")

print()
print("=" * 104)
print("结论")
print("=" * 104)
if o:
    best = None
    for r in rows:
        if r["tag"] == "orig":
            continue
        sc = (abs(r["slope"] - HT["slope"]) / 5.0
              + abs(r["b500_800"] - HT["b500_800"]) / 3.0
              + abs(r["body"] - HT["body"]) / 5.0
              - max(0, r["sim"] - o["sim"]) * 10)
        if best is None or sc < best[0]:
            best = (sc, r)
    if best:
        b = best[1]
        print(f"  综合最优: **{b['tag']}**（{b['why']}）")
        print()
        print(f"  {'指标':<14s} {'真人':>9s} {'orig':>9s} {b['tag']:>9s}   改善")
        print("  " + "-" * 56)
        for k, nm, unit in (("slope", "F0斜率", ""), ("body", "body%", "%"),
                            ("b500_800", "500-800Hz%", "%"),
                            ("cent", "谱质心", ""), ("sim", "相似度", "")):
            hv = HT.get(k, float("nan"))
            ov, bv = o[k], b[k]
            better = abs(bv - hv) < abs(ov - hv) if k != "sim" else bv > ov
            mark = "✓" if better else ("~" if abs(bv - ov) < 0.01 else "✗")
            print(f"  {nm:<14s} {hv:9.2f} {ov:9.2f} {bv:9.2f}   {mark}")
        print()
        print("  ## 预期听感（相对当前）")
        print(f"    · 「往上高」: F0斜率 {o['slope']:+.2f} -> {b['slope']:+.2f}"
              f"（真人 {HT['slope']:+.2f}）")
        print(f"    · 「浊」    : 500-800 {o['b500_800']:.2f}% -> "
              f"{b['b500_800']:.2f}%（真人 {HT['b500_800']:.2f}%）")
        print(f"    · 「虚」    : body {o['body']:.2f}% -> {b['body']:.2f}%"
              f"（真人 {HT['body']:.2f}%）")
        print(f"    · 音色      : 相似度 {o['sim']:.4f} -> {b['sim']:.4f}")

# 图
fig, axes = plt.subplots(1, 5, figsize=(19, 4))
labels = ["真人"] + [r["tag"] for r in rows]
for ax, key, ttl, unit in zip(
        axes, ("slope", "b500_800", "body", "cent", "sim"),
        ("F0斜率（正=往上高）", "500-800Hz（高=浊）",
         "body（低=虚）", "谱质心", "相似度"),
        ("semi/s", "%", "%", "Hz", "")):
    vals = [HT.get(key, np.nan)] + [r[key] for r in rows]
    colors = ["tab:green"] + ["tab:orange" if r["tag"] == "orig"
                              else "tab:blue" for r in rows]
    ax.bar(labels, vals, color=colors)
    ax.axhline(HT.get(key, np.nan), color="green", ls="--", lw=1.2)
    ax.set_title(ttl, fontsize=9)
    ax.set_ylabel(unit, fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "summary.png", dpi=130)
plt.close(fig)
print()
print(f"  -> {OUT / 'summary.png'}")

rep = ["=" * 80, "换参考后的听感指标评估", "=" * 80, ""]
rep.append(f"{'版本':<8s}{'F0斜率':>10s}{'body':>10s}{'500-800':>10s}"
           f"{'质心':>8s}{'相似度':>10s}")
rep.append(f"{'真人':<8s}{HT['slope']:+10.2f}{HT['body']:9.2f}%"
           f"{HT['b500_800']:9.2f}%{HT['cent']:8.0f}{'—':>10s}")
for r in rows:
    rep.append(f"{r['tag']:<8s}{r['slope']:+10.2f}{r['body']:9.2f}%"
               f"{r['b500_800']:9.2f}%{r['cent']:8.0f}{r['sim']:10.4f}")
rep.append("")
for tag, d, why in DIRS:
    rep.append(f"{tag}: {why}")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
(OUT / "results.json").write_text(json.dumps(
    {"human": HT, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  -> {OUT / 'report.txt'}")
print("=" * 104)
