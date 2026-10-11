# -*- coding: utf-8 -*-
"""验证用户的「往上高 + 虚」—— 对应哪个声学指标，以及哪个版本过头了。

## 用户的描述
    「尾音它声音会**往上高**一点，然后呢，会**虚**一点」

## 这两个词对应的声学量（我的判断）
    「往上高」 = **谱质心偏高**（音色变亮/变薄，不是音高）
                 也可能兼有 F0 上翘
    「虚」     = **H1-H2 偏高**（声门开得久 = 气多 = 空/虚）
                 或**低频（body）不足** = 单薄

## 关键：我做了两个修复版，哪个过头了？
    v1 (`zhuo_sweep/best`)  : boom -2  mud -3  clarity +4  h1 +4
    v2 (`zhuo_sweep2/best`) : boom -2  mud -6  clarity +2  h1 +1.5  air +4

    v2 新增 `air +4dB@6000Hz`（提齿音/空气感）并加强 `mud -6`（去 body）
    -> **这两项正好会造成「更亮 + 更薄」= 「往上高 + 虚」**

## 本脚本
对 orig / v1 / v2 测「虚/亮」相关指标，看谁**超过真人**：
    ① 谱质心（越高越亮/薄）
    ② H1-H2（越高越虚/气）
    ③ 低频 body（150~800Hz 合计，越低越单薄）
    ④ F0 末段斜率（音高走向）
    ⑤ 6~9kHz 空气感（过高 = 齿音重）

**判据**：若 v2 在 ①②⑤ 上**超过真人**，就证明 v2 过头 -> 需要 v3（介于 v1/v2 之间）

## 输出
    dist/zhuo_audit/report.txt
    dist/zhuo_audit/summary.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

V5 = Path(r"D:\A\zfy-v5")
OUT = V5 / "dist" / "zhuo_audit"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ORIG = V5 / "dist" / "ref_swap_train" / "orig"
V1 = V5 / "dist" / "zhuo_sweep" / "best"
V2 = V5 / "dist" / "zhuo_sweep2" / "best"
HUMAN = V5 / "GPT-SoVITS" / "ref" / "default.wav"
HUMAN_END = 6.95
TAIL_SEC = 1.0
SR = 16000


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


def h1h2(x, sr=SR):
    vals = []
    n = 2048
    for i in range(0, max(1, len(x) - n), n // 2):
        seg = x[i:i + n]
        if len(seg) < n or np.sqrt((seg ** 2).mean()) < 1e-3:
            continue
        S = np.abs(np.fft.rfft(seg * np.hanning(n)))
        f = np.fft.rfftfreq(n, 1 / sr)
        z = seg - seg.mean()
        ac = np.correlate(z, z, "full")[len(z) - 1:]
        ac /= (ac[0] + 1e-12)
        lo, hi = int(sr / 350), min(int(sr / 70), len(ac) - 1)
        if hi <= lo:
            continue
        f0 = sr / (lo + int(np.argmax(ac[lo:hi])))
        if not (70 <= f0 <= 350):
            continue

        def pk(fr):
            m = (f > fr * 0.75) & (f < fr * 1.25)
            return S[m].max() if m.sum() else np.nan
        a_, b_ = pk(f0), pk(2 * f0)
        if np.isfinite(a_) and np.isfinite(b_) and a_ > 0 and b_ > 0:
            vals.append(20 * np.log10(a_ / b_))
    return float(np.median(vals)) if vals else float("nan")


def spec(x, sr=SR):
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
        return None, None
    f = np.fft.rfftfreq(nfft, 1 / sr)
    return f, acc / (acc.sum() + 1e-20)


def f0_slope(x, sr=SR, tail=0.4):
    """末段 F0 斜率（semi/s）。用更稳的实现：只取可靠的有声帧。"""
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
        if ac[lo + k] < 0.4:            # 提高阈值，减少八度错误
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


def audit(d: Path, label: str, files: list[Path]):
    cents, hs, bodies, sl, airs = [], [], [], [], []
    for w in files:
        p = d / w.name
        if not p.exists():
            continue
        x = load16(p)
        e = voiced_end(x)
        n = min(int(TAIL_SEC * SR), e)
        tail = x[max(0, e - n):e]
        f, P = spec(tail)
        if f is None:
            continue
        cents.append(float((f * P).sum()))
        hs.append(h1h2(tail))
        bodies.append(float(P[(f >= 150) & (f < 800)].sum()) * 100)
        airs.append(float(P[(f >= 6000) & (f < min(9000, SR / 2))].sum()) * 100)
        sl.append(f0_slope(tail))
    return {"label": label, "n": len(cents),
            "cent": float(np.nanmean(cents)),
            "h": float(np.nanmean(hs)),
            "body": float(np.nanmean(bodies)),
            "air": float(np.nanmean(airs)),
            "slope": float(np.nanmean(sl))}


print("=" * 100)
print("审计：哪个版本「往上高 + 虚」过头了")
print("=" * 100)
print("""
  对应关系：
    「往上高」-> 谱质心偏高 / 6-9kHz 空气感偏多
    「虚」    -> H1-H2 偏高（气多）/ 低频 body 不足
""")

files = sorted(ORIG.glob("*.wav"))
rows = []
# 真人
hx = load16(HUMAN, max_sec=HUMAN_END)
he = voiced_end(hx)
hn = min(int(TAIL_SEC * SR), he)
htail = hx[max(0, he - hn):he]
hf, hP = spec(htail)
rows.append({"label": "真人", "n": 1,
             "cent": float((hf * hP).sum()), "h": h1h2(htail),
             "body": float(hP[(hf >= 150) & (hf < 800)].sum()) * 100,
             "air": float(hP[(hf >= 6000) & (hf < min(9000, SR / 2))].sum()) * 100,
             "slope": f0_slope(htail)})
for d, lbl in ((ORIG, "原版"), (V1, "v1修复"), (V2, "v2修复")):
    if d.is_dir():
        rows.append(audit(d, lbl, files))

print(f"  {'版本':<10s} {'谱质心':>8s} {'H1-H2':>8s} {'body150-800':>12s} "
      f"{'air6-9k':>9s} {'F0斜率':>9s}")
print("  " + "-" * 62)
for r in rows:
    print(f"  {r['label']:<10s} {r['cent']:8.0f} {r['h']:+8.2f} "
          f"{r['body']:11.2f}% {r['air']:8.2f}% {r['slope']:+9.2f}")

print()
print("=" * 100)
print("判读：谁超过了真人（超过 = 过头 = 「往上高/虚」）")
print("=" * 100)
human = rows[0]
print(f"  {'版本':<10s} {'谱质心':>16s} {'H1-H2':>14s} {'air6-9k':>14s} "
      f"{'body':>14s}  结论")
print("  " + "-" * 92)
verdicts = {}
for r in rows[1:]:
    dc = r["cent"] - human["cent"]
    dh = r["h"] - human["h"]
    da = r["air"] - human["air"]
    db = r["body"] - human["body"]
    over = []
    if dc > human["cent"] * 0.03:
        over.append("更亮")
    if dh > 0.8:
        over.append("**更虚**")
    if da > 0.8:
        over.append("**空气感多**")
    if db < -2:
        over.append("**更薄**")
    verdict = "、".join(over) if over else "未超过"
    verdicts[r["label"]] = over
    print(f"  {r['label']:<10s} {dc:+15.0f} {dh:+14.2f} {da:+13.2f}pp "
          f"{db:+13.2f}pp  {verdict}")

print()
print("=" * 100)
print("结论")
print("=" * 100)
v2_over = verdicts.get("v2修复", [])
v1_over = verdicts.get("v1修复", [])
if v2_over:
    print(f"  **v2 过头了** —— 在 {'、'.join(v2_over)} 上超过真人")
    print()
    print("  这正好解释用户的「往上高 + 虚」：")
    print("    · 更亮 + 空气感多 -> **往上高**")
    print("    · 更虚 + 更薄     -> **虚**")
    print()
    print("  罪魁祸首：`air +4dB@6000Hz`（新增）和 `mud -6dB@650Hz`（加强）")
    print("    · air 提齿音 -> 亮/薄")
    print("    · mud -6 削掉 500-800Hz 的**胸腔共鸣** -> 单薄 = 虚")
    if v1_over:
        print(f"  但 v1 也在 {'、'.join(v1_over)} 上略超")
    else:
        print("  ✓ v1 未超过真人（更保守）")
elif v1_over:
    print(f"  **v1 过头** —— {'、'.join(v1_over)}")
else:
    print("  两个版本都未超过真人 —— 「往上高 + 虚」可能来自**原版**（模型本身）")
    print("  检查原版是否已超：")
    o = [r for r in rows if r["label"] == "原版"][0]
    print(f"    原版 vs 真人: 质心 {o['cent']-human['cent']:+.0f}  "
          f"H1-H2 {o['h']-human['h']:+.2f}  "
          f"air {o['air']-human['air']:+.2f}pp  "
          f"body {o['body']-human['body']:+.2f}pp")

# 图
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
labels = [r["label"] for r in rows]
for ax, key, ttl, unit in zip(
        axes,
        ("cent", "h", "air", "body"),
        ("谱质心（越高越亮/薄）", "H1-H2（越高越虚）",
         "6-9kHz 空气感 %", "body 150-800Hz %"),
        ("Hz", "dB", "%", "%")):
    vals = [r[key] for r in rows]
    colors = ["tab:green"] + ["tab:red" if r["label"].startswith("v2")
                              else "tab:orange" if r["label"].startswith("v1")
                              else "tab:gray" for r in rows[1:]]
    ax.bar(labels, vals, color=colors)
    ax.axhline(human[key], color="green", ls="--", lw=1.2, label="真人")
    ax.set_title(ttl, fontsize=9)
    ax.set_ylabel(unit, fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "summary.png", dpi=130)
plt.close(fig)
print()
print(f"  -> {OUT / 'summary.png'}")

rep = ["=" * 80, "审计：哪个版本过头", "=" * 80, ""]
rep.append(f"{'版本':<10s}{'质心':>9s}{'H1-H2':>9s}{'body':>10s}{'air':>9s}{'F0斜率':>9s}")
for r in rows:
    rep.append(f"{r['label']:<10s}{r['cent']:9.0f}{r['h']:+9.2f}"
               f"{r['body']:9.2f}%{r['air']:8.2f}%{r['slope']:+9.2f}")
rep.append("")
for k, v in verdicts.items():
    rep.append(f"{k}: {'、'.join(v) if v else '未超过真人'}")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
print(f"  -> {OUT / 'report.txt'}")
print("=" * 100)
