# -*- coding: utf-8 -*-
"""解剖 ref/default.wav 的尾音 —— 看那段"气声"到底是什么。

## 已知
    尾/体 +29.83 dB   <- 最后 0.4s 的 RMS 比主体高 30 dB
    尾平坦 0.5619     <- 谱平坦度极高（白噪声是 1.0，纯音接近 0）
    平坦比 5.244      <- 尾部比主体"噪" 5 倍

**+30 dB 且平坦度 0.56** —— 这不是"气声"，更像是**一整段噪声/静音底噪**。

## 要查
  1. 尾部 1.5s 的**逐帧 RMS**（看是不是突然抬升）
  2. 尾部的**频谱形状**（白噪声？还是集中在某频段？）
  3. 文件**总时长 7.95s** 与"有声部分"的对比
  4. **是否 DC 偏移** / 是否有爆音
  5. 与"真正干净"的参考候选对比

## 为什么重要
如果 `ref/default.wav` 尾部有一段噪声：
  · 模型的 `ge` 风格向量会**编码这段噪声**
  · 合成音频会**继承**它
  · **换一个干净的参考音频**就可能解决"尾音气声"

## 输出
    dist/ref_tail_anatomy.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
REF = V5 / "GPT-SoVITS" / "ref" / "default.wav"
REFTXT = V5 / "GPT-SoVITS" / "ref" / "ref.txt"
# 候选：训练集里的干净素材（生产时挑过相似度）
CAND_DIRS = [
    V5 / "GPT-SoVITS" / "logs" / "zfy_v5" / "5-wav32k",
    Path(r"D:\A\voiceclone\data\zfh\wav"),
]


def load(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return x, int(sr)


print("=" * 96)
print("ref/default.wav 尾部解剖")
print("=" * 96)

if not REF.exists():
    print(f"  !! 找不到 {REF}")
    sys.exit(1)

x, sr = load(REF)
print(f"  文件     : {REF}")
print(f"  时长     : {len(x)/sr:.3f}s   ({len(x)} 样本 @ {sr} Hz)")
print(f"  总 RMS   : {np.sqrt((x**2).mean()):.5f}")
print(f"  峰值     : {np.abs(x).max():.4f}")
print(f"  DC 偏移  : {x.mean():+.6f}")
if REFTXT.exists():
    t = REFTXT.read_text(encoding="utf-8").strip()
    print(f"  参考文本 : {t}")

# ---------- 逐帧 RMS（0.1s 帧）----------
print()
print("=" * 96)
print("逐帧 RMS（0.1s 帧）—— 看尾部是否有异常抬升")
print("=" * 96)
nf = int(0.1 * sr)
m = (len(x) // nf) * nf
frames = x[:m].reshape(-1, nf)
rms = np.sqrt((frames ** 2).mean(axis=1))
peak = np.abs(frames).max(axis=1)
# 用 dB 显示（相对整体 RMS）
ref_rms = float(np.sqrt((x ** 2).mean())) + 1e-12
db = 20 * np.log10(np.maximum(rms, 1e-12) / ref_rms)
print(f"  共 {len(rms)} 帧")
print()
print("  前 10 帧:")
for i in range(min(10, len(rms))):
    bar = "#" * int(max(0, db[i] + 40) / 2)
    print(f"    {i*0.1:5.1f}s  {rms[i]:.5f}  {db[i]:+6.1f} dB  {bar}")
print("  ...")
print("  后 20 帧（**重点**）:")
start = max(0, len(rms) - 20)
for i in range(start, len(rms)):
    bar = "#" * int(max(0, db[i] + 40) / 2)
    flag = "  <<< 异常高" if db[i] > 0 else ""
    print(f"    {i*0.1:5.1f}s  {rms[i]:.5f}  {db[i]:+6.1f} dB  {bar}{flag}")

# ---------- 尾部频谱形状 ----------
print()
print("=" * 96)
print("尾部 0.4s 的频谱形状（判断是白噪声还是集中频段）")
print("=" * 96)
tail = x[-int(0.4 * sr):]
body = x[int(len(x) * 0.4):int(len(x) * 0.4) + int(0.4 * sr)]
for name, s in (("尾部 0.4s", tail), ("主体 0.4s", body)):
    n = min(len(s), 4096)
    S = np.abs(np.fft.rfft(s[:n] * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1 / sr)
    S = S / (S.sum() + 1e-12)
    bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000),
             (4000, 8000), (8000, 16000), (16000, 24000)]
    print(f"\n  {name}:")
    for lo, hi in bands:
        if lo >= sr / 2:
            continue
        e = S[(f >= lo) & (f < min(hi, sr / 2))].sum()
        bar = "#" * int(e * 200)
        print(f"    {lo:5d}~{hi:5d} Hz : {e*100:5.1f}%  {bar}")

# ---------- 与候选参考对比 ----------
print()
print("=" * 96)
print("候选参考音频对比（找更干净的）")
print("=" * 96)
print(f"  {'文件':<44s} {'时长':>6s} {'总RMS':>8s} {'尾0.4s':>8s} "
      f"{'尾/体dB':>8s} {'尾平坦':>7s} {'DC':>9s}")
print("  " + "-" * 92)


def flatness(s: np.ndarray) -> float:
    n = min(len(s), 2048)
    if n < 64:
        return float("nan")
    S = np.abs(np.fft.rfft(s[:n] * np.hanning(n))) + 1e-12
    return float(np.exp(np.log(S).mean()) / S.mean())


def stats(p: Path) -> dict:
    y, srr = load(p)
    if len(y) < srr * 0.5:
        return {}
    tail = y[-int(0.4 * srr):]
    body = y[int(len(y) * 0.4):int(len(y) * 0.4) + int(0.4 * srr)]
    tr = float(np.sqrt((tail ** 2).mean()))
    br = float(np.sqrt((body ** 2).mean())) + 1e-12
    return {"dur": len(y) / srr, "rms": float(np.sqrt((y ** 2).mean())),
            "tail": tr, "tb": 20 * np.log10(tr / br + 1e-12),
            "flat": flatness(tail), "dc": float(y.mean())}


print(f"  {REF.name:<44s} {len(x)/sr:5.2f}s {np.sqrt((x**2).mean()):8.5f} "
      f"{np.sqrt((tail**2).mean()):8.5f} "
      f"{20*np.log10(np.sqrt((tail**2).mean())/(np.sqrt((body**2).mean())+1e-12)):+8.2f} "
      f"{flatness(tail):7.4f} {x.mean():+9.6f}")

# 找训练集里"尾部干净"的候选
best = []
for d in CAND_DIRS:
    if not d.is_dir():
        continue
    for p in sorted(d.glob("*.wav"))[:400]:
        s = stats(p)
        if not s:
            continue
        # 干净判据：尾部不异常抬升（tb < +3dB）且平坦度低（<0.2）
        if s["tb"] < 3.0 and s["flat"] < 0.25 and s["dur"] > 3.0:
            best.append((s["tb"], s["flat"], p, s))
best.sort(key=lambda t: (t[0], t[1]))
print()
print(f"  训练集里「尾部干净」的候选（前 12 / 共 {len(best)}）:")
for tb, fl, p, s in best[:12]:
    print(f"  {p.name[:44]:<44s} {s['dur']:5.2f}s {s['rms']:8.5f} "
          f"{s['tail']:8.5f} {tb:+8.2f} {fl:7.4f} {s['dc']:+9.6f}")

print()
print("=" * 96)
print("判读")
print("=" * 96)
print("""
  ## 关键看「尾/体dB」
    · 接近 0 或负值 -> 尾部比主体轻（正常收尾）
    · **正值很大（如 +30dB）-> 尾部有一段比语音还响的东西**
      可能是：气声、噪声、爆音、或录音末尾的杂音

  ## 关键看「尾平坦」
    · <0.2  -> 谐波性强（正常语音收尾）
    · >0.5  -> 噪声性强（气声/白噪）

  ## 如果 ref/default.wav 确实脏
    换一个"尾部干净"的候选，重跑合成即可 ——
    **不需要重训模型**（参考音频是推理时参数）。

    挑候选的方法（技能里记录的）：
      · 时长 3~10s
      · 干净、无 BGM
      · **尾部平坦度低**（本次新增的判据）
      · 然后用说话人相似度筛（技能：21 个候选里 006_更换武器 最优 0.6459）
""")
print("=" * 96)

out = V5 / "dist" / "ref_tail_anatomy.txt"
lines = ["=" * 78, "ref/default.wav 尾部解剖", "=" * 78, ""]
lines.append(f"时长 {len(x)/sr:.3f}s @ {sr}Hz")
lines.append(f"总RMS {np.sqrt((x**2).mean()):.5f}  峰值 {np.abs(x).max():.4f}  "
             f"DC {x.mean():+.6f}")
lines.append("")
lines.append("后 20 帧（0.1s）RMS:")
for i in range(start, len(rms)):
    lines.append(f"  {i*0.1:5.1f}s  {rms[i]:.5f}  {db[i]:+6.1f} dB")
lines.append("")
lines.append(f"尾部0.4s 平坦度 {flatness(tail):.4f}   主体0.4s 平坦度 "
             f"{flatness(body):.4f}")
lines.append("")
lines.append("尾部干净候选（前 12）:")
for tb, fl, p, s in best[:12]:
    lines.append(f"  {p.name}  dur={s['dur']:.2f}s tb={tb:+.2f}dB flat={fl:.4f}")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"  -> {out}")
