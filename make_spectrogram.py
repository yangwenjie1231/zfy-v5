# -*- coding: utf-8 -*-
"""生成频谱图，用视觉直接判断「尾音浊」。

## 为什么换方法
逐帧标量指标噪声太大：
    · 模型 5.30s vs 真人 7.95s（**语速差 38%**）
    -> 归一化时间轴对齐后，同一 % 位置对应的是**不同音素**
    -> 谱质心 1740/5726、1/5118 这种跳跃完全是错位造成的

**标量指标在时序错位时失效。频谱图不会** —— 能直接看出：
    · 共振峰是否清晰（横条纹分明 = 清晰；糊成一片 = 浊）
    · 辅音是否有爆破/摩擦段（竖条纹）
    · 浊音是否拖尾（低频能量条拖长）
    · 句末是否"塌陷"（高频消失，只剩低频轰鸣）

## 输出
    dist/spec/
        06_model.png        模型输出全句
        06_human.png        真人全句（同文本）
        05_tail_zoom.png    两者末段并排放大
        NN_*.png            其他句
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

V5 = Path(r"D:\A\zfy-v5")
OUT = V5 / "dist" / "spec"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_DIR = V5 / "dist" / "ref_swap_train" / "orig"
HUMAN_REF = V5 / "GPT-SoVITS" / "ref" / "default.wav"
HELD = V5 / "dist" / "heldout" / "heldout_clean.json"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load16(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != 16000:
        try:
            from math import gcd

            from scipy.signal import resample_poly
            g = gcd(int(sr), 16000)
            x = resample_poly(x, 16000 // g, int(sr) // g)
        except Exception:  # noqa: BLE001
            n = int(round(len(x) * 16000 / sr))
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
        sr = 16000
    return np.asarray(x, dtype=np.float64), sr


def spec(ax, x, sr, title, fmax=8000):
    ax.specgram(x, NFFT=512, Fs=sr, noverlap=384, cmap="magma",
                vmin=-120, vmax=-20)
    ax.set_ylim(0, fmax)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Hz", fontsize=8)
    ax.tick_params(labelsize=7)


def voiced_span(x, sr, thresh_db=-40.0):
    n = int(0.01 * sr)
    m = (len(x) // n) * n
    r = np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))
    ref = r.max() + 1e-12
    v = np.where(r > ref * (10 ** (thresh_db / 20)))[0]
    return (int(v[0]) * n, int(v[-1] + 1) * n) if len(v) else (0, len(x))


print("=" * 96)
print("生成频谱图")
print("=" * 96)

# ---------- 1) 同文本对比（最有价值）----------
# ⚠️ 必须是 **05.wav**！
#    `compare_ref_swap.py` 的 TEXTS 是 0-based，索引 4 -> 05.wav
#    = "再好的武器…好吗？"，与 `ref/default.wav` **同文本**。
#    我第一版用了 06.wav（"武陵城外有一片梨树林…"）—— 两段完全不同的语音，
#    所有指标和频谱图全部无效（"语速差 38%" 是假的）。
mp = MODEL_DIR / "05.wav"
if mp.exists() and HUMAN_REF.exists():
    xm, sr = load16(mp)
    xh, _ = load16(HUMAN_REF)
    a, b = voiced_span(xm, sr)
    c, d = voiced_span(xh, sr)
    xm_v, xh_v = xm[a:b], xh[c:d]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    spec(axes[0], xm_v, sr,
         f"模型合成 (有声 {len(xm_v)/sr:.2f}s)  — 文本: 再好的武器…好吗？")
    spec(axes[1], xh_v, sr,
         f"真人参考 (有声 {len(xh_v)/sr:.2f}s)  — 同一句话")
    axes[1].set_xlabel("秒", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "05_compare.png", dpi=130)
    plt.close(fig)
    print(f"  -> 05_compare.png   模型 {len(xm_v)/sr:.2f}s vs 真人 {len(xh_v)/sr:.2f}s")

    # 末段放大（各取最后 1.2s）
    tail_sec = 1.2
    nt = int(tail_sec * sr)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    spec(axes[0], xm_v[-nt:], sr, f"模型 末 {tail_sec}s（放大）", fmax=6000)
    spec(axes[1], xh_v[-nt:], sr, f"真人 末 {tail_sec}s（放大）", fmax=6000)
    axes[1].set_xlabel("秒（相对末段起点）", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "05_tail_zoom.png", dpi=150)
    plt.close(fig)
    print(f"  -> 05_tail_zoom.png  末 {tail_sec}s 放大对比")

    # 逐帧能量对比（看尾部包络形状）
    fig, ax = plt.subplots(figsize=(12, 3.5))
    for x, lbl in ((xm_v, "模型"), (xh_v, "真人")):
        n = int(0.01 * sr)
        m = (len(x) // n) * n
        r = np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))
        r = r / (r.max() + 1e-12)
        t = np.arange(len(r)) * 0.01
        ax.plot(t, 20 * np.log10(r + 1e-6), label=lbl, lw=1.2)
    ax.set_xlabel("秒", fontsize=8)
    ax.set_ylabel("dB", fontsize=8)
    ax.set_title("能量包络对比（看末尾如何收）", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(-60, 3)
    fig.tight_layout()
    fig.savefig(OUT / "05_envelope.png", dpi=130)
    plt.close(fig)
    print(f"  -> 05_envelope.png   能量包络")

# ---------- 2) 其他句 ----------
for i in (1, 9, 11, 20):
    p = MODEL_DIR / f"{i:02d}.wav"
    if not p.exists():
        continue
    x, sr = load16(p)
    a, b = voiced_span(x, sr)
    xv = x[a:b]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6),
                             gridspec_kw={"height_ratios": [2, 1]})
    spec(axes[0], xv, sr, f"模型 {i:02d}  有声 {len(xv)/sr:.2f}s")
    nt = int(0.8 * sr)
    spec(axes[1], xv[-nt:], sr, "末 0.8s 放大", fmax=6000)
    axes[1].set_xlabel("秒", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{i:02d}_model.png", dpi=120)
    plt.close(fig)
    print(f"  -> {i:02d}_model.png")

# ---------- 3) 真人（留出集）参考 ----------
import json  # noqa: E402
if HELD.exists():
    hd = json.loads(HELD.read_text(encoding="utf-8"))
    for j, it in enumerate(hd.get("all", [])[:2], 1):
        p = Path(it["path"])
        if not p.exists():
            continue
        x, sr = load16(p)
        a, b = voiced_span(x, sr)
        xv = x[a:b]
        fig, axes = plt.subplots(2, 1, figsize=(13, 6),
                                 gridspec_kw={"height_ratios": [2, 1]})
        spec(axes[0], xv, sr, f"真人 {p.stem[:36]}  有声 {len(xv)/sr:.2f}s")
        nt = min(int(0.8 * sr), len(xv))
        spec(axes[1], xv[-nt:], sr, "末 0.8s 放大", fmax=6000)
        axes[1].set_xlabel("秒", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"human{j}.png", dpi=120)
        plt.close(fig)
        print(f"  -> human{j}.png")

print()
print("=" * 96)
print("怎么看频谱图")
print("=" * 96)
print("""
  ## 横轴 = 时间，纵轴 = 频率（Hz），亮度 = 能量

  ## 「浊」在图上的样子
    · **共振峰糊成一片**：正常语音在 0~4kHz 有清晰的横向条纹
      （每条 = 一个共振峰）。浊 = 条纹边界模糊、连成一块。
    · **低频轰鸣拖尾**：<500Hz 的亮带在句末**拖得很长**。
    · **高频塌陷**：4kHz 以上的能量在句末**过早消失**。
    · **辅音缺失**：句末应有短促的竖直能量条（爆破/摩擦），
      浊 = 没有或很弱。

  ## 对比要点
    `05_compare.png`  上下对比同一句话 —— **看末尾 1/4 的差异**
    `05_tail_zoom.png` 末 1.2s 放大 —— **最清楚**
    `05_envelope.png`  能量包络 —— 看末尾是"陡降"还是"缓拖"

  ## 关键问题
    模型末尾的**共振峰条纹**是否比真人更模糊？
    模型末尾的**低频亮带**是否拖得更长？
""")
print("=" * 96)
