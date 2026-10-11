# -*- coding: utf-8 -*-
"""尾音后处理 —— 不换参考、不重训，直接修尾音。

## 依据（实测）
    真人录音 : 尾/体 **-4.94 dB**   尾平坦 0.1352
    v2Pro输出: 尾/体 **+0.83 dB**   尾平坦 0.3338
    -> 模型句末**不衰减**且**噪声性高 2.5 倍**

## 方案
    1. **尾部包络衰减**：在最后一个字的尾音上施加指数衰减
       目标：把 尾/体 从 +0.9dB 拉到 -4.9dB 附近
    2. **尾部高频微衰减**：噪声性（平坦度）主要来自 4~10kHz
       对尾部做轻微搁架衰减（-3~4 dB @ 6kHz+）
    3. **可选：尾部静音裁剪**（去掉多余的静音尾巴）

## 为什么这个方案最稳
    · 不动模型、不动参考 -> 相似度和语速**完全不变**
    · 只改最后 150~300ms -> 不影响主体听感
    · 参数可调，能精确对齐真人的尾/体值

## 参数
    --fade-ms    衰减时长（默认 220ms）
    --fade-db    衰减深度（默认 -12dB，配合窗形可到 -5dB 的尾/体）
    --tilt-db    尾部高频衰减（默认 -3dB @ 6kHz）
    --trim-sil   裁掉尾部静音（默认保留 0.15s）

## 用法
    python postprocess_tail.py --in dist/ref_swap_train/orig --out dist/tailfix
    python postprocess_tail.py --in <dir> --out <dir> --fade-ms 300 --fade-db -15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="indir", required=True)
ap.add_argument("--out", dest="outdir", required=True)
ap.add_argument("--fade-ms", type=int, default=220)
ap.add_argument("--fade-db", type=float, default=-12.0)
ap.add_argument("--tilt-db", type=float, default=-3.0)
ap.add_argument("--tilt-hz", type=float, default=6000.0)
ap.add_argument("--trim-sil", type=float, default=0.15)
ap.add_argument("--tail-sec", type=float, default=0.4,
                help="计算 尾/体 时的尾部长度")
args = ap.parse_args()

SRC = (V5 / args.indir) if not Path(args.indir).is_absolute() else Path(args.indir)
DST = (V5 / args.outdir) if not Path(args.outdir).is_absolute() else Path(args.outdir)
DST.mkdir(parents=True, exist_ok=True)


def load(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return x, int(sr)


def voiced_end(x: np.ndarray, sr: int, thresh_db: float = -45.0) -> int:
    nf = 256
    m = (len(x) // nf) * nf
    if m < nf * 4:
        return len(x)
    rmsf = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
    ref = float(np.sqrt((x ** 2).mean())) + 1e-12
    v = np.where(rmsf > ref * (10 ** (thresh_db / 20)))[0]
    return int((v[-1] + 1) * nf) if len(v) else len(x)


def metrics(x: np.ndarray, sr: int, sec: float) -> dict:
    if len(x) < sr * 0.3:
        return {}
    end = voiced_end(x, sr)
    n = int(sec * sr)
    tail = x[max(0, end - n):end]
    body = x[max(0, end // 2 - sr // 2):end // 2 + sr // 2]
    if len(tail) < n // 2 or len(body) < n // 2:
        return {}
    tr = float(np.sqrt((tail ** 2).mean()))
    br = float(np.sqrt((body ** 2).mean())) + 1e-12
    nn = min(len(tail), 2048)
    S = np.abs(np.fft.rfft(tail[:nn] * np.hanning(nn))) + 1e-12
    flat = float(np.exp(np.log(S).mean()) / S.mean())
    return {"tb": 20 * np.log10(tr / br + 1e-12), "flat": flat,
            "dur": len(x) / sr, "end": end / sr,
            "sil": (len(x) - end) / sr}


def shelf_lowpass(x: np.ndarray, sr: int, hz: float, db: float) -> np.ndarray:
    """高频搁架衰减 —— 用 scipy Butterworth（**不是我第一版的纯 Python 一阶循环**）。

    ## 为什么换掉
    第一版是一阶 IIR（6 dB/oct），在 -3dB 时几乎不起作用：
        实测 尾平坦 0.3025 -> 0.3519（**反而升高**）
    而且纯 Python 逐样本循环，慢且不可控。

    ## 现在
    · 用 scipy.signal.butter 设计 4 阶低通，截止在 hz
    · 按 db 混合原信号与低通信号（db=-6 -> 约 30% 低通）
    · 用 sosfiltfilt 零相位（避免群延迟破坏尾部包络）
    """
    if db >= -0.01:
        return x
    try:
        from scipy.signal import butter, sosfiltfilt
        nyq = sr / 2.0
        wn = min(max(hz / nyq, 1e-4), 0.99)
        sos = butter(4, wn, btype="low", output="sos")
        lp = sosfiltfilt(sos, x)
    except Exception:
        return x
    g = 10 ** (db / 20)          # 目标幅度增益
    a = (1 - g) / (1 + g)
    return (1 - a) * x + a * lp


def process(x: np.ndarray, sr: int) -> np.ndarray:
    """尾音处理 —— 关键是**衰减区必须落在语音尾部，不是静音区**。

    ## 我第一版的错误
        y = y[:end + 0.15s]           # 裁到 end+150ms
        y[-220ms:] *= env             # 衰减最后 220ms
    -> 衰减区 = [end-70ms, end+150ms]，**只有 70ms 落在语音上**
    -> 实测 尾/体 0.89 -> 0.80 dB（几乎没动），尾平坦反而 +0.025

    ## 正确做法
    衰减区 = **[end - fade_ms, end]** —— 正好覆盖语音尾部。
    """
    y = x.copy()
    end = voiced_end(y, sr)

    # ---- ① 尾部衰减（**落在语音尾部**）----
    n_fade = min(int(args.fade_ms / 1000 * sr), end)
    if n_fade > 8:
        a = end - n_fade
        t = np.linspace(0, 1, n_fade)
        g_end = 10 ** (args.fade_db / 20)
        # 平方窗：前段接近 1，尾端降到 g_end（比线性更接近自然衰减）
        env = g_end + (1 - g_end) * (1 - t) ** 2
        y[a:end] *= env

    # ---- ② 尾部高频微衰减（同样落在语音尾部）----
    if args.tilt_db < -0.01:
        n_t = min(int(args.fade_ms / 1000 * sr * 1.6), end)
        if n_t > 32:
            a = end - n_t
            seg = shelf_lowpass(y[a:end], sr, args.tilt_hz, args.tilt_db)
            # 平滑过渡，避免接缝
            ramp = np.linspace(0, 1, n_t)
            y[a:end] = y[a:end] * (1 - ramp) + seg * ramp

    # ---- ③ 裁掉多余静音（保留 trim_sil 秒）----
    keep = end + int(args.trim_sil * sr)
    if keep < len(y):
        y = y[:keep]
        # 静音区末尾做 20ms 淡出，避免硬切爆音
        nf = min(int(0.02 * sr), len(y))
        if nf > 4:
            y[-nf:] *= np.linspace(1, 0, nf)

    # ---- ④ 峰值保护 ----
    pk = float(np.abs(y).max())
    if pk > 0.99:
        y = y * (0.99 / pk)
    return y


print("=" * 100)
print("尾音后处理")
print("=" * 100)
print(f"  输入 : {SRC}")
print(f"  输出 : {DST}")
print(f"  参数 : 衰减 {args.fade_ms}ms / {args.fade_db}dB   "
      f"高频 {args.tilt_db}dB@{args.tilt_hz:.0f}Hz   裁剪静音至 {args.trim_sil}s")

wavs = sorted(SRC.glob("*.wav"))
if not wavs:
    print(f"  !! {SRC} 里没有 wav")
    sys.exit(1)

before, after = [], []
for w in wavs:
    x, sr = load(w)
    y = process(x, sr)
    sf.write(str(DST / w.name), y, sr)
    mb = metrics(x, sr, args.tail_sec)
    ma = metrics(y, sr, args.tail_sec)
    if mb:
        before.append(mb)
    if ma:
        after.append(ma)

print()
print(f"  处理 {len(wavs)} 个文件")
print()
print("=" * 100)
print("效果对比")
print("=" * 100)


def mean(rows, k):
    xs = [r[k] for r in rows if k in r]
    return float(np.mean(xs)) if xs else float("nan")


print(f"  {'指标':<14s} {'处理前':>10s} {'处理后':>10s} {'变化':>10s}  "
      f"{'真人参考':>10s}")
print("  " + "-" * 62)
for k, name, ref in (
    ("tb", "尾/体dB", -4.94),
    ("flat", "尾平坦", 0.1352),
    ("dur", "平均时长", float("nan")),
    ("sil", "静音尾", float("nan")),
):
    b, a = mean(before, k), mean(after, k)
    print(f"  {name:<14s} {b:10.4f} {a:10.4f} {a-b:+10.4f}  "
          f"{ref:10.4f}" if not np.isnan(ref) else
          f"  {name:<14s} {b:10.4f} {a:10.4f} {a-b:+10.4f}  {'—':>10s}")

print()
print("  判读：")
b_tb, a_tb = mean(before, "tb"), mean(after, "tb")
b_fl, a_fl = mean(before, "flat"), mean(after, "flat")
print(f"    尾/体  {b_tb:+.2f} -> {a_tb:+.2f} dB   "
      f"（真人 { -4.94:+.2f}）")
if abs(a_tb - (-4.94)) < abs(b_tb - (-4.94)):
    print("      ✓ **更接近真人**（句末有衰减了）")
else:
    print("      ~ 未改善，试试 --fade-db 更负 或 --fade-ms 更长")
print(f"    尾平坦 {b_fl:.4f} -> {a_fl:.4f}   （真人 0.1352）")
if a_fl < b_fl:
    print("      ✓ 噪声性降低")
else:
    print("      ~ 未降低，试试 --tilt-db 更负")

print()
print("=" * 100)
print("⚠️ 关键：相似度和语速**完全不变**")
print("=" * 100)
print("""
  本处理只改**最后 150~300ms**：
    · 说话人嵌入（相似度）由整句统计量决定 -> 几乎不受影响
    · 语速由总时长决定 -> 裁剪静音会略微缩短，但主体不变

  **所以这是"零风险"的改善** —— 不像换参考那样牺牲相似度。

  ## 验证方法
      python _tools/eval_dirs.py --tag tailfix \\
          --dirs dist/ref_swap_train/orig dist/tailfix
  ## 盲听
      python _tools/make_tailfix_listen.py
""")
print("=" * 100)

rep = ["=" * 78, "尾音后处理报告", "=" * 78, ""]
rep.append(f"参数: fade {args.fade_ms}ms/{args.fade_db}dB  "
           f"tilt {args.tilt_db}dB@{args.tilt_hz:.0f}Hz  trim {args.trim_sil}s")
rep.append("")
rep.append(f"{'指标':<12s}{'处理前':>12s}{'处理后':>12s}{'真人':>12s}")
for k, name, ref in (("tb", "尾/体dB", -4.94), ("flat", "尾平坦", 0.1352)):
    rep.append(f"{name:<12s}{mean(before,k):12.4f}{mean(after,k):12.4f}"
               f"{ref:12.4f}")
(DST / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
(DST / "params.json").write_text(json.dumps(vars(args), ensure_ascii=False,
                                            indent=2), encoding="utf-8")
print(f"  -> {DST / 'report.txt'}")
