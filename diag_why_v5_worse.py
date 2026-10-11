# -*- coding: utf-8 -*-
"""诊断：v5 全量微调若不如 v2Pro，可能的原因 —— 重点查音量归一化。

## 用户的问题
  1. 如果 v5 测出来不如 v2Pro，会是什么原因？
  2. 当时给数据做音量归一化会有影响吗？

## 已知
  · 数据集 manifest.csv 有这些列：
      增益dB / 限幅dB / 峰值 / 削顶样本 / 底噪等级 / 底噪dBFS /
      语音dBFS / SNR / 谱平坦度
    -> 确实做过**逐条增益归一化 + 限幅**
  · v2Pro 的训练音频还在：D:\\A\\voiceclone\\GPT-SoVITS\\logs\\zfh\\5-wav32k
    -> 可以直接和 v5 的 5-wav32k 对比响度分布
  · 生产参考音频：D:\\A\\voiceclone\\zfh-models-v2pro\\ref\\default.wav

## 本脚本实测
  1. manifest 里增益/峰值/底噪/SNR 的分布
  2. v5 语料 vs v2Pro 语料的实际响度统计
  3. 参考音频的响度（推理时的"锚点"）
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DS = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频\庄方宜_TTS训练集")
V5_WAV = Path(r"D:\A\zfy-v5\GPT-SoVITS\logs\zfy_v5\5-wav32k")
V2_WAV = Path(r"D:\A\voiceclone\GPT-SoVITS\logs\zfh\5-wav32k")
REF = Path(r"D:\A\voiceclone\zfh-models-v2pro\ref\default.wav")


def dbfs(x: float) -> float:
    return 20 * np.log10(max(x, 1e-12))


def loudness_stats(p: Path, frame_ms: float = 25.0):
    """返回 (整体RMS dBFS, 语音段RMS dBFS, 峰值 dBFS, 底噪 dBFS, 波峰因数 dB)

    语音段/底噪用 25ms 帧能量分位数近似：
      语音段 = 能量在 60~95 分位的帧（避开静音与爆音）
      底噪   = 能量在 5~20 分位的帧
    """
    try:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001
        return None
    x = x.mean(axis=1)
    if x.size == 0:
        return None
    n = max(1, int(sr * frame_ms / 1000))
    usable = (x.size // n) * n
    if usable < n * 4:
        return None
    fr = x[:usable].reshape(-1, n)
    e = (fr ** 2).mean(axis=1) + 1e-12
    rms_all = float(np.sqrt(e.mean()))
    q = np.percentile(e, [5, 20, 60, 95])
    speech = float(np.sqrt(e[(e >= q[2]) & (e <= q[3])].mean())) \
        if ((e >= q[2]) & (e <= q[3])).any() else float(np.sqrt(e.mean()))
    floor = float(np.sqrt(e[(e >= q[0]) & (e <= q[1])].mean())) \
        if ((e >= q[0]) & (e <= q[1])).any() else 1e-6
    peak = float(np.abs(x).max())
    return (dbfs(rms_all), dbfs(speech), dbfs(peak), dbfs(floor))


print("=" * 80)
print("1. manifest.csv 的增益/峰值/底噪/SNR 分布")
print("=" * 80)
mf = DS / "manifest.csv"
if mf.exists():
    rows = list(csv.DictReader(mf.open(encoding="utf-8-sig")))
    print(f"  总行数: {len(rows)}")

    def nums(col):
        out = []
        for r in rows:
            v = (r.get(col) or "").strip()
            if not v:
                continue
            try:
                out.append(float(v))
            except ValueError:
                pass
        return out

    for col, unit in (("增益dB", "dB"), ("峰值", ""), ("限幅dB", "dB"),
                      ("底噪dBFS", "dBFS"), ("语音dBFS", "dBFS"),
                      ("SNR", "dB"), ("谱平坦度", ""), ("削顶样本", "")):
        v = nums(col)
        if not v:
            print(f"  {col:10s} 无有效数值")
            continue
        v.sort()
        print(f"  {col:10s} n={len(v):4d}  最小 {v[0]:8.2f}  "
              f"P10 {v[len(v)//10]:8.2f}  中位 {v[len(v)//2]:8.2f}  "
              f"P90 {v[len(v)*9//10]:8.2f}  最大 {v[-1]:8.2f}  [{unit}]")

    # 「处理」列看做了什么操作
    from collections import Counter
    print()
    print("  「处理」列取值分布（前 10）:")
    for k, c in Counter((r.get("处理") or "").strip() for r in rows).most_common(10):
        print(f"    {c:5d}  {k[:70]}")
else:
    print(f"  找不到 {mf}")

print()
print("=" * 80)
print("2. 实际响度对比：v5 语料 vs v2Pro 语料 vs 参考音频")
print("=" * 80)


def survey(name: str, d: Path, limit: int | None = None):
    if not d.is_dir():
        print(f"\n--- {name} ---\n  目录不存在: {d}")
        return None
    wavs = sorted(f for f in d.rglob("*.wav") if f.is_file())
    if limit:
        step = max(1, len(wavs) // limit)
        wavs = wavs[::step][:limit]
    print(f"\n--- {name} ---")
    print(f"  目录: {d}")
    print(f"  采样 {len(wavs)} 个")
    res = []
    for w in wavs:
        r = loudness_stats(w)
        if r:
            res.append(r)
    if not res:
        print("  无有效样本")
        return None
    a = np.array(res)
    labels = ["整体RMS", "语音段RMS", "峰值", "底噪"]
    for i, lab in enumerate(labels):
        col = a[:, i]
        print(f"  {lab:8s} 中位 {np.median(col):8.2f} dBFS  "
              f"P10 {np.percentile(col,10):8.2f}  "
              f"P90 {np.percentile(col,90):8.2f}  "
              f"标准差 {col.std():5.2f}")
    crest = a[:, 2] - a[:, 1]
    print(f"  {'波峰因数':8s} 中位 {np.median(crest):8.2f} dB    "
          f"（峰值 − 语音段RMS，越大说明动态越强）")
    snr = a[:, 1] - a[:, 3]
    print(f"  {'SNR估算':8s} 中位 {np.median(snr):8.2f} dB")
    return a


r5 = survey("v5 语料（本次全量微调用）", V5_WAV, limit=250)
r2 = survey("v2Pro 语料（生产用）", V2_WAV)
rr = survey("生产参考音频 default.wav", REF.parent)

print()
print("=" * 80)
print("3. 差异汇总")
print("=" * 80)
if r5 is not None and r2 is not None:
    l5, l2 = np.median(r5, axis=0), np.median(r2, axis=0)
    labels = ["整体RMS", "语音段RMS", "峰值", "底噪"]
    print(f"  {'指标':10s} {'v5':>10s} {'v2Pro':>10s} {'差(v5-v2Pro)':>14s}")
    for i, lab in enumerate(labels):
        print(f"  {lab:10s} {l5[i]:>10.2f} {l2[i]:>10.2f} {l5[i]-l2[i]:>14.2f}")
    print()
    print(f"  波峰因数  v5 {np.median(r5[:,2]-r5[:,1]):.2f} dB   "
          f"v2Pro {np.median(r2[:,2]-r2[:,1]):.2f} dB")
    print(f"  SNR估算   v5 {np.median(r5[:,1]-r5[:,3]):.2f} dB   "
          f"v2Pro {np.median(r2[:,1]-r2[:,3]):.2f} dB")

print()
print("=" * 80)
print("4. 结论")
print("=" * 80)
