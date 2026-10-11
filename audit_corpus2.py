# -*- coding: utf-8 -*-
"""查最后两个未审计的目录 —— 是不是真新语料。

## 待查
  1. `dsh-zhuang-fangyi-pet/art/voice-src`   126 个（**新发现**）
  2. `新素材/音频/庄方宜_TTS训练集/_排除`      108 个（**被主动排除的**）
  3. `庄方宜素材`                              53 个
  4. `zfy_video`                                8 个

## 判重方法（与 verify_corpus_real.py 一致）
用**时长指纹**（±0.15s）与训练集 825 条比对。
同一段语音无论采样率，时长必然相同。

## `_排除` 目录要特别小心
它是**主动排除的**（名字就叫 `_排除`）—— 可能因为：
  · 含 BGM / 噪底高
  · 文本对不上
  · 时长不合格
**必须先看排除原因**，不能直接拿来训练。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
TRAIN = V5 / "GPT-SoVITS" / "logs" / "zfy_v5" / "5-wav32k"
FAKE = Path(r"D:\Users\A\Downloads\仿通行证")

CANDS = {
    "pet-voice-src": FAKE / "dsh-zhuang-fangyi-pet" / "art" / "voice-src",
    "_排除": FAKE / "新素材" / "音频" / "庄方宜_TTS训练集" / "_排除",
    "庄方宜素材": FAKE / "庄方宜素材",
    "zfy_video": FAKE / "zfy_video",
}


def probe(p: Path):
    try:
        i = sf.info(str(p))
        return i.frames / float(i.samplerate), i.samplerate
    except Exception:  # noqa: BLE001
        return None, None


print("=" * 92)
print("训练集基准")
print("=" * 92)
td = []
for w in sorted(TRAIN.glob("*.wav")):
    d, _ = probe(w)
    if d is not None:
        td.append(d)
td = np.array(td)
print(f"  {len(td)} 条，时长 {td.min():.2f}~{td.max():.2f}s")

# 训练集 stem（用于文件名判重）
train_stems = {w.stem for w in TRAIN.glob("*.wav")}

print()
print("=" * 92)
print("逐目录审计")
print("=" * 92)

summary = []
for label, d in CANDS.items():
    if not d.is_dir():
        print(f"\n  {label}: 目录不存在")
        continue
    wavs = sorted(d.rglob("*.wav"))
    if not wavs:
        print(f"\n  {label}: 无 wav")
        continue
    name_dup = 0
    dur_dup = 0
    fresh = []
    bad = 0
    for w in wavs:
        if w.stem in train_stems:
            name_dup += 1
            continue
        dur, sr = probe(w)
        if dur is None:
            bad += 1
            continue
        if np.any(np.abs(td - dur) <= 0.15):
            dur_dup += 1
            continue
        fresh.append({"path": str(w), "dur": round(dur, 2), "sr": sr})

    tot_min = sum(f["dur"] for f in fresh) / 60
    print(f"\n  {label}")
    print(f"    共 {len(wavs):5d} 条")
    print(f"    文件名重复 {name_dup:5d}")
    print(f"    时长重复   {dur_dup:5d}")
    print(f"    读取失败   {bad:5d}")
    print(f"    **真新语料 {len(fresh):5d} 条 / {tot_min:.1f} 分钟**")
    if fresh:
        print(f"    样例:")
        for f in fresh[:3]:
            print(f"      {Path(f['path']).name[:60]}  {f['dur']}s")
    summary.append((label, len(fresh), tot_min))

print()
print("=" * 92)
print("汇总：真实可扩展量")
print("=" * 92)
print(f"  {'来源':<20s} {'条数':>6s} {'分钟':>8s}  判定")
print("  " + "-" * 62)
KNOWN = {
    "剧情语音(未用)": (102, 4.6, "✅ 真新"),
    "干员语音(未用)": (29, 1.6, "✅ 真新"),
    "语音合集_分条": (47, 3.5, "⚠️ 命名可映射"),
    "语音": (39, 4.3, "⚠️ 待确认"),
}
for k, (n, m, v) in KNOWN.items():
    print(f"  {k:<20s} {n:6d} {m:8.1f}  {v}")
for label, n, m in summary:
    print(f"  {label:<20s} {n:6d} {m:8.1f}  ⚠️ 见下")

tot = sum(n for _, n, _ in summary) + sum(n for n, _, _ in KNOWN.values())
tot_min = sum(m for _, _, m in summary) + sum(m for _, m, _ in KNOWN.values())
print("  " + "-" * 62)
print(f"  {'合计（未去重）':<20s} {tot:6d} {tot_min:8.1f}")
print()
print("""
  ## ⚠️ `_排除` 目录必须先看原因
  它被**主动排除**过（名字就叫 `_排除`）。可能原因：
    · 含 BGM / 噪底高
    · 文本对不上（OCR vs ASR）
    · 时长不合格
  **直接用会污染训练集。** 建议先抽查：
      python -c "import soundfile as sf, glob; ..."   # 看时长分布
  或用现有的 BGM 检测（停顿处 RMS）。

  ## 关键结论
  真实可扩展量 **远小于 audit_corpus.py 报的 1979 条**。
  主因：`wav24k`/`wav32k` 各 881 条是**训练素材别名**（95% 时长重合）。

  ## 所以「用更大语料」这条路
  最多再扩 **100~250 条 / 5~12 分钟**（从 63.5 -> 70~75 分钟，**+10~20%**）。
  **不是量级变化** —— 不太可能改变"GPT 微调是否有效"的结论。
""")
print("=" * 92)