# -*- coding: utf-8 -*-
"""测量「更大的语料」到底有多大 —— 决定 GPT 微调能扩到多少。

## 背景
当前训练集：825 条 / 63.5 分钟
（来自 剧情语音 745 + 干员语音 64 + 训练集audio 16）

用户想「用更大的语料微调」。所以要先量清楚：
  1. 现有全部音源加起来有多少条/多少分钟
  2. 扣掉**已在训练集里**的（按文件名精确匹配）
  3. 扣掉**时长不合格**的（<1.5s 音素覆盖不足；>12s 显存吃不下）
  4. 剩下的就是「可扩展量」

## 为什么必须精确
`3-bert` / `4-cnhubert` / `6-name2semantic.tsv` 都是**按条生成**的。
新增 N 条要重跑预处理（约 1 小时），所以必须知道 N 值不值得。

## ⚠️ 与留出集的冲突
之前筛出的 234 条「干净留出集」**正是这些未用的音频**。
如果拿来训练，就失去了相似度评估的基线。
本脚本会**分开报告**，让你决定取舍。

## 输出
    dist/corpus_audit.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
NEW = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频")
OUT = V5 / "dist"

TRAIN_WAV = GSV / "logs" / "zfy_v5" / "5-wav32k"

SOURCES = {
    "剧情语音": NEW / "终末地解包" / "剧情语音",
    "干员语音": NEW / "终末地解包" / "干员语音",
    "训练集audio": NEW / "庄方宜_TTS训练集" / "audio",
    "语音合集_分条": NEW / "语音合集_分条",
    "语音": NEW / "语音",
    "wav24k": Path(r"D:\A\voiceclone\data\endfield\wav24k"),
    "wav32k": Path(r"D:\A\voiceclone\data\endfield\wav32k"),
}
MIN_SEC, MAX_SEC = 1.5, 12.0


def stem_core(s: str) -> str:
    for suf in (" (1)", "_1", "-1", "_copy", " copy"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.strip()


print("=" * 92)
print("语料审计 —— 还能扩多少")
print("=" * 92)

# 训练集 stem 集合
train_stems: set[str] = set()
train_noprefix: set[str] = set()
if TRAIN_WAV.is_dir():
    for w in TRAIN_WAV.glob("*.wav"):
        c = stem_core(w.stem)
        train_stems.add(c)
        parts = c.split("_")
        if len(parts) >= 3:
            train_noprefix.add("_".join(parts[2:]))
print(f"  训练集: {len(train_stems)} 个 stem（去前缀版 {len(train_noprefix)}）")

rows = []
for label, d in SOURCES.items():
    if not d.is_dir():
        continue
    wavs = sorted(d.rglob("*.wav"))
    used = fresh = bad_dur = 0
    fresh_list = []
    for w in wavs:
        c = stem_core(w.stem)
        if c in train_stems:
            used += 1
            continue
        parts = c.split("_")
        if len(parts) >= 3 and "_".join(parts[2:]) in train_noprefix:
            used += 1
            continue
        try:
            info = sf.info(str(w))
            dur = info.frames / float(info.samplerate)
        except Exception:  # noqa: BLE001
            continue
        if not (MIN_SEC <= dur <= MAX_SEC):
            bad_dur += 1
            continue
        fresh += 1
        fresh_list.append({"path": str(w), "dur": round(dur, 3),
                           "sr": info.samplerate})
    tot_sec = sum(f["dur"] for f in fresh_list)
    rows.append({
        "label": label, "dir": str(d), "total": len(wavs),
        "used": used, "bad_dur": bad_dur, "fresh": fresh,
        "fresh_min": round(tot_sec / 60, 1),
        "files": fresh_list,
    })
    print(f"\n  {label}")
    print(f"    共 {len(wavs):5d} 条")
    print(f"    已在训练集 {used:5d}")
    print(f"    时长不合格 {bad_dur:5d}  （{MIN_SEC}~{MAX_SEC}s 之外）")
    print(f"    **可扩展 {fresh:5d} 条 / {tot_sec/60:.1f} 分钟**")

# 去重（跨目录同内容：用时长+采样率粗判）
print()
print("=" * 92)
print("去重（跨目录可能重复）")
print("=" * 92)
all_fresh = []
for r in rows:
    for f in r["files"]:
        all_fresh.append({**f, "src": r["label"]})

# 按 (dur 四舍五入到 0.05s) 分桶，同桶内若 sr 不同视为同一内容的不同采样率
seen: dict[float, list] = {}
for f in all_fresh:
    k = round(f["dur"] / 0.05)
    seen.setdefault(k, []).append(f)

# 保守：同桶内保留采样率最高的那个，其余算重复
dedup = []
dup = 0
for k, group in seen.items():
    if len(group) == 1:
        dedup.append(group[0])
        continue
    group.sort(key=lambda x: -x["sr"])
    dedup.append(group[0])
    dup += len(group) - 1
tot_min = sum(f["dur"] for f in dedup) / 60
print(f"  原始可扩展: {len(all_fresh)} 条 / {sum(f['dur'] for f in all_fresh)/60:.1f} 分钟")
print(f"  时长重复剔除: {dup} 条")
print(f"  **去重后: {len(dedup)} 条 / {tot_min:.1f} 分钟**")

# 训练集实际时长
train_min = 0.0
train_n = 0
if TRAIN_WAV.is_dir():
    for w in TRAIN_WAV.glob("*.wav"):
        try:
            info = sf.info(str(w))
            train_min += info.frames / float(info.samplerate)
            train_n += 1
        except Exception:  # noqa: BLE001
            pass

print()
print("=" * 92)
print("汇总")
print("=" * 92)
print(f"  当前训练集 : {train_n} 条 / {train_min:.1f} 分钟")
print(f"  可扩展     : {len(dedup)} 条 / {tot_min:.1f} 分钟")
print(f"  **扩展后   : {train_n + len(dedup)} 条 / {train_min + tot_min:.1f} 分钟**"
      f"   （+{(train_min+tot_min)/train_min:.2f}×）")
print()
print(f"  ⚠️ 社区建议：30+ 分钟语料 -> S1 训 8~12 轮")
print(f"     当前 {train_min:.0f} 分钟已属「大语料」档，扩展后 {train_min+tot_min:.0f} 分钟")

print()
print("=" * 92)
print("⚠️ 与留出集的冲突")
print("=" * 92)
print("""
  这 234 条「可扩展」**正是之前筛出的干净留出集**（用于测相似度 0.79）。

  如果拿来训练：
    · 训练集 825 -> 1059 条（+28%）
    · 但**失去相似度评估基线**（下次测会虚高，因为目标是训练集内素材）

  ## 三个选项
    ① **不用留出集**，只用现有 825 条（保守）
       -> 相似度基线保住，但语料没变
    ② **用留出集训练**，另留 10% 随机切分做新基线
       -> 语料 +28%，基线重建（需重测 v2Pro 的基线值）
    ③ **先测「扩大语料对 GPT 有没有用」**，再决定
       -> 本脚本的输出够做判断
""")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "corpus_audit.json").write_text(json.dumps({
    "train": {"n": train_n, "min": round(train_min, 1)},
    "expandable": {"n": len(dedup), "min": round(tot_min, 1),
                   "files": dedup},
    "by_source": [{k: v for k, v in r.items() if k != "files"} for r in rows],
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  -> {OUT / 'corpus_audit.json'}")
print("=" * 92)