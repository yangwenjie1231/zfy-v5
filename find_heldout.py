# -*- coding: utf-8 -*-
"""找出真正的「留出集」：同角色（庄方宜）但**没进训练**的素材。

## 为什么这一步是整个评估的地基
技能里的铁律：
    训练集内素材当相似度目标 = 0.91（虚高，只是背下了训练录音）
    训练集外同角色素材       = 0.67（真实水平）
    **天然上限**（真实录音 vs 另一段真实录音）= 0.66

**拿训练集内的音频当目标，测的是"背诵能力"不是"泛化能力"。**

## v5 训练集构成（从 logs/zfy_v5/2-name2text.txt 读出）
    745 条 剧情_*（各章节）
     64 条 干员语音_*
     16 条 训练集_*
    共 825 条

## 本脚本
  1. 列出训练集**用到的**所有源文件名
  2. 扫描候选素材目录，找出**同角色但未参与训练**的 wav
  3. 按来源分组（章节/类别），报告条数与时长
  4. 输出清单 JSON，供 eval 脚本使用

## 输出
    D:\\A\\zfy-v5\\dist\\heldout\\heldout_list.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
VC = Path(r"D:\A\voiceclone")
NEW = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频")
OUT = V5 / "dist" / "heldout"

TRAIN_LIST = GSV / "logs" / "zfy_v5" / "2-name2text.txt"

# 候选素材目录（同角色素材可能在这几处）
CANDIDATE_DIRS = [
    NEW / "终末地解包" / "剧情语音",
    NEW / "终末地解包" / "干员语音",
    NEW / "庄方宜_TTS训练集" / "audio",
    NEW / "语音合集_分条",
    NEW / "语音",
    VC / "data" / "endfield" / "wav24k",
]

print("=" * 92)
print("1. 训练集用了哪些源文件")
print("=" * 92)

train_names: set[str] = set()
train_prefix: dict[str, int] = defaultdict(int)
if TRAIN_LIST.exists():
    for line in TRAIN_LIST.read_text(encoding="utf-8", errors="replace").split("\n"):
        if not line.strip():
            continue
        fn = line.split("\t")[0].strip()
        if fn:
            train_names.add(fn)
            train_names.add(Path(fn).stem)          # 也存无扩展名版
            m = re.match(r"^([^_]+(?:_[^_]+)?)", fn)
            if m:
                train_prefix[m.group(1)] += 1
    print(f"  训练集文件名: {len(train_names)} 个标识")
    print("  前缀分布（前 12）:")
    for k, v in sorted(train_prefix.items(), key=lambda t: -t[1])[:12]:
        print(f"    {v:5d}  {k}")
else:
    print(f"  !! 找不到 {TRAIN_LIST}")

print()
print("=" * 92)
print("2. 扫描候选目录，找未参与训练的素材")
print("=" * 92)

heldout: dict[str, list[dict]] = {}
for d in CANDIDATE_DIRS:
    if not d.is_dir():
        continue
    wavs = sorted(d.rglob("*.wav"))
    if not wavs:
        continue
    fresh, used = [], 0
    for w in wavs:
        stem = w.stem
        if stem in train_names or w.name in train_names:
            used += 1
            continue
        # 模糊匹配：训练集里的名字可能被截断过（如 "剧情_c27m5_006_我答应你，汤汤。.wav"）
        # 用「去掉前缀后的前 8 个字」做二次判定
        core = re.sub(r"^(剧情|干员语音|训练集)_[a-z0-9]+_", "", stem)
        core = re.sub(r"^[^_]*_", "", stem) if not core else core
        hit = any(core and core[:8] in t for t in train_names) if core else False
        if hit:
            used += 1
            continue
        try:
            sz = w.stat().st_size
        except OSError:
            continue
        fresh.append({"path": str(w), "name": w.name, "bytes": sz})
    key = str(d.relative_to(NEW)) if NEW in d.parents or d == NEW else str(d)
    heldout[key] = fresh
    print(f"  {key}")
    print(f"      共 {len(wavs):5d}  已用 {used:5d}   **留出 {len(fresh):5d}**")

print()
print("=" * 92)
print("3. 留出集汇总")
print("=" * 92)

all_fresh = []
for k, v in heldout.items():
    all_fresh.extend(v)
print(f"  留出文件总数: {len(all_fresh)}")
if all_fresh:
    mb = sum(f["bytes"] for f in all_fresh) / 1e6
    print(f"  总大小: {mb:.1f} MB")
    print()
    print("  按目录:")
    for k, v in sorted(heldout.items(), key=lambda t: -len(t[1])):
        if v:
            print(f"    {len(v):5d}  {k}")
    print()
    print("  样例（前 12）:")
    for f in all_fresh[:12]:
        print(f"    {f['name'][:70]}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "heldout_list.json").write_text(
    json.dumps({"by_dir": {k: v for k, v in heldout.items()},
                "all": all_fresh}, ensure_ascii=False, indent=2),
    encoding="utf-8")
print()
print(f"  -> {OUT / 'heldout_list.json'}")
print("=" * 92)
