# -*- coding: utf-8 -*-
"""补充调查：语料构成、增益分布、v5 训练超参。

## 前一轮已知
  响度对比（实测）：
    指标          v5        v2Pro     差
    整体RMS     -18.37     -18.23    -0.14
    语音段RMS   -15.88     -15.37    -0.51
    峰值         -1.82      -2.35    +0.53
  => **两份语料的响度几乎一致**，增益归一化没有造成口径差

  但参考音频明显更轻：
    default.wav  整体RMS -22.26  语音段 -19.16  峰值 -6.82
    -> 比训练数据低约 **4~5 dB**

## 本脚本补查
  1. manifest 的增益dB / 峰值 完整分布
  2. v5 语料的**构成**（剧情 / 战斗 / UI 各占多少）
     —— 战斗喊招会拉高音高力度，污染音色基底
  3. v5 全量微调用的学习率等超参
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DS = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频\庄方宜_TTS训练集")
GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")


def dist(name, vals, unit=""):
    if not vals:
        print(f"  {name:12s} 无数据")
        return
    v = sorted(vals)
    n = len(v)
    print(f"  {name:12s} n={n:4d}  最小 {v[0]:8.2f}  P10 {v[n//10]:8.2f}  "
          f"中位 {v[n//2]:8.2f}  P90 {v[n*9//10]:8.2f}  最大 {v[-1]:8.2f} {unit}")


print("=" * 80)
print("1. manifest 增益与峰值")
print("=" * 80)
rows = list(csv.DictReader((DS / "manifest.csv").open(encoding="utf-8-sig")))
print(f"  总行数: {len(rows)}")

# 只看保留的
kept = [r for r in rows if (r.get("处理") or "").strip() == "保留"]
print(f"  其中「保留」: {len(kept)}")


def nums(rs, col):
    out = []
    for r in rs:
        v = (r.get(col) or "").strip()
        if not v:
            continue
        try:
            out.append(float(v))
        except ValueError:
            pass
    return out


for col in ("增益dB", "峰值", "限幅dB", "底噪dBFS", "语音dBFS", "SNR",
            "裁首尾静音s", "压缩停顿s", "新时长", "原时长"):
    dist(col, nums(kept, col), "dB" if "dB" in col else ("s" if "s" in col[-2:] else ""))

print()
print("=" * 80)
print("2. v5 语料构成（按文件名分类）")
print("=" * 80)
SEM = GSV / "logs" / "zfy_v5" / "6-name2semantic.tsv"
names = []
for line in SEM.read_text(encoding="utf-8", errors="replace").strip().split("\n"):
    p = line.split("\t")
    if p and p[0].strip() and p[0].strip() != "item_name":
        names.append(p[0].strip())
print(f"  语义文件条目: {len(names)}")


def classify(n: str) -> str:
    s = n.lower()
    if "combat" in s or "battle" in s:
        return "战斗/喊招"
    if "剧情" in n:
        return "剧情对白"
    if s.startswith("au_") or "greet" in s or "talk" in s:
        return "环境/问候"
    if re.match(r"^\d+_", n):
        return "UI/系统语音"
    return "其它"


c = Counter(classify(n) for n in names)
tot = sum(c.values())
for k, v in c.most_common():
    print(f"    {v:5d}  {v/tot*100:5.1f}%  {k}")

print()
print("  战斗类样例 8 条（这些音高力度偏离日常说话）:")
for n in names:
    if classify(n) == "战斗/喊招":
        print(f"    {n[:74]}")
        if names.index(n) and sum(1 for x in names[:names.index(n)+1]
                                  if classify(x) == "战斗/喊招") >= 8:
            break

print()
print("=" * 80)
print("3. v5 全量微调的超参")
print("=" * 80)
sj = GSV / "GPT_SoVITS" / "configs" / "s2.json"
if sj.exists():
    cfg = json.loads(sj.read_text(encoding="utf-8"))
    for k in ("batch_size", "learning_rate", "epochs", "lr_decay",
              "warmup_epochs", "grad_ckpt", "fp16_run", "text_low_lr_rate"):
        if k in cfg.get("train", {}):
            print(f"  train.{k:18s} = {cfg['train'][k]}")
    print()
    print("  模型侧:")
    for k, v in cfg.get("model", {}).items():
        print(f"    {k:22s} = {v}")
    print()
    print("  数据侧:")
    for k, v in cfg.get("data", {}).items():
        print(f"    {k:22s} = {v}")

print()
print("=" * 80)
print("4. v2Pro 的训练超参（对照）")
print("=" * 80)
vp = Path(r"D:\A\voiceclone\GPT-SoVITS\logs\zfh\config.json")
if vp.exists():
    c2 = json.loads(vp.read_text(encoding="utf-8"))
    for k in ("batch_size", "learning_rate", "epochs", "lr_decay",
              "warmup_epochs", "grad_ckpt", "fp16_run"):
        if k in c2.get("train", {}):
            print(f"  train.{k:18s} = {c2['train'][k]}")
    print()
    for k, v in c2.get("model", {}).items():
        print(f"    {k:22s} = {v}")
else:
    print(f"  找不到 {vp}")
