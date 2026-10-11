# -*- coding: utf-8 -*-
"""审查留出集候选目录的可信度 —— 决定哪些能用来测相似度。

## 背景
`find_heldout_v2.py` 按文件名精确匹配，得出：
    剧情语音        905 共 / 745 精确同名 / **144 留出**
    干员语音        165 共 /  75 精确同名 /  **90 留出**
    训练集audio     951 共 / 825 精确同名 / **123 留出**
    语音合集_分条     81 共 /   0 精确同名 /  **81 留出**
    wav24k          922 共 /   0 精确同名 / **922 留出**  <- 可疑

`wav24k` 一个都不匹配，但它是 `D:\\A\\voiceclone\\data\\endfield\\wav24k` ——
训练数据准备阶段的目录。**它极可能就是训练素材的另一种命名**，
只是命名规则完全不同（`干员语音_au_*` vs `01_行动准备1_chr_*`）。

若把训练素材当留出集 -> 相似度虚高（技能里：0.91 vs 真实 0.67）。

## 本脚本
对每个候选目录抽样，检查文件名模式与内容特征，
判定它是"训练素材的别名"还是"真正的新素材"。
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
HELD = V5 / "dist" / "heldout" / "heldout_v2.json"
TRAIN_WAV = GSV / "logs" / "zfy_v5" / "5-wav32k"
NEW = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频")

d = json.loads(HELD.read_text(encoding="utf-8"))

print("=" * 92)
print("1. 训练集文件名模式")
print("=" * 92)
tw = sorted(TRAIN_WAV.glob("*.wav"))
pat = Counter()
for w in tw:
    n = w.stem
    if n.startswith("剧情_"):
        pat["剧情_<章节>_<序号>_<文本前缀>"] += 1
    elif "chr_0030_zhuangfy" in n:
        pat["<序号>_<中文名>_chr_0030_zhuangfy_<动作>"] += 1
    else:
        pat["其他"] += 1
for k, v in pat.most_common():
    print(f"  {v:5d}  {k}")
print()
print("  样例:")
for w in tw[:3]:
    print(f"    {w.stem[:74]}")
for w in tw:
    if "chr_0030" in w.name:
        print(f"    {w.stem[:74]}")
        break

print()
print("=" * 92)
print("2. 各候选目录的文件名模式")
print("=" * 92)
for label, files in d["by_dir"].items():
    if not files:
        continue
    print(f"\n--- {label}  ({len(files)} 个留出文件) ---")
    print("  样例:")
    for p in files[:4]:
        print(f"    {Path(p).stem[:74]}")
    # 模式统计
    c = Counter()
    for p in files:
        n = Path(p).stem
        if n.startswith("剧情_"):
            c["剧情_<章节>_<序号>_<文本>"] += 1
        elif n.startswith("干员语音_"):
            c["干员语音_au_*"] += 1
        elif re.match(r"^\d+_", n):
            c["<序号>_<中文名>..."] += 1
        elif n.startswith("语音合集"):
            c["语音合集_*"] += 1
        else:
            c["其他"] += 1
    print("  模式分布:")
    for k, v in c.most_common():
        print(f"    {v:5d}  {k}")

print()
print("=" * 92)
print("3. 关键判定：wav24k 是不是训练素材的别名？")
print("=" * 92)
w24 = d["by_dir"].get("wav24k", [])
if w24:
    # 训练集里「干员语音」类（chr_0030 命名）的时长
    train_chr = []
    for w in tw:
        if "chr_0030" in w.name:
            try:
                i = sf.info(str(w))
                train_chr.append(i.frames / float(i.samplerate))
            except Exception:  # noqa: BLE001
                pass
    print(f"  训练集 chr_0030_* 音频: {len(train_chr)} 条")
    if train_chr:
        a = np.array(train_chr)
        print(f"    时长: 最小 {a.min():.2f}  中位 {np.median(a):.2f}  "
              f"最大 {a.max():.2f}  合计 {a.sum():.1f}s")

    # wav24k 里的干员语音类
    w24_gy = [p for p in w24 if Path(p).stem.startswith("干员语音_")]
    print(f"  wav24k 里 干员语音_* : {len(w24_gy)} 条")
    if w24_gy:
        ds = []
        for p in w24_gy[:200]:
            try:
                i = sf.info(p)
                ds.append(i.frames / float(i.samplerate))
            except Exception:  # noqa: BLE001
                pass
        if ds:
            b = np.array(ds)
            print(f"    时长: 最小 {b.min():.2f}  中位 {np.median(b):.2f}  "
                  f"最大 {b.max():.2f}  合计 {b.sum():.1f}s")
            if train_chr:
                print()
                print(f"  训练集 chr_0030 合计 {a.sum():.1f}s  vs  "
                      f"wav24k 干员语音抽样 {b.sum():.1f}s（{len(ds)} 条）")
                print(f"  训练集干员语音条数应为 64（从 2-name2text 读）")
                print(f"  wav24k 干员语音条数 {len(w24_gy)}")
                if abs(len(w24_gy) - 64) < 20 or b.sum() > a.sum() * 1.5:
                    print("  -> wav24k **更全**（可能包含训练用的那批 + 更多）")
                    print("     结论：**不能直接用**（会混入训练素材）")
                else:
                    print("  -> 数量接近，**很可能是同一批的别名**")
                    print("     结论：**不能当留出集**")

print()
print("=" * 92)
print("4. 建议：哪些目录可以安全用作留出集")
print("=" * 92)
print("""
  判定依据（按可靠性排序）：
    ① **文件名精确匹配**能排除的，就是干净的
    ② 命名规则与训练集**完全不同**的目录，需额外确认内容不重复
    ③ 训练集来源是 `logs/zfy_v5/`，其素材来自 剧情语音 + 干员语音 + 训练集audio

  因此：
    · **剧情语音 / 干员语音 / 训练集audio 的"留出"部分** = 可靠
      （同目录下已用同名文件被精确排除）
    · **wav24k** = 高度可疑（0 匹配 + 命名不同 + 位于训练准备目录）
      -> 除非内容比对证明不重复，否则**不用**
    · **语音合集_分条** = 需确认（81 个，命名独立）
""")
print("=" * 92)
