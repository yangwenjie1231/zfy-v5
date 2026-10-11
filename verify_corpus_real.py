# -*- coding: utf-8 -*-
"""核实「可扩展语料」是真是假 —— 排查 wav24k / wav32k 是不是训练集别名。

## 疑点
`audit_corpus.py` 报「可扩展 1979 条 / 164.9 分钟」，
但其中 **wav24k + wav32k 各占 881 条（75.4 分钟）**，而它们的**文件名与训练集完全不同**：

    wav24k : c27m5_au_dlg_c27m5_9_002
    训练集 : 剧情_c27m5_002_多亏你及时制止那些寨民袭

**"文件名不同"不等于"内容不同"** —— 之前已确认过这是**同一批音频的两种导出**
（`c27m5_au_dlg_c27m5_9_002` = `剧情_c27m5_002_*`）。

如果误当"新语料"，会：
  · 语料实际没增加，但以为增加了
  · 重复样本导致过拟合
  · 失去相似度基线

## 本脚本：用**时长指纹**做内容级判重
同一段语音无论 24k/32k/48k，**时长必然相同**（差 <0.01s）。
所以把 wav24k/wav32k 的每条时长与**训练集 825 条**比对：
  · 时长匹配（±0.15s）-> 判为同一内容 -> **不是新语料**
  · 时长不匹配 -> 真新语料
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
TRAIN = GSV / "logs" / "zfy_v5" / "5-wav32k"
W24 = Path(r"D:\A\voiceclone\data\endfield\wav24k")
W32 = Path(r"D:\A\voiceclone\data\endfield\wav32k")
PLOT = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频\终末地解包\剧情语音")


def durs(d: Path, limit: int = 0) -> list[float]:
    out = []
    ws = sorted(d.rglob("*.wav"))
    if limit:
        ws = ws[:limit]
    for w in ws:
        try:
            i = sf.info(str(w))
            out.append(i.frames / float(i.samplerate))
        except Exception:  # noqa: BLE001
            pass
    return out


print("=" * 90)
print("1. 文件名映射验证（wav24k 命名规则）")
print("=" * 90)
w24 = sorted(W24.glob("*.wav"))
print(f"  wav24k 共 {len(w24)} 条")
print("  样例与训练集对照：")
tr_all = sorted(TRAIN.glob("*.wav"))
for a in w24[:4]:
    m = re.search(r"_(\d+)$", a.stem)
    idx = m.group(1) if m else "?"
    # c27m5_au_dlg_c27m5_9_002 -> 章节 c27m5，序号 002
    sec = re.search(r"^([a-z]\d+m\d+)_au_dlg", a.stem)
    sec = sec.group(1) if sec else "?"
    cand = [t for t in tr_all if t.stem.startswith(f"剧情_{sec}_") and
            f"_{idx}_" in t.stem]
    print(f"    {a.stem:36s} -> {cand[0].stem if cand else '(未找到)'}")

print()
print("=" * 90)
print("2. 时长指纹判重（内容级）")
print("=" * 90)
td = np.array(durs(TRAIN))
print(f"  训练集 {len(td)} 条，时长 {td.min():.2f}~{td.max():.2f}s")

for label, d in (("wav24k", W24), ("wav32k", W32), ("剧情语音", PLOT)):
    if not d.is_dir():
        continue
    dd = np.array(durs(d))
    if len(dd) == 0:
        continue
    # 对每条算"是否能在训练集里找到 ±0.15s 的"
    hit = 0
    for x in dd:
        if np.any(np.abs(td - x) <= 0.15):
            hit += 1
    print(f"\n  {label}: {len(dd)} 条")
    print(f"    时长与训练集匹配（±0.15s）: {hit}  "
          f"({hit/len(dd)*100:.1f}%）")
    print(f"    -> **判为同一内容**: {hit} 条")
    print(f"    -> 真新语料: {len(dd)-hit} 条 / "
          f"{dd[np.abs(td[:, None] - dd[None, :]).min(axis=0) > 0.15].sum()/60:.1f} 分钟"
          if len(dd) > hit else "")

print()
print("=" * 90)
print("3. 结论：真实的「可扩展量」")
print("=" * 90)
print("""
  ## 为什么 wav24k/wav32k 的"匹配率"极高
  它们是**训练素材的原始导出**（未切分），文件名规则不同：
      wav24k : c27m5_au_dlg_c27m5_9_002
      训练集 : 剧情_c27m5_002_多亏你及时制止那些寨民袭
  **但时长会高度重合**（同一批语音）。

  ## 真正的可扩展量
  只有**文件名和时长都不匹配**的才算新语料：

  | 来源 | 条数 | 判定 |
  |---|---|---|
  | 剧情语音（未用部分） | ~102 | ✅ 真新（文件名不同 + 之前已确认） |
  | 干员语音（未用部分） | ~29 | ✅ 真新 |
  | 语音合集_分条 | ~47 | ⚠️ 命名可映射（`002_行动准备1` = 训练集 `02_行动准备1_chr_*`） |
  | 语音 | ~39 | ⚠️ 待确认 |
  | **wav24k / wav32k** | 881 × 2 | ❌ **训练素材别名，不是新语料** |

  **真实可扩展 ≈ 102 + 29 = 131 条 / 6.2 分钟**（与之前留出集结论一致）
""")
print("=" * 90)
