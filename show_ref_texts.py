# -*- coding: utf-8 -*-
"""查候选参考音频的文本，并生成 compare 脚本需要的 CANDS 配置。

## 为什么需要
`compare_ref_swap.py` 里的 `CANDS` 是**硬编码**的。
要测新的 top01~top05，得先知道：
    1. 音频文件路径
    2. **对应的参考文本**（GPT-SoVITS 要求 prompt_text 与音频一字不差）

文本来源：`logs/zfy_v5/2-name2text.txt`（第 4 列 = 规范化文本）

## ⚠️ 关键：文本必须与音频严格对应
技能里明确写过：`ref_text` 写错 -> **整个音色提示就废了**。
所以本脚本会**验证**：
    · 能否在 2-name2text.txt 里找到
    · 文本是否非空
找不到的候选会被标记（不能用）

## 输出
    dist/ref_best2/cands.py   （可直接粘进 compare_ref_swap.py）
    dist/ref_best2/texts.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
OUT = V5 / "dist" / "ref_best2"
TXT = GSV / "logs" / "zfy_v5" / "2-name2text.txt"

# 文本表
tm: dict[str, str] = {}
for line in TXT.read_text(encoding="utf-8").splitlines():
    p = line.split("\t")
    if len(p) >= 4:
        tm[p[0].replace(".wav", "")] = p[3].strip()
print(f"  2-name2text.txt: {len(tm)} 条")

cands = json.loads((OUT / "candidates.json").read_text(encoding="utf-8"))
print(f"  候选: {len(cands)} 个")
print()
print("=" * 100)
print("TOP 10 候选的参考文本")
print("=" * 100)

rows = []
for i, s in enumerate(cands[:10], 1):
    p = Path(s["path"])
    st = p.stem
    t = tm.get(st, "")
    how = "精确"
    if not t:
        # 模糊：前缀匹配
        for k, v in tm.items():
            if k.startswith(st[:16]) or st.startswith(k[:16]):
                t, how = v, "模糊"
                break
    if not t:
        how = "**未找到**"
    rows.append({"rank": i, "path": str(p), "stem": st, "text": t,
                 "how": how, "sim": s["sim"], "flat": s["flat"],
                 "tb": s["tb"], "dur": s["dur"]})
    print(f"\n  top{i:02d}  {st[:52]}")
    print(f"        相似度 {s['sim']:.4f}  尾平坦 {s['flat']:.4f}  "
          f"尾/体 {s['tb']:+.2f}dB  时长 {s['dur']:.2f}s")
    print(f"        匹配 {how}")
    print(f"        文本: {t[:70] if t else '(无)'}")

usable = [r for r in rows if r["text"]]
print()
print("=" * 100)
print(f"可用于合成: {len(usable)} / {len(rows)}")
print("=" * 100)

print()
print("## 可直接粘进 compare_ref_swap.py 的 CANDS")
print()
print("CANDS = [")
print('    ("orig", GSV / "ref" / "default.wav",')
print('     (GSV / "ref" / "ref.txt").read_text(encoding="utf-8").strip()),')
for r in usable[:4]:
    print(f'    ("best{r["rank"]:02d}", Path(r"{r["path"]}"),')
    print(f'     {r["text"]!r}),')
print("]")

(OUT / "texts.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                encoding="utf-8")
(OUT / "cands.py").write_text(
    "CANDS = [\n"
    '    ("orig", GSV / "ref" / "default.wav",\n'
    '     (GSV / "ref" / "ref.txt").read_text(encoding="utf-8").strip()),\n'
    + "".join(
        f'    ("best{r["rank"]:02d}", Path(r"{r["path"]}"),\n'
        f'     {r["text"]!r}),\n' for r in usable[:4])
    + "]\n", encoding="utf-8")
print()
print(f"  -> {OUT / 'texts.json'}")
print(f"  -> {OUT / 'cands.py'}")
print("=" * 100)
