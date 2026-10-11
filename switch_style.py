# -*- coding: utf-8 -*-
"""一键切换参考音频风格。

## 用法
    python switch_style.py --list
    python switch_style.py normal     # 正常说话（默认）
    python switch_style.py firm       # 语气强硬
    python switch_style.py gentle     # 语音娇
    python switch_style.py --show     # 看当前是哪个

## 原理
`ref/default.wav` 是推理时读取的参考音频。
本脚本把 `ref/styles/<name>.wav` 复制过去，并同步更新 `ref/ref.txt`。

⚠️ 会覆盖 `ref/default.wav`，但**原文件已备份为 `ref/styles/normal.wav`**。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(__file__).resolve().parent
GSV = V5 / "GPT-SoVITS"
REF = GSV / "ref"
STYLES = REF / "styles"
CFG = STYLES / "styles.json"

if not CFG.exists():
    print(f"  !! 找不到 {CFG}（先跑 make_style_variants.py）")
    sys.exit(1)

cfg = json.loads(CFG.read_text(encoding="utf-8"))
styles = {s["name"]: s for s in cfg["styles"]}
verdict = cfg.get("user_verdict", {})

ap = argparse.ArgumentParser()
ap.add_argument("style", nargs="?", default=None)
ap.add_argument("--list", action="store_true")
ap.add_argument("--show", action="store_true")
args = ap.parse_args()

if args.list or (args.style is None and not args.show):
    print("可用风格:")
    for n, s in styles.items():
        print(f"  {n:8s} {s['label']:10s} {s['dur']:.2f}s   "
              f"评价: {verdict.get(n, '—')}")
        print(f"           {(s['text'] or '')[:60]}")
    print()
    print("用法: python switch_style.py <name>")
    sys.exit(0)

if args.show:
    cur = REF / "default.wav"
    if not cur.exists():
        print("  当前: (无 default.wav)")
        sys.exit(0)
    cur_bytes = cur.read_bytes()
    for n, s in styles.items():
        p = STYLES / s["file"]
        if p.exists() and p.read_bytes() == cur_bytes:
            print(f"  当前风格: **{n}**（{s['label']}）")
            print(f"  文本: {s['text']}")
            sys.exit(0)
    print("  当前风格: 未知（default.wav 与三份风格都不一致）")
    sys.exit(0)

name = args.style
if name not in styles:
    print(f"  !! 未知风格 '{name}'。可用: {list(styles)}")
    sys.exit(1)

s = styles[name]
src = STYLES / s["file"]
if not src.exists():
    print(f"  !! 找不到 {src}")
    sys.exit(1)

# 备份当前（若非三份之一）
cur = REF / "default.wav"
if cur.exists():
    cur_bytes = cur.read_bytes()
    known = any((STYLES / x["file"]).exists()
                and (STYLES / x["file"]).read_bytes() == cur_bytes
                for x in styles.values())
    if not known:
        bak = REF / "default.wav.bak"
        if not bak.exists():
            shutil.copy2(cur, bak)
            print(f"  已备份原 default.wav -> {bak.name}")

shutil.copy2(src, cur)
txt = REF / "ref.txt"
txt.write_text((s["text"] or "") + "\n", encoding="utf-8")

print(f"  ✓ 已切换到 **{name}**（{s['label']}）")
print(f"    参考音频: {cur}")
print(f"    参考文本: {s['text']}")
print()
print("  ## 验证")
print("    听一下合成结果，确认语气符合预期：")
print("      python compare_ref_swap.py")
