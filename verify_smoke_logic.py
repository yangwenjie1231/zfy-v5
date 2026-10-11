# -*- coding: utf-8 -*-
"""本地验证冒烟测试的**数据抽样逻辑**（不跑训练）。

核对 notebook cell 8 的做法是否可行：
  · 从 logs/zfy_v5/2-name2text.txt 抽 20 行
  · 只保留 4-cnhubert/*.pt 与 5-wav32k/* 同时存在的
  · 软链过去

这样能提前发现"抽样后样本不足""名字对不上"之类的问题。
"""
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FULL = Path(r"D:\A\zfy-v5\GPT-SoVITS\logs\zfy_v5")
TMP = Path(r"D:\A\zfy-v5\_smoke_test")
N = 20

print("=" * 70)
print("1. 全量数据检查")
print("=" * 70)
for name in ("2-name2text.txt", "3-bert", "4-cnhubert", "5-wav32k",
             "6-name2semantic.tsv"):
    p = FULL / name
    if not p.exists():
        print(f"  ✗ {name} 不存在")
        continue
    if p.is_dir():
        print(f"  ✓ {name:24s} {len(list(p.iterdir())):4d} 项")
    else:
        n = len([l for l in p.read_text(encoding="utf-8").strip().split("\n")
                 if l.strip()])
        print(f"  ✓ {name:24s} {n:4d} 行")

txt = FULL / "2-name2text.txt"
lines = [l for l in txt.read_text(encoding="utf-8").strip().split("\n")
         if l.strip()]
print(f"\n  name2text 行数: {len(lines)}")
print(f"  首行字段数: {len(lines[0].split(chr(9)))}（应为 4）")

print()
print("=" * 70)
print("2. 模拟抽样（seed=0, N=20）")
print("=" * 70)
random.seed(0)
picked = random.sample(lines, min(N, len(lines)))
print(f"  抽中: {len(picked)} 行")

hubert = FULL / "4-cnhubert"
wavs = FULL / "5-wav32k"
valid = []
for l in picked:
    parts = l.split("\t")
    if len(parts) != 4:
        print(f"  跳过（字段数 {len(parts)}）: {l[:60]}")
        continue
    name = parts[0]
    hp, wp = hubert / f"{name}.pt", wavs / name
    if hp.exists() and wp.exists():
        valid.append(l)
    else:
        print(f"  跳过（缺文件）: {name[:50]}  pt={hp.exists()} wav={wp.exists()}")

print(f"  有效: {len(valid)} 条")

if len(valid) < 2:
    print("\n  ✗ 有效样本不足！")
    sys.exit(1)

print()
print("=" * 70)
print("3. 软链测试")
print("=" * 70)
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True)
(TMP / "2-name2text.txt").write_text("\n".join(valid) + "\n", encoding="utf-8")

for sub, ext in (("4-cnhubert", ".pt"), ("5-wav32k", "")):
    d = TMP / sub
    d.mkdir()
    n = 0
    for l in valid:
        name = l.split("\t")[0]
        src = FULL / sub / f"{name}{ext}"
        if src.exists():
            dst = d / f"{name}{ext}"
            try:
                dst.symlink_to(src)
            except (OSError, NotImplementedError):
                try:
                    __import__("os").link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            n += 1
    real = sum(1 for f in d.iterdir() if not f.is_symlink())
    print(f"  {sub:14s} {n} 个链接（其中非链接 {real} 个）")

# 验证加载器能识别
print()
print("=" * 70)
print("4. 加载器逻辑校验（模拟 TextAudioSpeakerLoaderV4.__init__）")
print("=" * 70)
names4 = set(f.name[:-3] for f in (TMP / "4-cnhubert").iterdir())
names5 = set(f.name for f in (TMP / "5-wav32k").iterdir())
phoneme = {}
for line in (TMP / "2-name2text.txt").read_text(encoding="utf-8").strip().split("\n"):
    t = line.split("\t")
    if len(t) == 4:
        phoneme[t[0]] = [t[1]]
inter = set(phoneme) & names4 & names5
print(f"  phoneme_data: {len(phoneme)}")
print(f"  names4      : {len(names4)}")
print(f"  names5      : {len(names5)}")
print(f"  三者交集    : {len(inter)}  <- 训练实际使用的样本数")
print()
if len(inter) >= 2:
    print(f"  ✓ 冒烟逻辑可行，{len(inter)} 条样本会被加载")
    print(f"    注意：加载器要求 <100 条时会重复复制到 100 条")
else:
    print("  ✗ 交集不足，抽样逻辑有问题")
    sys.exit(1)

shutil.rmtree(TMP, ignore_errors=True)
print(f"\n已清理 {TMP}")
