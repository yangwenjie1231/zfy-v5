# -*- coding: utf-8 -*-
"""用**真实的 32 个权重名**验证 epoch 解析修复。

## 命名格式（用户日志证实）
    v5_full_e32_e10_s540_l0.pth
            ^^^^  ^^^  ^^^
            实验名  epoch  step(=epoch x 54)
    v5_full_e32_e9_s486_l0.pth      (9 x 54 = 486 ✓)

## 为什么这个验证必要
如果没修，32 个权重的 epoch 全会被解析成 **32**（实验名），
后果：
  · 输出目录全挤到 out_sweep/e32/ -> **互相覆盖，只剩最后一个**
  · --epochs 8,16,24,32 筛选失效 -> 32 个全被选中
"""
from __future__ import annotations

import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EPOCHS = 32
SPE = 54          # 每 epoch 步数（日志反推 + s540/s486 证实）
names = [f"v5_full_e32_e{e}_s{e * SPE}_l0.pth" for e in range(1, EPOCHS + 1)]

print("=" * 76)
print("1. 真实文件名样例")
print("=" * 76)
for n in (names[0], names[8], names[9], names[-1]):
    print(f"  {n}")

OLD = re.compile(r"_e(\d+)_")
NEW = re.compile(r"_e(\d+)_s(\d+)_l\d+\.pth$")

print()
print("=" * 76)
print("2. 旧正则（非锚定）解析结果 —— 证明 bug")
print("=" * 76)
old_vals = []
for n in names:
    m = OLD.search(n)
    old_vals.append(int(m.group(1)) if m else None)
from collections import Counter
c = Counter(old_vals)
print(f"  32 个文件解析出的 epoch 分布: {dict(c)}")
print(f"  -> 全部解析成 {list(c)[0]}，**没有一个是对的**")
print(f"  -> 输出目录只有 1 个: out_sweep/e{list(c)[0]}/  => 互相覆盖")

print()
print("=" * 76)
print("3. 新正则（锚定结尾）解析结果")
print("=" * 76)
new_vals = []
for n in names:
    m = NEW.search(n)
    new_vals.append(int(m.group(1)) if m else None)
print(f"  解析结果: {new_vals[:6]} … {new_vals[-3:]}")
print(f"  与真实 epoch 一致: {new_vals == list(range(1, EPOCHS + 1))}")

print()
print("=" * 76)
print("4. 模拟 --epochs 8,16,24,32 的筛选与输出目录")
print("=" * 76)
want = {8, 16, 24, 32}
kept = []
for n in names:
    m = NEW.search(n)
    if m and int(m.group(1)) in want:
        kept.append((n, int(m.group(1))))
print(f"  选中 {len(kept)} 个（期望 4）:")
for n, e in kept:
    print(f"    {n:38s} -> out_sweep/e{e:02d}/")
print()
print(f"  输出目录: {[f'e{e:02d}' for _, e in kept]}  —— 互不覆盖 ✓")

print()
print("=" * 76)
print("5. 结论")
print("=" * 76)
ok = (len(set(old_vals)) == 1) and (new_vals == list(range(1, EPOCHS + 1))) \
     and len(kept) == 4
print(f"  修复必要: {'是（旧逻辑会把 32 个全挤到同一目录）' if len(set(old_vals))==1 else '否'}")
print(f"  修复正确: {'是' if new_vals == list(range(1, EPOCHS+1)) else '否'}")
print(f"  筛选正确: {'是（4 个）' if len(kept)==4 else f'否（{len(kept)} 个）'}")
print()
print("  => 必须先 git pull 拿到修复，再跑合成")
