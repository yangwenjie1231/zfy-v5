# -*- coding: utf-8 -*-
"""为权重名选一个**既宽松又不误判**的正则（merged 支持 + 不误匹配实验名）。

## 需求（三次踩坑总结）
  1. 必须匹配 `v5_full_e32_e16_s864_l0.pth`           -> epoch 16
  2. 必须匹配 `v5_full_e32_e16_s864_l0_merged.pth`    -> epoch 16
  3. 必须匹配 `v5_full_e32_e16_merged.pth`            -> epoch 16  （我建议的简写）
  4. 必须匹配旧命名 `zfy_v5_e8_s3256_l32.pth`          -> epoch 8
  5. **不能**匹配 `v5_full_e32.pth`（实验名，无 epoch）
  6. 不能匹配 `random_name.pth` / `G_*.pth` / `s2Gv5dev.pth`

## 前两次的错法
  · `_e(\\d+)_s(\\d+)_l\\d+\\.pth$`          -> 漏掉 merged（第 1 次）
  · `_e(\\d+)(?:_s\\d+_l\\d+)?(?:_merged)?\\.pth$`
      -> `(?:_s\\d+_l\\d+)?` 是**可选**组，`v5_full_e32.pth` 会跳过它
         直接匹配实验名 e32（误判）
  · 加 `(?!.*_e\\d)` 前瞻**无效** —— 因为 `v5_full_e32.pth` 的 e32 后面
     什么都没有，前瞻条件天然成立

## 正确做法
  要求 epoch 后面**必须**跟 `_s{S}_l{L}` 或 `_merged` 之一：
      _e(\\d+)(?:_s\\d+_l\\d+|_merged)(?:_merged)?\\.pth$
  这样裸的 `v5_full_e32.pth` 就匹配不上了。
"""
from __future__ import annotations

import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PAT = r'_e(\d+)(?:_s\d+_l\d+|_merged)(?:_merged)?\.pth$'

TESTS = [
    # (文件名, 期望 epoch)
    ("v5_full_e32_e16_s864_l0.pth", 16),
    ("v5_full_e32_e16_s864_l0_merged.pth", 16),
    ("v5_full_e32_e16_merged.pth", 16),
    ("v5_full_e32_e8_s432_l0.pth", 8),
    ("v5_full_e32_e8_merged.pth", 8),
    ("v5_full_e32_e32_s1728_l0.pth", 32),
    ("zfy_v5_e8_s3256_l32.pth", 8),
    ("zfy_v5_full_e8_s432_l0.pth", 8),
    ("v5_full_e8_e8_s432_l0.pth", 8),
    ("v5_full_e32.pth", None),
    ("random_name.pth", None),
    ("G_233333333333.pth", None),
    ("s2Gv5dev.pth", None),
    ("v5_full_e32_e16.pth", None),
]

print("=" * 84)
print(f"PAT = {PAT}")
print("=" * 84)
print(f"  {'结果':6s} {'文件名':46s} {'got':>5s} {'want':>5s}")
bad = 0
for name, want in TESTS:
    m = re.search(PAT, name)
    got = int(m.group(1)) if m else None
    ok = got == want
    bad += 0 if ok else 1
    print(f"  {'OK' if ok else 'FAIL':6s} {name:46s} {str(got):>5s} {str(want):>5s}")

print()
print("=" * 84)
print("--epochs 16 会选中哪些（输出目录）")
print("=" * 84)
for name, _ in TESTS:
    m = re.search(PAT, name)
    if m and int(m.group(1)) == 16:
        sub = "e16" + ("_merged" if "merged" in name else "")
        print(f"  {name:46s} -> {sub}/")

print()
print("=" * 84)
print("结论")
print("=" * 84)
print("  ✓ 全部正确（merged 支持 + 不误匹配实验名）" if bad == 0
      else f"  ✗ {bad} 个错误")
