# -*- coding: utf-8 -*-
"""验证 parse_epoch 的正则 —— 修掉"全部显示 32"的 bug。

## bug
`finalize_training.py` 的权重列表全部显示 epoch=32：
    v5_full_e32_nofreeze_e10_s540_l0.pth   -> 显示 32（应为 10）
      ^^^^^ 配置名里的 _e32_ 被先匹配到

原写法：
    re.search(r"_e(\\d+)_", stem)          # 贪婪？不，是**最左**优先

## 修法
锚定完整模式 `_e{N}_s{step}_l{rank}`：
    re.search(r"_e(\d+)_s\d+_l\d+", stem)
"""
from __future__ import annotations

import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_epoch_old(stem: str) -> int:
    """旧（有 bug）的实现。"""
    m = re.search(r"_e(\d+)_", stem)
    return int(m.group(1)) if m else 0


def parse_epoch(stem: str) -> int:
    """新实现：锚定 `_e{N}_s{step}_l{rank}`，兜底取**最后一个** `_e{N}_`。"""
    m = re.search(r"_e(\d+)_s\d+_l\d+", stem)
    if m:
        return int(m.group(1))
    m = re.search(r"_e(\d+)_merged", stem)
    if m:
        return int(m.group(1))
    hits = re.findall(r"_e(\d+)_", stem)
    return int(hits[-1]) if hits else 0


CASES = [
    ("v5_full_e32_nofreeze_e1_s54_l0", 1),
    ("v5_full_e32_nofreeze_e10_s540_l0", 10),
    ("v5_full_e32_nofreeze_e16_s864_l0", 16),
    ("v5_full_e32_nofreeze_e32_s1728_l0", 32),
    ("v5_full_e32_nofreeze_e16_merged", 16),
    ("v5_full_e32_e8_s432_l0", 8),          # 旧 freeze 版（配置名也是 e32）
    ("v5_full_e16_e8_s432_l0", 8),          # 配置名 e16、真 epoch e8
    ("zfy_v5_e1_s407_l32", 1),              # LoRA 版（l32 而非 l0）
]

print("=" * 78)
print("parse_epoch 正则验证")
print("=" * 78)
print(f"  {'文件名':<40s} {'旧实现':>7s} {'新实现':>7s} {'期望':>7s}")
print("  " + "-" * 66)
bad_old = bad_new = 0
for stem, want in CASES:
    o, n = parse_epoch_old(stem), parse_epoch(stem)
    bad_old += 0 if o == want else 1
    bad_new += 0 if n == want else 1
    mo = "✓" if o == want else "✗"
    mn = "✓" if n == want else "✗"
    print(f"  {stem:<40s} {o:5d}{mo} {n:5d}{mn} {want:7d}")

print()
print("=" * 78)
print("结论")
print("=" * 78)
print(f"  旧实现: {len(CASES)-bad_old}/{len(CASES)} 正确")
print(f"  新实现: {len(CASES)-bad_new}/{len(CASES)} 正确")
print()
if bad_old:
    print(f"  旧实现错在：`_e(\\d+)_` 会先匹配到**配置名**里的 `_e32_`")
    print(f"     v5_full_e32_nofreeze_e10_s540_l0")
    print(f"              ^^^^^ 命中这里 -> 显示 32，而不是 10")
if bad_new == 0:
    print("  ✓ 新实现全部正确（锚定 `_e{N}_s{step}_l{rank}`）")
print("=" * 78)
sys.exit(1 if bad_new else 0)
