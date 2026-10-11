# -*- coding: utf-8 -*-
"""验证 cloud_synth_sweep.py 的权重名正则能覆盖所有命名形式。

## 为什么需要
`merged` 文件被**静默过滤**已经坑了两次：
  第 1 次：正则 `_l\\d+\\.pth$` 匹配不上 `..._l0_merged.pth`
  第 2 次：我建议的 `--out ..._e16_merged.pth` 没有 `_s{S}_l{L}` 段，
           修好的正则仍匹配不上 -> 又是 1 个权重、又用未合并版

两次都是"过滤掉但不报错"，所以这次：
  · 正则同时兼容 `_s{S}_l{L}` 与 `_merged` 为可选
  · 脚本会把**被跳过的文件及原因**打印出来（不再静默）
"""
from __future__ import annotations

import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 与 cloud_synth_sweep.py 中保持一致
#
# ⚠️ `(?!.*_e\d)` 负向前瞻是必需的：
#    没有它时 `(?:_s\d+_l\d+)?` 作为**可选**组会被跳过，
#    于是 `v5_full_e32.pth` 直接匹配实验名的 e32（误判成 epoch 32）。
#    加上前瞻后，若后面还有 `_e{数字}` 就不允许匹配。
PAT = r'_e(\d+)(?!.*_e\d)(?:_s\d+_l\d+)?(?:_merged)?\.pth$'

CASES = [
    ("v5_full_e32_e16_s864_l0.pth",          16, "e16"),
    ("v5_full_e32_e16_s864_l0_merged.pth",   16, "e16_merged"),
    ("v5_full_e32_e16_merged.pth",           16, "e16_merged"),   # <- 我建议的命名
    ("v5_full_e32_e8_s432_l0.pth",            8, "e08"),
    ("v5_full_e32_e8_merged.pth",             8, "e08_merged"),
    ("v5_full_e32_e32_s1728_l0.pth",         32, "e32"),
    ("zfy_v5_e8_s3256_l32.pth",               8, "e08"),          # 旧实验名
    ("zfy_v5_full_e8_s432_l0.pth",            8, "e08"),          # 旧全量名
]

print("=" * 84)
print("1. 逐个解析")
print("=" * 84)
print(f"  {'文件名':44s} {'epoch':>6s} {'输出目录':>14s} {'结果':>6s}")
ok_all = True
for name, want_ep, want_sub in CASES:
    m = re.search(PAT, name)
    if not m:
        print(f"  {name:44s} {'-':>6s} {'-':>14s} **FAIL**")
        ok_all = False
        continue
    ep = int(m.group(1))
    sub = f"e{ep:02d}"
    if "merged" in name.rsplit(".", 1)[0]:
        sub += "_merged"
    ok = (ep == want_ep and sub == want_sub)
    ok_all &= ok
    print(f"  {name:44s} {ep:6d} {sub:>14s} {'OK' if ok else '**FAIL**'}")

print()
print("=" * 84)
print("2. 不该匹配的（确认没有过度宽松）")
print("=" * 84)
BAD = ["random_name.pth", "v5_full_e32.pth", "G_233333333333.pth",
       "s2Gv5dev.pth"]
for name in BAD:
    m = re.search(PAT, name)
    got = m.group(1) if m else None
    print(f"  {name:44s} -> {got}   "
          f"{'OK（不匹配）' if got is None else '**误匹配**'}")
    if got is not None:
        ok_all = False

print()
print("=" * 84)
print("3. --epochs 16 会选中哪些")
print("=" * 84)
ALL = [c[0] for c in CASES] + BAD
for name in ALL:
    m = re.search(PAT, name)
    if m and int(m.group(1)) == 16:
        sub = f"e16" + ("_merged" if "merged" in name else "")
        print(f"  {name:44s} -> {sub}/")

print()
print("=" * 84)
print("结论")
print("=" * 84)
print("  ✓ 正则覆盖全部命名形式" if ok_all else "  ✗ 有问题")
print("""
  关键：现在 `--epochs 16` 会同时选中
     v5_full_e32_e16_s864_l0.pth        -> e16/         （对照，应 ~11.4s）
     v5_full_e32_e16_merged.pth         -> e16_merged/  （修复后，应 ~1.2s）
  分目录输出，可直接 A/B。
""")
