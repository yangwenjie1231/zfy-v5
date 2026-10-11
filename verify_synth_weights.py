# -*- coding: utf-8 -*-
"""验证 cloud_synth_sweep.py 能正确处理**新的 nofreeze 权重**。

## 要验证的两件事

### 1) 权重名正则会先撞上配置名里的 `_e32_` 吗？
`v5_full_e32_nofreeze_e16_s864_l0.pth`
           ^^^^^ 配置名里的 e32
                                 ^^^^ 真正的 e16

`re.search` 从左往右扫：
  · 在 `_e32_` 处：捕获 32，然后要求 `(?:_s\d+_l\d+|_merged)`，
    但后面是 `_nofreeze_...` -> **不匹配** -> 这次尝试失败
  · 继续往后扫，在 `_e16_s864_l0.pth` 处：捕获 16，
    后面是 `_s864_l0.pth` -> **匹配** ✓

所以 `$` 锚定让正则**自愈**了。但必须实测确认 —— 这类"看起来对"的
正则在文件名多段相似时经常出错（我已犯过 3 次）。

### 2) 新权重**不该**再走 merge_base_encp.py
· 旧的 freeze 版权重**缺** enc_p -> 必须 merge
· 新的 nofreeze 版权重**自带** enc_p（已验证 +30.73 MB）-> **不能再 merge**
若误 merge，会用底模原值**覆盖掉**已训练好的 enc_p
-> 音质回退到"底模水平"，白训。

## 用法
    python verify_synth_weights.py
"""
from __future__ import annotations

import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PATTERN = r'_e(\d+)(?:_s\d+_l\d+|_merged)(?:_merged)?\.pth$'

CASES = [
    # 新权重（nofreeze）—— 配置名里含 _e32_
    ("v5_full_e32_nofreeze_e1_s54_l0.pth", 1),
    ("v5_full_e32_nofreeze_e8_s432_l0.pth", 8),
    ("v5_full_e32_nofreeze_e16_s864_l0.pth", 16),
    ("v5_full_e32_nofreeze_e24_s1296_l0.pth", 24),
    ("v5_full_e32_nofreeze_e32_s1728_l0.pth", 32),
    # 旧权重（freeze）—— 配置名也是 _e32_
    ("v5_full_e32_e8_s432_l0.pth", 8),
    ("v5_full_e32_e32_s1728_l0.pth", 32),
    # merge 过的（两种命名都出现过）
    ("v5_full_e32_e16_s864_l0_merged.pth", 16),
    ("v5_full_e32_e16_merged.pth", 16),
    # 其他配置名
    ("v5_full_e16_e8_s432_l0.pth", 8),
    ("zfy_v5_e1_s407_l32.pth", 1),
    # 应该被过滤掉的
    ("s2Gv5dev.pth", None),
    ("G_1728.pth", None),
]

print("=" * 92)
print("1) 权重名正则验证")
print("=" * 92)
print(f"  模式: {PATTERN}")
print()
print(f"  {'文件名':<44s} {'解析 epoch':>10s} {'期望':>6s}")
print("  " + "-" * 66)
bad = 0
for name, want in CASES:
    m = re.search(PATTERN, name)
    got = int(m.group(1)) if m else None
    ok = got == want
    bad += 0 if ok else 1
    print(f"  {'✓' if ok else '✗'} {name:<42s} {str(got):>10s} "
          f"{str(want):>6s}")

print()
print(f"  {len(CASES)-bad}/{len(CASES)} 正确")

# 额外：确认不会**误报**配置名里的 e32
print()
print("=" * 92)
print("2) 关键：配置名里的 `_e32_` 会不会被误当成 epoch？")
print("=" * 92)
for name in ("v5_full_e32_nofreeze_e16_s864_l0.pth",
             "v5_full_e32_nofreeze_e1_s54_l0.pth"):
    m = re.search(PATTERN, name)
    got = int(m.group(1)) if m else None
    print(f"  {name}")
    print(f"    匹配到 {got}  "
          f"{'✓（正确跳过配置名里的 e32）' if got != 32 or 'e32_s' in name else '✗ 误匹配配置名'}")
    # 说明为什么
    idx = m.start() if m else -1
    print(f"    匹配位置 {idx}（配置名 `_e32_` 在第 9 位）")

print()
print("=" * 92)
print("3) 新权重是否该走 merge_base_encp.py")
print("=" * 92)
print("""
  | 权重 | 含 enc_p | 该不该 merge |
  |---|---|---|
  | 旧 freeze 版 735,663,517 字节  | ✗ 缺 | **必须 merge** |
  | 新 nofreeze 版 766,394,418 字节 | ✓ 有 | **不能 merge** |

  ## 为什么新权重不能再 merge
  `merge_base_encp.py` 的作用是把**底模的** enc_p/ssl_proj 值填进
  "训练时被冻结、因此没进权重"的那些键。

  而新权重**已经含**训练好的 enc_p/ssl_proj（+30.73 MB，吻合 100.23%）。
  再 merge 会：
    · 用**底模原值**覆盖掉**已在新语料上微调过**的 enc_p
    · 音质回退到"底模水平" —— **白训 32 轮**

  ## 判断方法（跑前必查）
      python _tools/verify_encp_in_weights.py --dir <权重目录> --max 1
      # 键数 >= 660 且 enc_p=235 -> 自带，**跳过 merge**
      # 键数 == 444 且 enc_p=0   -> 缺，需要 merge

  ## 现状
  `SoVITS_weights_v5_full_e32_nofreeze/` 的 32 个权重都已是 766,394,418 字节
  -> **自带 enc_p，直接合成，不要 merge**
""")
print("=" * 92)
sys.exit(1 if bad else 0)
