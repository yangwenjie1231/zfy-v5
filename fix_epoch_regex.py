# -*- coding: utf-8 -*-
"""修 cloud_synth_sweep.py / cloud_push_weights.py 的 epoch 解析 bug。

## bug

权重文件名格式是：

    {实验名}_e{epoch}_s{step}_l{lora}.pth
    v5_full_e32_e8_s432_l0.pth
            ^^^^  ^^
            实验名  epoch

而两个脚本用的是**非锚定**正则：

    re.search(r"_e(\d+)_", name)      # 从左往右第一个匹配 -> 命中 e32（实验名）！
    re.search(r"_e(\d+)_s\d+", name)  # 这个碰巧对，但不够明确

后果（cloud_synth_sweep.py）：
  · 所有 32 个权重的 sub 都算成 "e32" -> 输出到同一个目录 -> **互相覆盖**
  · --epochs 32 的筛选也失效（每个文件名都含 "e32_"）

## 修法

用**锚定到结尾**的正则，唯一确定 epoch：

    _e(\d+)_s(\d+)_l\d+\.pth$

用法：
    python fix_epoch_regex.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")

NEW_REGEX = r'_e(\d+)_s(\d+)_l\d+\.pth$'

# ---------- 1) cloud_synth_sweep.py ----------
p1 = V5 / "cloud_synth_sweep.py"
s = p1.read_text(encoding="utf-8")

OLD_FILTER = '''    if args.epochs:
        want = {f"e{e.strip()}_" for e in args.epochs.split(",") if e.strip()}
        weights = [w for w in weights if any(k in w.name for k in want)]'''
NEW_FILTER = '''    if args.epochs:
        # ⚠️ 必须**锚定到结尾**解析 epoch。
        #    权重名形如 v5_full_e32_e8_s432_l0.pth —— 同时含 "e32"（实验名）
        #    和 "e8"（真 epoch）。用非锚定的 re.search 或 `"e8_" in name`
        #    都会误判（比如 --epochs 32 会匹配到全部 32 个文件）。
        want = {int(x) for x in args.epochs.split(",") if x.strip()}
        kept = []
        for w in weights:
            m = re.search(r'_e(\\d+)_s(\\d+)_l\\d+\\.pth$', w.name)
            if m and int(m.group(1)) in want:
                kept.append(w)
        weights = kept'''

if OLD_FILTER in s:
    s = s.replace(OLD_FILTER, NEW_FILTER)
    print("  cloud_synth_sweep.py: 已修筛选逻辑")
else:
    print("  cloud_synth_sweep.py: 未找到筛选片段（可能已改）")

OLD_SUB = '''    for w in weights:
        m = re.search(r"_e(\\d+)_", w.name)
        sub = f"e{m.group(1)}" if m else w.stem
        jobs.append((args.version, w, sub, v5_steps, v5_cfg))'''
NEW_SUB = '''    for w in weights:
        m = re.search(r'_e(\\d+)_s(\\d+)_l\\d+\\.pth$', w.name)
        if m:
            sub = f"e{int(m.group(1)):02d}"          # e08 / e32，补零便于排序
        else:
            sub = w.stem
        jobs.append((args.version, w, sub, v5_steps, v5_cfg))'''

if OLD_SUB in s:
    s = s.replace(OLD_SUB, NEW_SUB)
    print("  cloud_synth_sweep.py: 已修输出子目录命名")
else:
    print("  cloud_synth_sweep.py: 未找到 sub 片段（可能已改）")

p1.write_text(s, encoding="utf-8")

# ---------- 2) cloud_push_weights.py ----------
p2 = V5 / "cloud_push_weights.py"
s2 = p2.read_text(encoding="utf-8")
OLD_EP = '''def ep_of(name: str) -> int | None:
    m = re.search(r"_e(\\d+)_s\\d+", name)
    if m:
        return int(m.group(1))
    m = re.search(r"_e(\\d+)_", name)
    return int(m.group(1)) if m else None'''
NEW_EP = '''def ep_of(name: str) -> int | None:
    """从权重名解析 epoch —— **必须锚定到结尾**。

    名字形如 v5_full_e32_e8_s432_l0.pth，同时含 "e32"（实验名）和
    "e8"（真 epoch）。非锚定正则会命中前者。
    """
    m = re.search(r"_e(\\d+)_s(\\d+)_l\\d+\\.pth$", name)
    if m:
        return int(m.group(1))
    m = re.search(r"_e(\\d+)_l\\d+\\.pth$", name)
    return int(m.group(1)) if m else None'''

if OLD_EP in s2:
    s2 = s2.replace(OLD_EP, NEW_EP)
    print("  cloud_push_weights.py: 已修 ep_of")
else:
    print("  cloud_push_weights.py: 未找到 ep_of（可能已改）")
p2.write_text(s2, encoding="utf-8")

# ---------- 3) 验证 ----------
print()
print("=" * 74)
print("验证：新旧正则对比")
print("=" * 74)
NAMES = [
    "v5_full_e32_e8_s432_l0.pth",
    "v5_full_e32_e32_s1728_l0.pth",
    "v5_full_e8_e8_s432_l0.pth",
    "zfy_v5_e8_s432_l32.pth",
    "zfy_v5_r128_e24_s1296_l128.pth",
    "smoke_e2_s96_l0.pth",
]
print(f"  {'文件名':36s} {'旧(非锚定)':>12s} {'新(锚定)':>10s}")
for n in NAMES:
    old = re.search(r"_e(\d+)_", n)
    old_v = int(old.group(1)) if old else None
    new = re.search(r"_e(\d+)_s(\d+)_l\d+\.pth$", n)
    new_v = int(new.group(1)) if new else None
    flag = "  <-- 旧错" if old_v != new_v else ""
    print(f"  {n:36s} {str(old_v):>12s} {str(new_v):>10s}{flag}")

print()
print("=" * 74)
print("结论")
print("=" * 74)
print("""
  旧正则对 v5_full_e32_e8_*.pth 解析成 **32**（实验名），实际应是 **8**。
  新正则锚定结尾，正确解析。

  修好后 --epochs 8,16,24,32 会正确挑出 4 个，且输出到
  out_sweep/e08/ e16/ e24/ e32/ 四个**独立**目录，不再互相覆盖。
""")
