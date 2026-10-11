# -*- coding: utf-8 -*-
"""验证 patch_s1_lr.py：三种 LR 模式都真的生效，且**默认行为不变**。

## 要验证的四件事
  1. 未设 ZFY_S1_LR  -> LR 恒 0.002（**与上游完全一致**，这是安全底线）
  2. ZFY_S1_LR=1e-4  -> LR 恒 1e-4（且写进 optimizer.param_groups）
  3. ZFY_S1_LR=schedule -> 真的按 warmup 上升、之后按 cosine 下降
  4. 三种模式都要写进 **optimizer.param_groups**（不只是 self.lr）
     —— 上游 bug 就是 set_lr 里写死 end_lr，导致外部算什么都没用

## 关键：模式 1 必须与「未打补丁」逐值一致
所以本脚本先用 git stash 之外的办法：直接对比打补丁前后同一组参数下的
LR 序列（未打补丁时恒 0.002）。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
LR_FILE = GSV / "GPT_SoVITS" / "AR" / "modules" / "lr_schedulers.py"
PY = sys.executable

# 用子进程跑，才能干净地控制环境变量（同一进程里改 os.environ 后
# 已 import 的模块不会重读）
#
# ⚠️ 用占位符替换而不是 str.format() —— 探针里有 f-string 的
#    `{x:.8g}`，会被 format() 当成占位符 -> KeyError: 'x'。
PROBE = r'''
import os, sys
sys.path.insert(0, GSV_PATH)
sys.path.insert(0, GSV_PATH + r"\GPT_SoVITS")
import torch
from AR.modules.lr_schedulers import WarmupCosineLRSchedule

m = torch.nn.Linear(4, 4)
opt = torch.optim.Adam(m.parameters(), lr=1e-5)
s = WarmupCosineLRSchedule(
    opt, init_lr=1e-5, peak_lr=0.01, end_lr=1e-4,
    warmup_steps=10, total_steps=40, current_step=0,
)
lrs = []
for i in range(50):
    s.step()
    lrs.append(opt.param_groups[0]["lr"])
print("|".join("%.8g" % x for x in lrs))
'''


def probe(env_extra: dict) -> list[float]:
    env = os.environ.copy()
    env.pop("ZFY_S1_LR", None)
    env.update(env_extra)
    code = PROBE.replace("GSV_PATH", repr(str(GSV)))
    r = subprocess.run([PY, "-c", code],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=300)
    if r.returncode != 0:
        print(f"    探针失败: {r.stderr[-400:]}")
        return []
    line = [l for l in (r.stdout or "").strip().split("\n") if "|" in l]
    if not line:
        print(f"    无输出: {r.stdout[-200:]!r}")
        return []
    return [float(x) for x in line[-1].split("|")]


def summarize(tag: str, lrs: list[float]) -> None:
    if not lrs:
        print(f"  {tag:34s} 无数据")
        return
    uniq = sorted(set(lrs))
    print(f"  {tag:34s} 首={lrs[0]:.8g}  末={lrs[-1]:.8g}  "
          f"最大={max(lrs):.8g}  不同值={len(uniq)} 个")


print("=" * 82)
print("1. 打补丁前（上游行为：恒 0.002）")
print("=" * 82)
before = probe({})
summarize("未打补丁 / 未设环境变量", before)

print()
print("=" * 82)
print("2. 打补丁")
print("=" * 82)
r = subprocess.run([PY, r"D:\A\zfy-v5\patch_s1_lr.py",
                    "--root", str(GSV)],
                   capture_output=True, text=True, encoding="utf-8",
                   errors="replace")
for l in (r.stdout or "").split("\n"):
    if l.strip() and ("[1/" in l or "[2/" in l or "[3/" in l
                      or "已写入" in l or "备份" in l):
        print(f"  {l.strip()}")

print()
print("=" * 82)
print("3. 打补丁后 —— 三种模式")
print("=" * 82)

modes = [
    ("未设 ZFY_S1_LR（默认）", {}),
    ("ZFY_S1_LR=1e-4", {"ZFY_S1_LR": "1e-4"}),
    ("ZFY_S1_LR=5e-5", {"ZFY_S1_LR": "5e-5"}),
    ("ZFY_S1_LR=schedule", {"ZFY_S1_LR": "schedule"}),
    ("ZFY_S1_LR=abc（非法）", {"ZFY_S1_LR": "abc"}),
]
results = {}
for tag, env in modes:
    lrs = probe(env)
    results[tag] = lrs
    summarize(tag, lrs)

print()
print("=" * 82)
print("4. 断言")
print("=" * 82)
checks = []

# 4.1 默认必须与上游一致（恒 0.002）
d = results["未设 ZFY_S1_LR（默认）"]
checks.append(("默认模式恒 0.002",
               bool(d) and all(abs(x - 0.002) < 1e-12 for x in d),
               f"{d[0] if d else '?'}"))
# 4.2 默认与打补丁前逐值一致
checks.append(("默认模式与打补丁前逐值一致",
               bool(d) and bool(before) and all(
                   abs(a - b) < 1e-12 for a, b in zip(d, before)),
               f"len {len(d)} vs {len(before)}"))

# 4.3 常量模式
c = results["ZFY_S1_LR=1e-4"]
checks.append(("ZFY_S1_LR=1e-4 -> 恒 1e-4",
               bool(c) and all(abs(x - 1e-4) < 1e-15 for x in c),
               f"{c[0] if c else '?'}"))
c2 = results["ZFY_S1_LR=5e-5"]
checks.append(("ZFY_S1_LR=5e-5 -> 恒 5e-5",
               bool(c2) and all(abs(x - 5e-5) < 1e-15 for x in c2),
               f"{c2[0] if c2 else '?'}"))

# 4.4 schedule 模式：应上升（warmup）后下降（cosine）
s = results["ZFY_S1_LR=schedule"]
if s and len(s) >= 40:
    warm = s[:10]
    peak = max(s)
    tail = s[40:]
    rising = all(warm[i] <= warm[i + 1] + 1e-12 for i in range(len(warm) - 1))
    falling = all(tail[i] >= tail[i + 1] - 1e-12 for i in range(len(tail) - 1))
    checks.append(("schedule：warmup 段单调上升", rising,
                   f"{warm[0]:.4g} -> {warm[-1]:.4g}"))
    checks.append(("schedule：cosine 段单调下降", falling,
                   f"{tail[0]:.4g} -> {tail[-1]:.4g}"))
    checks.append(("schedule：峰值接近 peak_lr=0.01",
                   abs(peak - 0.01) < 1e-3, f"{peak:.6g}"))
else:
    checks.append(("schedule：有数据", False, f"len={len(s)}"))

# 4.5 非法值应回退默认
b = results["ZFY_S1_LR=abc（非法）"]
checks.append(("非法值回退到 0.002",
               bool(b) and all(abs(x - 0.002) < 1e-12 for x in b),
               f"{b[0] if b else '?'}"))

bad = 0
for name, ok, extra in checks:
    bad += 0 if ok else 1
    print(f"  {'✓' if ok else '✗'} {name}" + (f"  ({extra})" if extra else ""))

print()
print("=" * 82)
print("结论")
print("=" * 82)
if bad == 0:
    print("  ✓ patch_s1_lr.py 全部验证通过")
    print("    · 默认行为与上游**逐值一致**（零风险）")
    print("    · 常量 / schedule 两种模式都真正写进了 optimizer")
else:
    print(f"  ✗ {bad} 项失败")

print()
print("  各模式 LR 序列（前 12 步 / 后 5 步）:")
for tag, lrs in results.items():
    if not lrs:
        continue
    head = " ".join(f"{x:.4g}" for x in lrs[:12])
    tail = " ".join(f"{x:.4g}" for x in lrs[-5:])
    print(f"    {tag:28s} {head} … {tail}")
