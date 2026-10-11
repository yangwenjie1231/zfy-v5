# -*- coding: utf-8 -*-
"""验证：GPT 训练的 LR 调度是**死代码**，实际恒为 0.002。

## 代码证据（lr_schedulers.py:58）
    def step(self):
        if self._current_step < self.warmup_steps:
            lr = self.init_lr + self._warmup_rate * self._current_step
        elif ...
        else:
            ...cosine...
        self.lr = lr = self.end_lr = 0.002   # <-- 无论上面算什么，都被覆盖
        self.set_lr(lr)                       # set_lr 里也是 g["lr"] = self.end_lr

所以 warmup / cosine 分支的结果**全部被丢弃**，LR 恒为 0.002。

## 这意味着什么
  · 我 train_gpt.py 里的「LR 调度诊断」（warmup 2000 / decay 40000）
    是**无效的** —— 不管传什么，实际 LR 都是 2e-3。
  · 生产微调 zfh-e4 也跑的是这条路径（同仓库同文件）-> 也是 2e-3。
  · 所以「LR 全程在 warmup」这个担心**不成立**，但也说明
    `--warmup-steps` / `--decay-steps` 参数**没有任何作用**。

## 本脚本
  1. 直接实例化 scheduler，跑 100 步打印实际 LR
  2. 确认与传入参数无关
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
sys.path.insert(0, str(GSV))
sys.path.insert(0, str(GSV / "GPT_SoVITS"))

import torch  # noqa: E402
from AR.modules.lr_schedulers import WarmupCosineLRSchedule  # noqa: E402

print("=" * 78)
print("1. 实例化 scheduler，跑 20 步看实际 LR")
print("=" * 78)

m = torch.nn.Linear(4, 4)
opt = torch.optim.Adam(m.parameters(), lr=1e-5)

s = WarmupCosineLRSchedule(
    opt, init_lr=1e-5, peak_lr=0.01, end_lr=1e-5,
    warmup_steps=2000, total_steps=40000, current_step=0,
)
print(f"  传入: init 1e-5 / peak 0.01 / end 1e-5 / warmup 2000 / total 40000")
print()
print(f"  {'step':>5s} {'s.lr':>12s} {'optimizer lr':>14s}")
for i in range(20):
    s.step()
    opt_lr = opt.param_groups[0]["lr"]
    print(f"  {i:5d} {s.lr:12.6f} {opt_lr:14.6f}")

print()
print("=" * 78)
print("2. 换一组完全不同的参数，看 LR 是否变化")
print("=" * 78)
# 注意：total_steps == warmup_steps 会在 __init__ 里除零
# （self._decay_rate = (end_lr - peak_lr) / (total_steps - warmup_steps)）
# 这本身也是上游的一个小 bug，这里避开。
for (init, peak, end, ws, ts) in [(1e-9, 1e-9, 1e-9, 1, 2),
                                  (1e-5, 0.01, 1e-5, 2000, 40000),
                                  (0.1, 0.2, 0.3, 10, 100)]:
    m2 = torch.nn.Linear(4, 4)
    o2 = torch.optim.Adam(m2.parameters(), lr=init)
    s2 = WarmupCosineLRSchedule(o2, init, peak, end, ws, ts)
    for _ in range(5):
        s2.step()
    print(f"  传入 init={init:g} peak={peak:g} end={end:g} "
          f"warmup={ws} total={ts}")
    print(f"    -> 实际 LR = {o2.param_groups[0]['lr']:.6f}")

print()
print("=" * 78)
print("3. 结论")
print("=" * 78)
print("""
  ✓ 无论传什么，实际 LR 恒为 **0.002**。
  ✓ warmup / cosine 两个分支的 lr 计算结果在 line 58 被覆盖。

  影响：
    · `--warmup-steps` / `--decay-steps` **无作用**，可以不管
    · train_gpt.py 的「LR 调度诊断」是误导性的，应改掉
    · 生产 zfh-e4 走的同一条路径 -> 也是 2e-3 恒定

  好处：不用担心「全程 warmup 导致 LR 太小」—— 实际是 2e-3 恒定，
        这是个**偏大**的学习率（微调通常 1e-5~1e-4）。

  ⚠️ 但这也意味着：若训练发散/过拟合，LR 是**不可调**的
     （除非改 lr_schedulers.py 或绕过 scheduler）。
""")
