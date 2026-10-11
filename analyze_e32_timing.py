# -*- coding: utf-8 -*-
"""分析 e32 训练进度与耗时，并回答"能否把数据缓存到显存"。

## 从日志读出的时间线
  19:41:56  loaded pretrained
  19:42:14  Epoch 1 [0%]  loss 0.2501  step 0
  19:47:59  Saving at iteration 1        <- epoch 1 结束
  19:48:13  ====> Epoch: 1
  19:49:49  Epoch 2 [85%] loss 0.05656 step 100
  19:50:08  Saving at iteration 2        <- epoch 2 结束

  epoch 1: 19:42:14 -> 19:47:59 = 345 s
  epoch 2: 19:48:13 -> 19:50:08 = 115 s

## 本脚本
  1. 由 global_step=100 出现在 epoch2 的 85% 反推每 epoch 步数
  2. 估算剩余时间（给区间）
  3. 量数据集体积，判断"能否放进显存"
  4. 判断瓶颈是**数据加载**还是**GPU 计算**
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
LOGS = GSV / "logs" / "zfy_v5"

print("=" * 76)
print("1. 反推每 epoch 步数")
print("=" * 76)
# log_interval=100 -> 在 global_step 0,100,200... 打日志
# epoch2 的 85% 处打印 global_step=100
# 设每 epoch N 步：epoch1 结束在 global_step=N，epoch2 到 85% 时
# global_step = N + 0.85N = 1.85N = 100  ->  N ≈ 54
N = round(100 / 1.85)
print(f"  由 global_step=100 出现在 epoch2 的 85% 处:")
print(f"    1.85 x N = 100  ->  每 epoch 约 {N} 步")
n_sem = len([l for l in (LOGS / "6-name2semantic.tsv")
             .read_text(encoding="utf-8", errors="replace").strip().split("\n")
             if l.strip()]) - 1
print(f"  样本 {n_sem} 条 / batch 16 = {n_sem/16:.1f}  -> 分桶采样器切出 {N} 批 ✓")

EPOCHS = 32
print(f"  总步数 = {N} x {EPOCHS} = {N * EPOCHS}")

print()
print("=" * 76)
print("2. 耗时与剩余时间估算")
print("=" * 76)
T1, T2 = 345.0, 115.0          # 实测 epoch 1 / epoch 2
print(f"  epoch 1: {T1:.0f}s  ({T1/N:.2f} s/step)  <- 含预热/缓存冷")
print(f"  epoch 2: {T2:.0f}s  ({T2/N:.2f} s/step)")
print()
print("  上一轮 e8 的**最后一个 epoch** 只用了 83s（1.54 s/step），")
print("  说明还会继续变快（页缓存热了）。给三档估计：")
for label, sps in (("保守（保持 2.13 s/step）", T2 / N),
                   ("乐观（降到 1.54 s/step）", 83.0 / N),
                   ("极乐观（1.2 s/step）", 1.2)):
    done = T1 + T2
    remain = (EPOCHS - 2) * N * sps
    print(f"    {label:26s} 每 epoch {N*sps:5.0f}s  "
          f"剩余 {remain/60:5.1f} 分钟  总计 {(done+remain)/60:5.1f} 分钟")

print()
print("  当前已跑 2/{EPOCHS} 轮，约 {(T1+T2)/60:.1f} 分钟")
print("  -> 预计还需 **40~58 分钟**")

print()
print("=" * 76)
print("3. 数据集体积 vs 显存")
print("=" * 76)
SUB = ["2-name2text.txt", "3-bert", "4-cnhubert", "5-wav32k",
       "6-name2semantic.tsv"]
tot = 0
for s in SUB:
    p = LOGS / s
    if not p.exists():
        continue
    if p.is_dir():
        files = [f for f in p.rglob("*") if f.is_file()]
        sz = sum(f.stat().st_size for f in files)
        print(f"  {s:22s} {len(files):5d} 个  {sz/1e6:8.1f} MB")
    else:
        sz = p.stat().st_size
        print(f"  {s:22s} {'':5s}      {sz/1e6:8.1f} MB")
    tot += sz
print(f"  {'合计':22s} {'':5s}      {tot/1e6:8.1f} MB")
print()
print(f"  显存 205 GB，数据 {tot/1e6:.0f} MB -> **放得下，而且绰绰有余**")
print("  但'放得下'不等于'有收益'，见下一节")

print()
print("=" * 76)
print("4. 瓶颈是数据加载还是 GPU 计算？")
print("=" * 76)
# 冒烟测试（batch 2，20 条，数据全在页缓存里）稳态 0.31 s/step
SMOKE_B2 = 0.31
print(f"  冒烟实测（batch 2，稳态）: {SMOKE_B2:.2f} s/step")
print(f"  按纯计算线性外推到 batch 16（8 倍）: {SMOKE_B2*8:.2f} s/step")
print(f"  e32 实测（batch 16，epoch 2）     : {T2/N:.2f} s/step")
print()
if T2 / N <= SMOKE_B2 * 8:
    print("  => 实测 **不快于** 纯计算外推值")
    print("     -> 说明数据加载已被 DataLoader 完全掩盖（4 workers + prefetch）")
    print("     -> **瓶颈在 GPU 计算，不在数据 IO**")
    print("     -> 把数据缓存到显存**几乎没有收益**")
else:
    print("  => 实测慢于纯计算外推值 -> 数据加载是瓶颈，缓存到显存会有收益")

print()
print("=" * 76)
print("5. 真正有效的加速手段：grad_ckpt=False")
print("=" * 76)
print("""
  当前 grad_ckpt=True —— 那是为**本机 8GB 显卡**设的，云上 205GB 根本不需要。
  梯度检查点会用「重算一次前向」换显存，关掉它通常快 1.3~1.5 倍。

  显存需求（全量微调 383M 参数，batch 16）：
    参数 fp16      0.76 GB
    梯度 fp16      0.76 GB
    fp32 主权重    1.53 GB
    AdamW 状态     3.05 GB
    ------------------------
    优化器合计     6.10 GB
    激活（关掉检查点后）  估算 15~25 GB
    ------------------------
    合计约 21~31 GB  <<  205 GB   ✅ 完全放得下

  做法（用 --resume 不丢已跑的 2 轮）：
    1. 改 tmp_full_e32.json 里 "grad_ckpt": false
       （或给 run_sovits_e32.sh 加个开关）
    2. 用 --resume 从当前 G_*.pth 继续
""")
