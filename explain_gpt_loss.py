# -*- coding: utf-8 -*-
"""修正 loss 读法：`reduction="sum"` 下 22 到底意味着什么。

## 代码事实（t2s_model.py:440-448）
    loss_1 = F.cross_entropy(logits.permute(0,2,1), targets, reduction="sum")
    loss_2 = F.cross_entropy(...)                       # 第二个头
    loss = loss_1 + loss_2
    return loss, acc

**`reduction="sum"`** —— 返回的是整个 batch 所有 token 的**损失之和**，
不是平均值。所以：

    真实 per-token loss = total_loss / (batch 内 token 数)

## 关键：分母是 token 数，不是步数
我上一轮算的 `2266 / 103 步 = 22` **是错的** —— 分母应该是 token 数。

batch_size=8，GPT 的序列长度 = 文本 token + 语义 token（音频帧数）。
每条样本约 100~400 token（63.5 分钟 / 825 条 ≈ 4.6s/条 ≈ 115 语义帧）。

设平均每 batch 的 token 数 = T：
    per-token loss = total_loss_epoch / T

## 本脚本
  1. 从语料实测平均 token 数
  2. 反推 per-token loss
  3. 与理论基准（ln(1025) ≈ 6.93 是随机猜测的 CE）对比
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
SEM = GSV / "logs" / "zfy_v5" / "6-name2semantic.tsv"
PHO = GSV / "logs" / "zfy_v5" / "2-name2text.txt"

print("=" * 78)
print("1. 语料的 token 数（决定 loss 的分母）")
print("=" * 78)

sem_lens, pho_lens = [], []
if SEM.exists():
    lines = [l for l in SEM.read_text(encoding="utf-8", errors="replace")
             .strip().split("\n") if l.strip()]
    for l in lines[1:]:
        parts = l.split("\t")
        if len(parts) >= 2:
            sem_lens.append(len(parts[1].split()))
if PHO.exists():
    for l in PHO.read_text(encoding="utf-8", errors="replace").strip().split("\n"):
        if not l.strip():
            continue
        parts = l.split("\t")
        if len(parts) >= 2:
            pho_lens.append(len(parts[1].split()))

import statistics  # noqa: E402

if sem_lens:
    print(f"  语义 token: {len(sem_lens)} 条，"
          f"平均 {statistics.mean(sem_lens):.0f}，"
          f"中位 {statistics.median(sem_lens):.0f}，"
          f"最大 {max(sem_lens)}")
if pho_lens:
    print(f"  音素 token: {len(pho_lens)} 条，"
          f"平均 {statistics.mean(pho_lens):.0f}，"
          f"中位 {statistics.median(pho_lens):.0f}")

if sem_lens and pho_lens:
    per_sample = statistics.mean(sem_lens) + statistics.mean(pho_lens)
    print(f"\n  每条样本合计 token ≈ {per_sample:.0f}")
    print(f"  batch_size=8 -> 每 batch ≈ {per_sample*8:.0f} token")

print()
print("=" * 78)
print("2. 修正后的 per-token loss")
print("=" * 78)
TOTAL_LOSS_EPOCH = 2266.368          # 日志里的 total_loss_epoch
BATCH = 8
if sem_lens and pho_lens:
    per_batch = (statistics.mean(sem_lens) + statistics.mean(pho_lens)) * BATCH
    # 注意：loss 是两步（loss_1 + loss_2），所以除以 2 个头的 token 总数
    per_token = TOTAL_LOSS_EPOCH / per_batch
    print(f"  total_loss_epoch = {TOTAL_LOSS_EPOCH}")
    print(f"  每 batch token 数 ≈ {per_batch:.0f}")
    print(f"  **per-token loss ≈ {per_token:.4f}**")
    print()
    import math
    rand = math.log(1025)
    print(f"  随机猜测的 CE = ln(1025) = {rand:.4f}")
    print(f"  比值: {per_token/rand:.3f}x  "
          f"（<1 说明优于随机，越小越好）")
    print()
    if per_token < 1.0:
        print("  ✓ per-token loss < 1.0 —— **训练良好**")
    elif per_token < 3.0:
        print("  ~ per-token loss 1~3 —— 正常范围")
    else:
        print("  !! per-token loss > 3 —— 偏高，需检查")

print()
print("=" * 78)
print("3. 我上一轮的错误")
print("=" * 78)
print("""
  我写的是：
      total_loss_epoch 2266 / 103 步 ≈ 22   <- 错

  错在把**步数**当分母。但 `reduction="sum"` 的分母是 **token 数**：
      2266 / (每 batch 约 1000+ token) ≈ 2 左右   <- 这才是真实量级

  而且 loss = loss_1 + loss_2（两个头），所以还要注意是"双头之和"。

  ## 教训
  "loss 量级偏大" 这个判断，必须先确认 `reduction` 是 sum 还是 mean。
  我直接按 mean 的直觉去读 sum 的值，差点误判成"LR 太大"。
""")

print("=" * 78)
print("4. 真正的判据：loss 是否在降 + acc 是否在升")
print("=" * 78)
print(f"""
  日志只给了最后一个 epoch：
      total_loss_epoch: 2266.368
      top_3_acc_epoch:  0.585

  top_3_acc = 0.585 是**强证据**：
      随机猜 = 3/1025 = 0.29%
      实测 58.5% -> 模型确实在预测，不是随机

  但要看**趋势**必须读 TensorBoard。命令：
      cd /mnt/workspace/zfy-v5-fullft-pkg && python -c "
from tensorboard.backend.event_processing import event_accumulator
import glob
for tag in ['A','B']:
    fs = glob.glob(f'logs/gpt{{tag}}/**/events.out.*', recursive=True)
    if not fs: print(f'[{{tag}}] 无 event'); continue
    ea = event_accumulator.EventAccumulator(sorted(fs)[-1]); ea.Reload()
    print(f'--- [{{tag}}] ---')
    for k in ea.Tags().get('scalars', []):
        if 'loss' in k or 'acc' in k:
            vs = ea.Scalars(k)
            step = max(1, len(vs)//10)
            print(f'  {{k}}: {{[(s.step, round(s.value,3)) for s in vs[::step]]}}')
"
""")
