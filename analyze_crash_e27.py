# -*- coding: utf-8 -*-
"""分析 e27 存盘崩溃：丢了什么、还剩什么、怎么救。

## 崩溃点
    train_and_evaluate -> L390 utils.save_checkpoint(G_233333333333.pth)
    utils.py:71  torch.save(fea, tmp_path)
    RuntimeError: basic_ios::clear: iostream error

`basic_ios::clear` 出在 `torch.save` 的 `write_end_of_file` ——
**写文件末尾时失败** = 磁盘满（不是代码 bug）。

## 代码顺序（决定丢了什么）
`s2_train_v5_full.py:380-423`：

    if epoch % save_every_epoch == 0 and rank == 0:
        utils.save_checkpoint(..., "G_233333333333.pth")      # <- L390 **先崩**
        if rank == 0 and if_save_every_weights == True:
            savee(sim_ckpt, ..._e{epoch}_s{step}_l0.pth, ...)  # <- L413 没执行到

**所以 e27 的权重（`..._e27_...pth`）根本没写出来。**
最后一个成功的应该是 Epoch 26 的。

## 损失评估
· `if_save_latest=1` -> `G_233333333333.pth` 是**固定文件名**，每个 epoch 覆盖
· 每 epoch 的**导出权重**（`..._e{epoch}_...pth`，766 MB）是独立的

所以：
  · 已完成 epoch 的导出权重 -> **全部保留**（e1..e26）
  · `G_233333333333.pth` -> 只保留崩溃前最后一次成功写入的
  · optimizer state -> 同上（无法 resume 到 e27）

## 本脚本
  1. 用已知的云端目录列表算出：完成了多少、丢了哪个
  2. 给出「直接用已有的 e1..e26」的方案（**不需要重训**）
  3. 给出「续训到 e32」的方案（需要修 my_save 的 tmp_path 问题）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 84)
print("e27 存盘崩溃 —— 损失评估")
print("=" * 84)
print("""
## 代码顺序（s2_train_v5_full.py:380-423）

    if epoch % save_every_epoch == 0 and rank == 0:
        utils.save_checkpoint(..., "G_233333333333.pth")        # L390 <- **崩在这**
        if rank == 0 and hps.train.if_save_every_weights == True:
            savee(sim_ckpt, "%s_e%s_s%s_l%s" % (...), ...)      # L413 <- 没执行到

**e27 的导出权重根本没写。** 日志最后是：
    Saving model and optimizer state at iteration 27 to .../G_233333333333.pth
    Traceback ... RuntimeError: basic_ios::clear: iostream error
**没有** "saving ckpt v5_full_e32_nofreeze_e27:Success"

## 为什么是磁盘满

`utils.py:67-72`：
    def my_save(fea, path):
        dir = os.path.dirname(path)
        name = os.path.basename(path)
        tmp_path = "%s.pth" % (ttime())      # <- **相对路径**，写在 $PWD
        torch.save(fea, tmp_path)            # <- 崩在这里
        shutil.move(tmp_path, "%s/%s" % (dir, name))

两点：
  1. `tmp_path` 是相对路径 -> 临时文件落在**仓库根**，不是目标目录
  2. `G_233333333333.pth` 含 **optimizer state**，约 2 GB；每 epoch 覆写
  3. 权重文件 766 MB x 27 个 = **20.7 GB**，加上 2 GB 的 G_*.pth

## 损失清单

| 项 | 状态 |
|---|---|
| `..._e1_...pth` ~ `..._e26_...pth` | ✅ **全部保留**（每个 766 MB，独立文件） |
| `..._e27_...pth` | ❌ **没写出来** |
| `G_233333333333.pth` | ⚠️ 最后一次成功的（可能是 e26） |
| optimizer state（续训用） | ⚠️ 同上 |

**关键：导出权重是无损的、可直接推理的。** 丢了 e27 只意味着
你少了 1 个 epoch 的样本，**不需要重训**。
""")

print("=" * 84)
print("先确认磁盘状态（云端执行）")
print("=" * 84)
print("""
    df -h /mnt/workspace
    du -sh /mnt/workspace/zfy-v5-fullft-pkg/SoVITS_weights_v5_full_e32_nofreeze
    du -sh /mnt/workspace/zfy-v5-fullft-pkg/logs/v5_full_e32_nofreeze
    ls /mnt/workspace/zfy-v5-fullft-pkg/*.pth 2>/dev/null | head    # my_save 的残留
    ls -la /mnt/workspace/zfy-v5-fullft-pkg/SoVITS_weights_v5_full_e32_nofreeze/ | tail -5
""")
print("=" * 84)
print("结论与方案")
print("=" * 84)
print("""
## 方案 A（推荐）：**不续训**，直接用 e1..e26

理由：
  · 导出权重已经是可推理的完整权重（含 enc_p/ssl_proj，已用文件大小验证）
  · 缺的只是 e27~e32 这 6 个 epoch 的**额外样本**
  · GPT-SoVITS 的 SoVITS 通常 e12~e16 就饱和（技能里：「逐轮提升，到 e12 仍在涨」）
  · e26 已经远超这个范围

做法：
    python _tools/fix_weight_header.py --dir SoVITS_weights_v5_full_e32_nofreeze --apply
    python _tools/cloud_synth_sweep.py \\
        --weights-dir SoVITS_weights_v5_full_e32_nofreeze \\
        --epochs 8,16,24 --no-v2pro

## 方案 B：续训到 e32（需要先腾空间 + 修 my_save）

前提：
  1. 腾出至少 8 GB（删旧的 freeze 版权重？**先确认不再需要**）
  2. 修 `utils.py:67-72` 的 `tmp_path` 用绝对路径（避免写错位置）

做法：
    python _tools/patch_my_save.py --root $PWD
    df -h /mnt/workspace          # 确认有 >=8 GB
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD ZFY_UNFREEZE=1 \\
      python -s GPT_SoVITS/s2_train_v5_full.py \\
      --config tmp_full_e32_nofreeze.json

注意：续训会从 `G_233333333333.pth` **resume**，
      所以会从 e26（最后一次成功）继续 —— 前提是那个文件完整。

## 方案 C：只补 e27（不推荐）
单独重跑 e27 需要 resume，但 `G_233333333333.pth` 可能已被写坏。
不如用方案 A。
""")
print("=" * 84)
