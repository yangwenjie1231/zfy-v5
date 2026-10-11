# -*- coding: utf-8 -*-
"""决策：续训到 e32，还是直接用 e1~e26？

## 背景
配额满导致 e27 存盘崩溃。已删 45 GB（旧 freeze 版权重）+ 4.3 GB（my_save 垃圾），
现 54 GB。`G_233333333333.pth`（e26，4.58 GB）完好 -> **技术上可以续训**。

## 续训会丢什么（关键！）
`s2_train_v5_full.py:397-423` 的导出逻辑：

    if rank == 0 and hps.train.if_save_every_weights == True:
        for key in ckpt:
            if key not in no_grad_names:
                sim_ckpt[key] = ckpt[key].half().cpu()
        savee(sim_ckpt, "%s_e%s_s%s_l%s" % (hps.name, epoch, global_step, lora_rank), ...)

而 `global_step` 在 resume 时：

    epoch_str += 1                                      # L219
    global_step = (epoch_str - 1) * len(train_loader)   # L220

`epoch_str` 来自 checkpoint 里存的 iteration（e26 -> 26），
所以续训后从 **e27** 开始，global_step = 26 × 54 = 1404。

**文件名是 `..._e27_s1458_l0.pth`** —— 与之前的 `..._e26_s1404_l0.pth`
**不冲突，不会覆盖**。这点是安全的。

## 但真正的代价
1. **每 epoch 写一个 766 MB 权重** —— 6 个 = 4.6 GB
2. **每 epoch 写一个 4.58 GB 的 `G_233333333333.pth`** —— 覆盖写，峰值需 4.58 GB 余量
   （`my_save` 先写 tmp 再 move -> **瞬时需要 2 倍空间 = 9.2 GB**）
3. 约 **10 分钟**（1.6 分钟/epoch × 6）

## 收益
技能里的经验：「GPT 早期即饱和，SoVITS 越训越好（到 e12 仍在涨）」
-> e26 -> e32 的**边际收益可能很小**

## 本脚本
  1. 列出清点后的现状
  2. 给出两条路的成本/收益对比
  3. 给出**验证续训是否值得**的廉价方法（先用已有 e8/e16/e24 对比）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 88)
print("现状清点（清理后）")
print("=" * 88)
print("""
    54G   .                                   <- 清理前 108G
    ---
    20G   logs                                <- 含 4 个 4.4~4.6 GB 的 G_*.pth
    19G   SoVITS_weights_v5_full_e32_nofreeze <- **成果**（e1~e26，766 MB × 27）
    5.5G  SoVITS_weights_v5_full              <- 早期 LoRA（8 个）
    5.5G  .git
    2.6G  GPT_SoVITS                          <- 底模（必须保留）
    1.2G  GPT_weights_gptA / gptB             <- GPT A/B（各 8 个）
     15M  out_sweep
    279M  ref

## logs/ 里的 G_*.pth（每个 4.4~4.6 GB）

| 路径 | 大小 | 用途 |
|---|---|---|
| `logs/v5_full_e32_nofreeze/...` | 4.58 GB | **续训 resume 点（e26）** |
| `logs/v5_full_e32/...` | 4.46 GB | 旧 freeze 实验（权重已删，无用） |
| `logs/smoke/...` | 4.46 GB | 冒烟测试残留（无用） |
| `logs/zfy_v5_full/...` | 4.35 GB | 旧实验（无用） |

**后三个可删，腾 13.3 GB**（保留 nofreeze 那个）。
""")

print("=" * 88)
print("两条路的成本/收益")
print("=" * 88)
print("""
┌─────────────────────────────────────────────────────────────────┐
│ 路 A：直接用 e1~e26（推荐）                                     │
├─────────────────────────────────────────────────────────────────┤
│ 成本：0                                                         │
│ 收益：e26 是完整可推理的导出权重（含 enc_p/ssl_proj，已验证）    │
│ 风险：无                                                        │
│ 依据：SoVITS 在 e12 就接近饱和，e26 已远超                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 路 B：续训到 e32                                                │
├─────────────────────────────────────────────────────────────────┤
│ 成本：约 10 分钟 + 瞬时 9.2 GB 空间（my_save 先写 tmp 再 move）  │
│       最终多占 4.6 GB（e27~e32 六个权重）                        │
│ 收益：多 6 个 epoch 的样本，**边际收益未知**                     │
│ 风险：若中途再撞配额 -> 又留垃圾 + 又崩                          │
│       且 `G_*.pth` 被覆盖写，崩在中间可能**毁掉 resume 点**      │
└─────────────────────────────────────────────────────────────────┘

## 关键：路 B 的收益是**可先验证的**

已有 e8 / e16 / e24 三个导出权重。先合成对比这三者：
  · 若 e8 与 e24 的相似度**已无明显差异** -> SoVITS 已饱和，路 B 无意义
  · 若 e24 明显优于 e8            -> 还在涨，路 B 值得做

**这个验证只需 25 MB 音频空间，不需要续训。**
""")

print("=" * 88)
print("建议流程")
print("=" * 88)
print("""
## 第 1 步：清掉无用的 G_*.pth（腾 13.3 GB）

    cd /mnt/workspace/zfy-v5-fullft-pkg
    rm -fv logs/smoke/logs_s2_v5dev_full/G_*.pth
    rm -fv logs/v5_full_e32/logs_s2_v5dev_full/G_*.pth
    rm -fv logs/zfy_v5_full/logs_s2_v5dev_full/G_*.pth
    du -sh . && ls logs/v5_full_e32_nofreeze/logs_s2_v5dev_full/

## 第 2 步：加版本头（写 ~770 MB）

    python _tools/fix_weight_header.py --dir SoVITS_weights_v5_full_e32_nofreeze --apply

## 第 3 步：合成 e8 / e16 / e24 —— **先验证饱和**

    python _tools/cloud_synth_sweep.py \\
        --weights-dir SoVITS_weights_v5_full_e32_nofreeze \\
        --epochs 8,16,24 --no-v2pro

## 第 4 步：看结果决定

  · e24 明显优于 e8 -> 走**路 B**（续训到 e32）
  · 看不出差异     -> 走**路 A**（停手，转别的方向）

## 第 5 步（若走路 A）：下载 + 客观评估

    在本机：
    python -c "from modelscope import snapshot_download as d; \\
      d('yangwenjie1231/zfy-v5-fullft-pkg', repo_type='dataset', \\
        allow_patterns=['out_sweep_nofreeze/**'], \\
        local_dir=r'D:\\A\\zfy-v5\\dist\\nofreeze')"

    然后跑 eval_from_wav.py，看相似度能否突破 0.7895 平台期
""")
print("=" * 88)
