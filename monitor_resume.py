# -*- coding: utf-8 -*-
"""续训监控要点（e27 -> e32）。

## 这次续训的关键差异

| | 上次（崩了） | 这次 |
|---|---|---|
| 进程数 | **5 个**（各写一份 4.58 GB ckpt） | **1 个**（`CUDA_VISIBLE_DEVICES=0`） |
| ckpt 命名 | `G_233333333333.pth` 固定名覆盖写 | `G_{global_step}.pth` **唯一名** |
| 临时文件 | 写在 `$PWD`（崩溃留 4.3 GB 垃圾） | 写在**目标同目录**，失败自动清理 |
| 保存方式 | 先写 tmp 再 `shutil.move` | `fsync` + `os.replace` **原子** |
| 崩溃后果 | **旧 ckpt 也可能毁掉** | 只丢当前一个，旧的可 resume |
| 保留策略 | 无（无限累积） | 最新 **2 个**（`ZFY_KEEP_CKPTS`） |

## 空间预算（这次）
    旧 ckpt G_233333333333.pth (e26)     4.58 GB
    新 ckpt G_1404.pth (写入中)           4.58 GB
    导出权重 766 MB × 6 (e27~e32)         4.60 GB
    ─────────────────────────────────────────────
    峰值                                  ≈ 14 GB
    稳定后（保留 2 个 ckpt + 6 个权重）    ≈ 13.8 GB

## 监控命令
    # 进度
    tail -3 /mnt/workspace/zfy-v5-fullft-pkg/train_nofreeze2.log

    # 进程数（应为 1，因为 CUDA_VISIBLE_DEVICES=0）
    ps -eo pid,ppid,etime,cmd | grep "[s]2_train_v5_full" | wc -l

    # checkpoint 命名（应出现唯一的 G_<step>.pth，且只保留 2 个）
    ls -la /mnt/workspace/zfy-v5-fullft-pkg/logs/v5_full_e32_nofreeze/logs_s2_v5dev_full/

    # 导出权重
    ls /mnt/workspace/zfy-v5-fullft-pkg/SoVITS_weights_v5_full_e32_nofreeze/ | tail -8

    # 磁盘
    du -sh /mnt/workspace/zfy-v5-fullft-pkg
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print(__doc__)
print("=" * 88)
print("预期时间线（约 1.5 分钟/epoch，6 个 epoch）")
print("=" * 88)
print("""
    e27  ── 1.5 min ──> e28  ── 1.5 min ──> e29
    e30  ── 1.5 min ──> e31  ── 1.5 min ──> e32 完成
    预计总耗时：约 9~12 分钟
""")
print("=" * 88)
print("完成后立刻做的三件事")
print("=" * 88)
print("""
## ① 验证权重含 enc_p（用文件大小，非侵入）
    python _tools/verify_by_filesize.py \\
        --freeze <旧目录已删，用已知数值> \\
        --nofreeze SoVITS_weights_v5_full_e32_nofreeze
    # 或数键数：
    python _tools/verify_encp_in_weights.py \\
        --dir SoVITS_weights_v5_full_e32_nofreeze --max 3
    # 判据：总键数 >= 660（freeze 版 444），enc_p=235，ssl_proj=2

## ② 加版本头（必须，否则推理当 v3 加载 24k BigVGAN）
    python _tools/fix_weight_header.py \\
        --dir SoVITS_weights_v5_full_e32_nofreeze --apply

## ③ 合成 e8/e16/e24 对比（验证是否饱和）
    python _tools/cloud_synth_sweep.py \\
        --weights-dir SoVITS_weights_v5_full_e32_nofreeze \\
        --epochs 8,16,24 --no-v2pro
""")
print("=" * 88)
print("⚠️ 需要盯的两个信号")
print("=" * 88)
print("""
1. **`grep -c "Disk quota exceeded" train_nofreeze2.log` 必须是 0**
   （上次就是配额满崩的）

2. **checkpoint 目录应只保留 2 个 `G_*.pth`**
   若累积到 3 个以上 -> `ZFY_KEEP_CKPTS` 的清理逻辑没生效
   若只剩 1 个且不是最新的 -> 需检查

   健康的样子：
       G_1404.pth        4.58 GB   <- e26（旧）
       G_1458.pth        4.58 GB   <- e27（新）
   下一次保存后变成：
       G_1458.pth        4.58 GB
       G_1512.pth        4.58 GB
   （`G_233333333333.pth` 会被当作最旧清掉）
""")
print("=" * 88)
