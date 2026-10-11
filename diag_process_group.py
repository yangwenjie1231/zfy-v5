# -*- coding: utf-8 -*-
"""定位「Default process group has not been initialized」的**真实**调用点。

## 症状
云端训练日志：
    INFO:v5_full_e32_nofreeze:Train Epoch: 1 [0%]
    INFO:v5_full_e32_nofreeze:[0.22658592462539673, 0, 9.99875e-05]
      7%|  | 4/54 [01:06<13:26, 16.14s/it]
    ...
    ValueError: Default process group has not been initialized,
                please make sure to call init_process_group.

**关键**：跑到了第 4 步才崩（前 3 步正常），说明不是启动期问题。

## 我的怀疑链
1. `core_vq.py:250 if is_distributed(): dist.all_reduce(...)`
   —— 但 `is_distributed()` = `world_size() > 1`，
      而 `world_size()` 在未初始化时返回 1 -> False -> 不该进。
   **除非** `torch.distributed.is_initialized()` 返回 True 但 world_size 也是 1
   （单进程 gloo 组），此时 `is_distributed()` 仍是 False。

2. 所以崩的地方**不是** core_vq，而是别处直接调 `dist.xxx`。
   要用完整 traceback 定位。

## 本脚本
  · 复现：单进程构造模型 + 解冻 + 走一次量化器前向
  · 打印 `is_initialized()` / `get_world_size()` 的真实值
  · 若复现不了，说明与 DDP/训练循环有关，需要完整日志
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
sys.path[:0] = [str(GSV), str(GSV / "GPT_SoVITS")]
os.chdir(str(GSV))

import torch  # noqa: E402

print("=" * 84)
print("1. torch.distributed 状态")
print("=" * 84)
print(f"  is_available      : {torch.distributed.is_available()}")
print(f"  is_initialized    : {torch.distributed.is_initialized()}")
try:
    print(f"  get_world_size    : {torch.distributed.get_world_size()}")
except Exception as e:  # noqa: BLE001
    print(f"  get_world_size    : 抛 {type(e).__name__}: {e}")
try:
    print(f"  get_rank          : {torch.distributed.get_rank()}")
except Exception as e:  # noqa: BLE001
    print(f"  get_rank          : 抛 {type(e).__name__}: {e}")

from module.distrib import is_distributed, world_size  # noqa: E402

print(f"  distrib.world_size()    : {world_size()}")
print(f"  distrib.is_distributed(): {is_distributed()}")

print()
print("=" * 84)
print("2. 单进程构造模型 + 解冻 + 走量化器前向")
print("=" * 84)
os.environ["ZFY_UNFREEZE"] = "1"
import json  # noqa: E402

cfg = json.loads(
    (GSV / "GPT_SoVITS" / "configs" / "s2.json").read_text(encoding="utf-8"))
from module.models import SynthesizerTrnV3  # noqa: E402

m = SynthesizerTrnV3(
    cfg["data"]["filter_length"] // 2 + 1,
    cfg["train"]["segment_size"] // cfg["data"]["hop_length"],
    n_speakers=cfg["data"]["n_speakers"], **cfg["model"])
print(f"  freeze_quantizer = {m.freeze_quantizer}")
print(f"  quantizer.training = {m.quantizer.training}")
print(f"  enc_p requires_grad: "
      f"{next(m.enc_p.parameters()).requires_grad}")

# 直接调量化器（模拟 core_vq.py:246 的 training 分支）
x = torch.randn(1, 768, 100)
print()
print("  调 quantizer(x) ...")
try:
    q, codes, commit, ql = m.quantizer(x, layers=[0])
    print(f"  ✓ 成功  codes.shape={tuple(codes.shape)}")
except Exception as e:  # noqa: BLE001
    import traceback
    print(f"  ✗ {type(e).__name__}: {e}")
    traceback.print_exc()

print()
print("=" * 84)
print("3. 分析")
print("=" * 84)
print("""
  `distrib.is_distributed()` = `world_size() > 1`。
  未初始化时 world_size() 返回 1 -> is_distributed() 返回 **False**
  -> core_vq.py:250 的 `if is_distributed():` 不会进
  -> **不会**触发 "Default process group has not been initialized"。

  所以云端那个 ValueError **不是** core_vq 的 all_reduce。

  需要完整 traceback 才能定位。云端命令：
      grep -n -B5 -A40 "Default process group" train_nofreeze.log | head -80
""")
print("=" * 84)
