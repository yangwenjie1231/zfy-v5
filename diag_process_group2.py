# -*- coding: utf-8 -*-
"""拿完整 traceback —— 定位 `init_embed_` 里**剩余**的无守卫调用。

## 进展
打完 `patch_core_vq_dist.py` 后，kmeans **跑起来了**
（打印了 `kmeans start ...` 和进度条），说明 L165/170/171 的守卫生效。

但仍然抛 `Default process group has not been initialized`。
-> 说明 `init_embed_` 里**还有别的**无守卫调用。

## 候选
`core_vq.py:178`：
    broadcast_tensors(self.buffers())
它来自 `module/distrib.py`。而 distrib.py 里：
    def broadcast_tensors(tensors, src=0):
        ...
（L57 开始，需看是否在内部直接调 dist.xxx 而无守卫）

另外 `core_vq.py:211`：
    embed_ind = dist.max(dim=-1).indices
`dist` 在这里可能是**局部变量**（`x` 的别名），不是 torch.distributed。
需确认。

## 本脚本：完整 traceback + 逐行排查
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
sys.path[:0] = [str(GSV), str(GSV / "GPT_SoVITS")]
os.chdir(str(GSV))

import json  # noqa: E402

import torch  # noqa: E402

print("=" * 86)
print("1. distrib.broadcast_tensors 的实现")
print("=" * 86)
import inspect  # noqa: E402

from module import distrib  # noqa: E402

print(inspect.getsource(distrib.broadcast_tensors))
print("--- 依赖的 _check_number_of_params ---")
print(inspect.getsource(distrib._check_number_of_params))

print()
print("=" * 86)
print("2. core_vq.py 里所有 dist.* 调用及行号")
print("=" * 86)
cq = (GSV / "GPT_SoVITS" / "module" / "core_vq.py").read_text(
    encoding="utf-8").split("\n")
for i, l in enumerate(cq, 1):
    if "dist." in l or "broadcast_tensors" in l:
        print(f"  L{i:4d}: {l.rstrip()}")

print()
print("=" * 86)
print("3. 完整 traceback")
print("=" * 86)
cfg = json.loads(
    (GSV / "GPT_SoVITS" / "configs" / "s2.json").read_text(encoding="utf-8"))
from module.models import SynthesizerTrnV3  # noqa: E402

m = SynthesizerTrnV3(
    cfg["data"]["filter_length"] // 2 + 1,
    cfg["train"]["segment_size"] // cfg["data"]["hop_length"],
    n_speakers=cfg["data"]["n_speakers"], **cfg["model"])

x = torch.randn(1, 768, 100)
try:
    q, codes, commit, ql = m.quantizer(x, layers=[0])
    print(f"  ✓ 成功 codes.shape={tuple(codes.shape)}")
except Exception:  # noqa: BLE001
    tb = traceback.format_exc()
    # 只打印与 core_vq / distrib 相关的帧
    for line in tb.split("\n"):
        if any(k in line for k in ("core_vq", "distrib", "File \"",
                                   "Error", "error", "raise", "dist.")):
            print(f"  {line}")
print("=" * 86)
