# -*- coding: utf-8 -*-
"""核查 Notebook 环境（py3.12 + torch2.12 + ROCm7.2.3）的兼容性风险。

## 环境差异

    本机（已跑通）      Notebook（目标）
    py 3.13.13         py 3.12
    torch 2.7.1+cu128  torch 2.12.0（ROCm 版）
    CUDA 12.8          ROCm 7.2.3

torch 2.12 相对 2.7 跨了 5 个大版本，**部分 API 可能已移除**。

## 已发现的两个硬依赖（会在训练前就崩）

1. **`from torch.cuda.amp import ...`（5 处）**
   `torch.cuda.amp` 自 2.4 起弃用，官方计划移除。
   若 2.12 已删，则 import 直接 ImportError。
   涉及：models.py:23、s2_train_v5_full.py:20（我的脚本已加 try/except 回退）

2. **`numpy<2.0`**
   requirements 钉死 numpy<2.0，但 **py3.12 + 新 scipy/librosa 通常要 numpy>=2**。
   强行降级 numpy 可能与预装环境冲突。

## 本脚本

不猜测，逐项报告"需要验证什么"，并生成一个**自检脚本**丢到 Notebook 里跑。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")

print("=" * 78)
print("一、torch.cuda.amp 的 5 处硬 import")
print("=" * 78)
hits = []
for f in GSV.rglob("*.py"):
    try:
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace")
                                 .split("\n"), 1):
            if "from torch.cuda.amp import" in line:
                hits.append((f.relative_to(GSV), i, line.strip()))
    except Exception:  # noqa: BLE001
        pass
for rel, i, line in hits:
    mark = "  (我的脚本，已有 try/except 回退)" if "v5_full" in str(rel) else "  **无回退**"
    print(f"  {rel}:{i}")
    print(f"      {line}{mark}")

print()
print("=" * 78)
print("二、`models.py` 里的 torch.cuda.amp")
print("=" * 78)
mp = GSV / "GPT_SoVITS" / "module" / "models.py"
src = mp.read_text(encoding="utf-8", errors="replace")
for i, line in enumerate(src.split("\n"), 1):
    if "cuda.amp" in line or "autocast" in line:
        print(f"  L{i}: {line.strip()}")
print()
print("  models.py 是**模型定义**，训练和推理都会 import 它。")
print("  若 torch 2.12 移除了 torch.cuda.amp，这里会直接崩。")

print()
print("=" * 78)
print("三、numpy 版本冲突")
print("=" * 78)
print("""
  requirements.txt: numpy<2.0
  但 py3.12 + 新版 scipy/librosa/numba 通常要求 numpy>=2

  本机实测（py3.13）：numpy 2.1.3 工作正常，且**必须**用 2.x ——
  之前 numpy 1.26.4 曾导致 `OverflowError: cannot convert longdouble
  infinity to integer`。

  => requirements 里的 numpy<2.0 可能是过时约束，建议**不要盲目降级**，
     优先保留环境预装的 numpy。
""")

print("=" * 78)
print("四、建议：先在 Notebook 跑自检")
print("=" * 78)
print("""
  我生成了一个自检脚本 `notebook_preflight.py`，丢到 Notebook 里先跑，
  它会逐项报告：torch/ROCm 版本、关键 API 是否存在、依赖是否齐备、
  f5_tts 是否解析正确。**在下载 2.77GB 包之前就能发现问题。**
""")
