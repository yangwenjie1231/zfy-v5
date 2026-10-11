# -*- coding: utf-8 -*-
"""诊断 `is_offline_mode` ImportError，并输出本机"可用版本组合"作为基准。

## 报错（云上）
  ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'

## 已知
  · 仓库自身**不** import is_offline_mode（已全仓搜过）
  · 本机 huggingface_hub 0.36.2 —— 顶层/constants/utils **都没有** is_offline_mode
  · 但本机三条链路都能 import

## 所以推断
  `is_offline_mode` 在旧版 huggingface_hub 里存在，新版被移到了别处
  （或改名）。是**某个第三方包**在用旧 API，且它期望的 hf-hub 版本
  比 0.36.2 **更旧**。

  本机能用，是因为本机那些包的版本组合恰好不触发这条路径。

本脚本：
  1. 本机版本组合（作为"已验证可用"的基准）
  2. 云上需要哪些包 —— 从三条链路追踪结果推
  3. 给出**两个方向的修法**（升 hf-hub / 降触发包）
"""
from __future__ import annotations

import importlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 74)
print("1. 本机版本组合（三条链路都能 import，作为基准）")
print("=" * 74)
MODS = ["torch", "torchaudio", "numpy", "pandas", "huggingface_hub",
        "transformers", "tokenizers", "sentencepiece", "funasr",
        "modelscope", "datasets", "peft", "pytorch_lightning",
        "torchmetrics", "gradio", "librosa", "soundfile", "jieba",
        "fast_langdetect", "split_lang", "loguru", "rich", "torchao"]
base = {}
for m in MODS:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        base[m] = v
        print(f"  {m:22s} {v}")
    except Exception as e:  # noqa: BLE001
        print(f"  {m:22s} 缺失 ({type(e).__name__})")

print()
print("=" * 74)
print("2. is_offline_mode 到底在哪")
print("=" * 74)
import huggingface_hub as h
for path in ("huggingface_hub", "huggingface_hub.constants",
             "huggingface_hub.utils", "huggingface_hub.utils._errors"):
    try:
        mod = importlib.import_module(path)
        print(f"  {path:34s} has is_offline_mode: "
              f"{hasattr(mod, 'is_offline_mode')}")
    except Exception as e:  # noqa: BLE001
        print(f"  {path:34s} 导入失败 {type(e).__name__}")
print(f"  本机 hf-hub 版本: {h.__version__}")

print()
print("=" * 74)
print("3. 云上可能的触发者（按三条链路的第三方清单推）")
print("=" * 74)
print("""
  推理链路第三方（追踪器 v2 结果）：
    transformers / huggingface_hub / peft / resampy / torchaudio /
    fast_langdetect / split_lang / jieba / matplotlib / loguru / rich /
    torchao / librosa / ffmpeg / x_transformers / einops / packaging

  其中**会用 hf-hub 旧 API** 的最可能是：
    · transformers  —— 内部有大量 hf-hub 兼容层
    · peft          —— 依赖 transformers
    · funasr        —— 老代码常用 HfFolder / is_offline_mode

  注意：仓库自己没用 is_offline_mode，所以一定是**这些包之一**。

  修法二选一：
    A) 升 huggingface_hub 到有 is_offline_mode 的版本
       —— 但会再次破坏 gradio（它要 HfFolder）
    B) 降触发它的包（很可能是 transformers / funasr）到与本机一致的版本

  **推荐 B**：本机这套版本组合已被验证可用（三条链路都能 import），
  照抄最稳。
""")
print("=" * 74)
print("4. 建议钉住的版本（照抄本机可用组合）")
print("=" * 74)
PIN = ["huggingface_hub", "transformers", "tokenizers", "peft",
       "pytorch_lightning", "torchmetrics", "gradio", "numpy", "pandas",
       "funasr", "modelscope"]
for m in PIN:
    if m in base:
        print(f'    "{m.replace("_", "-")}=={base[m]}"')
