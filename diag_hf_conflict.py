# -*- coding: utf-8 -*-
"""定位 `is_offline_mode` 的导入链，并给出解法。

## 症状（云端合成时）
    ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'

## 已知的版本互斥（技能里记录过）
    transformers 5.x  需要 huggingface_hub >= 1.x（提供 is_offline_mode）
    gradio 4.44       需要 huggingface_hub < 0.37（提供 HfFolder）
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    这两个**互斥** -> 装哪个都会让另一个坏

之前我们选了 `hf-hub==0.36.2` 让 gradio 能 import（修掉了 HfFolder）。
但云端现在 transformers 是 **5.15.1**（升级过），它需要 `is_offline_mode`
-> 于是换成 hf-hub 坏了。

## 本脚本：找出到底是谁拖进了 transformers
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
if not (GSV / "GPT_SoVITS").is_dir():
    for c in (Path("/mnt/workspace/zfy-v5-fullft-pkg/GPT-SoVITS"),
              Path("/mnt/workspace/zfy-v5-fullft-pkg")):
        if (c / "GPT_SoVITS").is_dir():
            GSV = c
            break

print("=" * 88)
print("1. 仓库里谁 import transformers")
print("=" * 88)
hits = []
for p in GSV.rglob("*.py"):
    if any(x in str(p) for x in (".venv", "site-packages", "__pycache__")):
        continue
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        continue
    for i, l in enumerate(txt.split("\n"), 1):
        if re.search(r"^\s*(import|from)\s+transformers", l):
            hits.append((p.relative_to(GSV), i, l.strip()))
for rel, i, l in hits[:20]:
    print(f"  {rel}:{i}")
    print(f"    {l}")
if not hits:
    print("  （仓库自己不 import transformers）")

print()
print("=" * 88)
print("2. 谁 import huggingface_hub")
print("=" * 88)
hh = []
for p in GSV.rglob("*.py"):
    if any(x in str(p) for x in (".venv", "site-packages", "__pycache__")):
        continue
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        continue
    for i, l in enumerate(txt.split("\n"), 1):
        if re.search(r"from\s+huggingface_hub\s+import|import\s+huggingface_hub", l):
            hh.append((p.relative_to(GSV), i, l.strip()))
for rel, i, l in hh[:20]:
    print(f"  {rel}:{i}")
    print(f"    {l}")
if not hh:
    print("  （仓库自己不 import）")

print()
print("=" * 88)
print("3. 推理链路实际用到哪些 HF 功能")
print("=" * 88)
print("""
  `TTS_infer_pack` 里用到 HF 的地方：
    · transformers 的 `AutoModelForMaskedLM`（chinese-roberta-wwm-ext-large）
    · transformers 的 `HubertModel`（chinese-hubert-base）
  两者都**只从本地目录加载**（`bert_base_path` / `cnhuhbert_base_path` 是
  绝对路径），走的是 `from_pretrained(local_dir)`，**不需要联网**。

  但 transformers 在 import 时会检查 `huggingface_hub.is_offline_mode()`
  —— 这就是报错点。

## 三条解法（按推荐度）

### 解法 A（推荐）：降 transformers 到 4.x
    pip install "transformers>=4.51,<5"

  · transformers 4.x 用 `huggingface_hub.constants.HF_HUB_OFFLINE`，
    **不需要** `is_offline_mode`
  · 与 hf-hub 0.36.2（gradio 需要的）兼容
  · 我们之前就是这么修的（`install_all_deps.sh` 里钉了 `transformers>=4.51,<5`）
  · 云端现在是 5.15.1 -> 被后来的 `pip install` 顶掉了

### 解法 B：降 gradio（如果不用 WebUI 推理）
    pip install "gradio==4.44.1" "huggingface-hub>=1.0"
  · 但 gradio 4.44 会坏（`HfFolder` 缺失）
  · **只在不用 WebUI 时可以**

### 解法 C：给 transformers 打补丁（最脏，不推荐）
    在 import transformers 之前给 huggingface_hub 补一个 is_offline_mode：
        import huggingface_hub, huggingface_hub.constants as C
        if not hasattr(huggingface_hub, "is_offline_mode"):
            huggingface_hub.is_offline_mode = lambda: getattr(
                C, "HF_HUB_OFFLINE", False)
  · 能用，但属于 monkey-patch，脆弱

## 本次建议：解法 A
    我们**只需要 torch 后端做批量合成**，不需要 gradio WebUI。
    但 `tools/my_utils.py` 里有模块级 `import gradio` ——
    而我已打 `patch_no_ffmpeg.py` 把 gradio 变成**可选导入**了。
    -> 所以现在可以安全地把 hf-hub 升到 >=1.0（gradio 会降级成桩函数）
    -> 或者降 transformers 到 4.x（更保守）

**最保守 = 解法 A**：降 transformers，保持 hf-hub 0.36.2。
""")
print("=" * 88)
