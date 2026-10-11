# -*- coding: utf-8 -*-
"""定位 `HfFolder` ImportError 的来源，并确认正确的 huggingface_hub 版本。

## 报错
  ImportError: cannot import name 'HfFolder' from 'huggingface_hub'

## 已知
  · 仓库自身**不直接** import HfFolder（只用了 PyTorchModelHubMixin /
    hf_hub_download / snapshot_download）
  · 本地环境 huggingface_hub 0.36.2 —— `hasattr(h, 'HfFolder')` 为 True
  · 所以是**某个第三方包**（funasr / modelscope / transformers）在用旧 API，
    而云上装的 huggingface_hub 版本比 0.36 新，把它移除了

## 本脚本
  1. 遍历已装包，找出谁 import HfFolder
  2. 报告本机 huggingface_hub 版本（作为"能用"的基准）
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 74)
print("1. 本机 huggingface_hub（作为可用基准）")
print("=" * 74)
try:
    import huggingface_hub as h
    print(f"  版本: {h.__version__}")
    print(f"  有 HfFolder: {hasattr(h, 'HfFolder')}")
except Exception as e:  # noqa: BLE001
    print(f"  导入失败: {e}")

print()
print("=" * 74)
print("2. 相关包的版本")
print("=" * 74)
for m in ("torch", "transformers", "datasets", "funasr", "modelscope",
          "peft", "pytorch_lightning", "torchmetrics", "gradio",
          "numpy", "pandas"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:20s} {getattr(mod, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"  {m:20s} 导入失败 ({type(e).__name__})")

print()
print("=" * 74)
print("3. 谁在源码里 import HfFolder（扫 site-packages）")
print("=" * 74)
import site
hits = []
roots = []
try:
    roots += list(site.getsitepackages())
except Exception:  # noqa: BLE001
    pass
try:
    roots.append(site.getusersitepackages())
except Exception:  # noqa: BLE001
    pass
for r in roots:
    p = Path(r)
    if not p.is_dir():
        continue
    for f in p.rglob("*.py"):
        if "huggingface_hub" in str(f):
            continue
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        if "HfFolder" in txt:
            for i, line in enumerate(txt.split("\n"), 1):
                if "HfFolder" in line and ("import" in line):
                    hits.append((str(f.relative_to(p)), i, line.strip()))
if hits:
    print(f"  找到 {len(hits)} 处：")
    for rel, ln, line in hits[:20]:
        print(f"    {rel}:{ln}")
        print(f"        {line[:100]}")
else:
    print("  site-packages 里没找到（可能是间接依赖或已修复）")

print()
print("=" * 74)
print("4. 结论与修法")
print("=" * 74)
print("""
  `HfFolder` 在 huggingface_hub **0.36 之后被移除**。
  本机能跑是因为正好是 0.36.2（还有这个符号）。

  云上刚才 `install_all_deps.sh` **没有钉 huggingface_hub 版本**，
  pip 装了更新的版本 -> 某个包（funasr/modelscope/transformers 之一）
  在 import 时失败。

  修法：把 huggingface_hub 钉到 0.36.x。
    pip install "huggingface-hub==0.36.2"
""")
