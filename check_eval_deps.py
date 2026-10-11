# -*- coding: utf-8 -*-
"""检查 eval_objective.py 的依赖是否齐全。"""
from __future__ import annotations

import importlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NEED = ["torch", "torchaudio", "soundfile", "numpy", "scipy",
        "opencc", "faster_whisper", "librosa"]
bad = []
for m in NEED:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  OK   {m:16s} {v}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {m:16s} {type(e).__name__}")
        bad.append(m)

print()
if bad:
    print(f"  缺: {bad}")
    print("  装：pip install " + " ".join(
        {"faster_whisper": "faster-whisper", "opencc": "opencc-python-reimplemented",
         "soundfile": "soundfile"}.get(b, b) for b in bad))
else:
    print("  ✓ 依赖齐全")
sys.exit(1 if bad else 0)
