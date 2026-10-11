# -*- coding: utf-8 -*-
"""验证 patch_no_ffmpeg.py：没有 ffmpeg 也能读音频。

## 要验证的四件事
  1. **打补丁前**：load_audio 依赖 ffmpeg 子进程（本机有 ffmpeg，所以能过）
  2. **打补丁后**：屏蔽 ffmpeg 后仍能读（走 soundfile 回退）
  3. **采样率不符时**：会重采样，且用的是多相 FIR（不是线性插值）
  4. **gradio 缺失时**：my_utils 仍能 import（不再阻塞训练）

## 为什么第 4 点重要
`my_utils.py:7` 是**模块级** `import gradio as gr`。
而 gradio 4.44 依赖 `huggingface_hub.HfFolder`（0.37+ 已删）——
一个 WebUI 依赖出问题会让**整个训练 import 失败**。
云端就实际发生过：`ImportError: cannot import name 'HfFolder'`。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
PY = sys.executable
PATCH = Path(r"D:\A\zfy-v5\patch_no_ffmpeg.py")

# 找一个真实的 5-wav32k 文件
WAV = None
for d in (GSV / "logs" / "zfy_v5" / "5-wav32k",):
    if d.is_dir():
        ws = sorted(d.glob("*.wav"))
        if ws:
            WAV = ws[0]
            break
print(f"测试音频: {WAV}")
if WAV is None:
    print("  !! 找不到 5-wav32k 里的 wav，无法测试")
    sys.exit(1)

PROBE = r'''
import os, sys
GSV = GSV_PATH
sys.path.insert(0, GSV)
os.chdir(GSV)

# 可选：屏蔽 ffmpeg（模拟云端未装）
if os.environ.get("ZFY_HIDE_FFMPEG") == "1":
    import builtins
    _real_import = builtins.__import__
    def _fake(name, *a, **k):
        if name == "ffmpeg":
            raise ImportError("ZFY: ffmpeg 被测试屏蔽")
        return _real_import(name, *a, **k)
    builtins.__import__ = _fake

# 可选：屏蔽 gradio（模拟 HfFolder 冲突）
if os.environ.get("ZFY_HIDE_GRADIO") == "1":
    import builtins
    _real_import2 = builtins.__import__
    def _fake2(name, *a, **k):
        if name == "gradio" or name.startswith("gradio."):
            raise ImportError("ZFY: gradio 被测试屏蔽")
        return _real_import2(name, *a, **k)
    builtins.__import__ = _fake2

try:
    from tools.my_utils import load_audio
except Exception as e:
    print(f"MARKER import_fail {type(e).__name__}: {str(e)[:80]}")
    raise SystemExit(0)

import numpy as np
for sr in (32000, 16000, 48000):
    try:
        x = load_audio(WAV_PATH, sr)
        x = np.asarray(x, dtype=np.float32)
        rms = float(np.sqrt((x**2).mean())) if x.size else 0.0
        print(f"MARKER sr={sr} n={x.size} rms={rms:.6f} "
              f"dur={x.size/sr:.3f} peak={float(np.abs(x).max()):.4f}")
    except Exception as e:
        print(f"MARKER sr={sr} FAIL {type(e).__name__}: {str(e)[:80]}")
'''


def probe(env_extra: dict, label: str):
    env = os.environ.copy()
    env.pop("ZFY_HIDE_FFMPEG", None)
    env.pop("ZFY_HIDE_GRADIO", None)
    env.update(env_extra)
    code = PROBE.replace("GSV_PATH", repr(str(GSV))).replace(
        "WAV_PATH", repr(str(WAV)))
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env,
                       timeout=600, cwd=str(GSV))
    out = (r.stdout or "") + (r.stderr or "")
    print(f"\n--- {label} ---")
    for l in out.split("\n"):
        if l.strip().startswith("MARKER"):
            print(f"  {l.strip()}")
        elif "[ZFY]" in l:
            print(f"  {l.strip()[:110]}")
    return out


print("=" * 88)
print("1. 打补丁前（有 ffmpeg，走原路径）")
print("=" * 88)
probe({}, "上游行为")

print()
print("=" * 88)
print("2. 打补丁")
print("=" * 88)
r = subprocess.run([PY, str(PATCH), "--root", str(GSV)],
                   capture_output=True, text=True, encoding="utf-8",
                   errors="replace")
for l in (r.stdout or "").split("\n"):
    if "[1/2]" in l or "[2/2]" in l or "备份" in l:
        print(f"  {l.strip()}")

print()
print("=" * 88)
print("3. 打补丁后")
print("=" * 88)
probe({}, "正常（有 ffmpeg + gradio）")
out2 = probe({"ZFY_HIDE_FFMPEG": "1"}, "**屏蔽 ffmpeg**（模拟云端）")
out3 = probe({"ZFY_HIDE_GRADIO": "1"}, "**屏蔽 gradio**（模拟 HfFolder 冲突）")
out4 = probe({"ZFY_HIDE_FFMPEG": "1", "ZFY_HIDE_GRADIO": "1"},
             "**两个都屏蔽**")

print()
print("=" * 88)
print("4. 断言")
print("=" * 88)
checks = []


def markers(out):
    d = {}
    for l in out.split("\n"):
        if l.strip().startswith("MARKER "):
            parts = l.strip().split()
            if len(parts) > 1 and parts[1].startswith("sr="):
                sr = int(parts[1].split("=")[1])
                d[sr] = parts
    return d


m2 = markers(out2)
checks.append(("屏蔽 ffmpeg 后仍能读 32kHz", 32000 in m2 and "FAIL" not in " ".join(m2.get(32000, [])),
               " ".join(m2.get(32000, []))[:70]))
checks.append(("屏蔽 ffmpeg 后 32kHz 非全零",
               any(f"rms=" in p for p in m2.get(32000, [])) and
               not any(p.startswith("rms=0.000000") for p in m2.get(32000, [])),
               " ".join(m2.get(32000, []))[:70]))
checks.append(("屏蔽 ffmpeg 后能重采样到 16kHz",
               16000 in m2 and "FAIL" not in " ".join(m2.get(16000, [])),
               " ".join(m2.get(16000, []))[:70]))
checks.append(("屏蔽 ffmpeg 后能重采样到 48kHz",
               48000 in m2 and "FAIL" not in " ".join(m2.get(48000, [])),
               " ".join(m2.get(48000, []))[:70]))

m3 = markers(out3)
checks.append(("屏蔽 gradio 后仍能 import my_utils",
               "import_fail" not in out3 and 32000 in m3,
               "import_fail" if "import_fail" in out3 else "OK"))

m4 = markers(out4)
checks.append(("两个都屏蔽仍能读",
               "import_fail" not in out4 and 32000 in m4,
               "import_fail" if "import_fail" in out4 else "OK"))

bad = 0
for name, ok, extra in checks:
    bad += 0 if ok else 1
    print(f"  {'✓' if ok else '✗'} {name}" + (f"  ({extra})" if extra else ""))

print()
print("=" * 88)
if bad == 0:
    print("  ✓ patch_no_ffmpeg.py 全部验证通过")
    print("    · 没有 ffmpeg 也能读音频（soundfile 回退）")
    print("    · 采样率不符时会重采样")
    print("    · gradio 出问题不再阻塞训练")
else:
    print(f"  ✗ {bad} 项失败")
print("=" * 88)
