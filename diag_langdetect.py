# -*- coding: utf-8 -*-
"""查 fast_langdetect 的缓存目录要求，以及 pip 包里是否自带 lite 模型。

## 报错
    FileNotFoundError: fast-langdetect: Cache directory not found:
    .../GPT_SoVITS/pretrained_models/fast_langdetect

## 背景（skill 里记过）
langsegmenter.py:11 把 cache_dir 硬指向
`GPT_SoVITS/pretrained_models/fast_langdetect`，而**该目录不会自建**。
缺目录时它会去下载 125 MB 的 lid.176.bin（fbaipublicfiles.com，
国内常被墙/极慢）。

而 fast_langdetect 的 **lite 模型（约 916 KB）随 pip 包分发**，
实测与 full 分段**逐条一致** -> 用 lite 即可。

本脚本：
  1. 找 fast_langdetect 包位置与自带的模型文件
  2. 打印 LangDetectConfig 的字段（确认 cache_dir / model 参数名）
  3. 给出"用 lite 填满缓存目录"的具体做法
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 76)
print("1. fast_langdetect 包位置与自带文件")
print("=" * 76)
try:
    import fast_langdetect
    pkg = Path(fast_langdetect.__file__).parent
    print(f"  包目录: {pkg}")
    print(f"  版本  : {getattr(fast_langdetect, '__version__', '?')}")
    print()
    print("  包内文件（含模型）：")
    for f in sorted(pkg.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            sz = f.stat().st_size
            tag = "  <-- 模型" if f.suffix in (".bin", ".ftz", ".onnx") else ""
            print(f"    {str(f.relative_to(pkg)):46s} {sz/1024:9.1f} KB{tag}")
except Exception as e:  # noqa: BLE001
    print(f"  导入失败: {type(e).__name__}: {e}")

print()
print("=" * 76)
print("2. LangDetectConfig 的字段（确认参数名）")
print("=" * 76)
try:
    from fast_langdetect.infer import LangDetectConfig, LangDetector
    sig = inspect.signature(LangDetectConfig)
    for name, prm in sig.parameters.items():
        print(f"  {name:20s} default={prm.default!r}")
    print()
    print("  LangDetector.__init__:")
    print("   ", inspect.signature(LangDetector.__init__))
    print()
    print("  LangDetector.__call__:")
    print("   ", inspect.signature(LangDetector.__call__))
except Exception as e:  # noqa: BLE001
    print(f"  失败: {type(e).__name__}: {e}")

print()
print("=" * 76)
print("3. 仓库里 langsegmenter.py 的前 20 行")
print("=" * 76)
lg = Path(r"D:\A\zfy-v5\GPT-SoVITS\GPT_SoVITS\text\LangSegmenter"
          r"\langsegmenter.py")
if lg.exists():
    for i, line in enumerate(lg.read_text(encoding="utf-8",
                                          errors="replace").split("\n")[:20], 1):
        print(f"  L{i:3d}: {line}")
else:
    print("  找不到")

print()
print("=" * 76)
print("4. 结论与修法")
print("=" * 76)
print("""
  修法（三选一，按优先级）：

  A. 用 pip 包自带的 lite 模型填满缓存目录（推荐，离线可用）
       mkdir -p GPT_SoVITS/pretrained_models/fast_langdetect
       cp <包目录>/lid.176.lite.bin \\
          GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin

  B. 联网下载 full 模型（125 MB，国内可能很慢或被墙）

  C. 给 langsegmenter.py 打补丁，直接用 lite 路径

  注意：cache_dir 只是"放模型的目录"，文件名必须叫 lid.176.bin
        （fast_langdetect 内部按这个固定名字找）。
""")
