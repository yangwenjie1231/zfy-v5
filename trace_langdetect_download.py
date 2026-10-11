# -*- coding: utf-8 -*-
"""抓出「谁在发起下载」——打印完整调用栈。

上一版验证：设了 model="lite" 仍触发下载。
说明发起下载的不是我们补丁里的那个 detector，而是**别处**新建的实例。

最可疑：`split_lang` 包的 LangSplitter 可能自己 new 了一个 LangDetector，
完全绕过 langsegmenter.py 里设的 _default_detector。

本脚本在下载函数里抛异常并打印调用栈，直接定位。
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

root = Path(r"D:\A\zfy-v5\GPT-SoVITS")
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "GPT_SoVITS"))
os.chdir(root)

import fast_langdetect.infer as fli  # noqa: E402

captured = []


def _boom(self, *a, **k):
    captured.append(traceback.format_stack())
    raise AssertionError("DOWNLOAD ATTEMPT")


cls = getattr(fli, "ModelLoader", None)
if cls is not None and hasattr(cls, "load_with_download"):
    cls.load_with_download = _boom
    print("已拦截 ModelLoader.load_with_download")

print()
print("=" * 76)
print("触发一次分段，抓调用栈")
print("=" * 76)
from text.LangSegmenter import LangSegmenter  # noqa: E402

try:
    r = LangSegmenter.getTexts("这是 OK 的。")
    print(f"  成功: {[(x['lang'], x['text']) for x in r]}")
except Exception as e:  # noqa: BLE001
    print(f"  失败: {type(e).__name__}: {str(e)[:80]}")

if captured:
    print()
    print("=" * 76)
    print("调用栈（只看我们自己代码的帧）")
    print("=" * 76)
    frames = captured[0]
    for line in frames:
        for l in line.rstrip().split("\n"):
            if "site-packages" in l:
                continue
            print(f"  {l.strip()}")

print()
print("=" * 76)
print("检查：谁创建了 LangDetector")
print("=" * 76)
import fast_langdetect  # noqa: E402
print(f"  fast_langdetect.infer._default_detector = "
      f"{type(fli._default_detector).__name__}")
cfg = getattr(fli._default_detector, "config", None)
if cfg is not None:
    print(f"    其 config.model = {getattr(cfg, 'model', '?')!r}")
    print(f"    其 config.cache_dir = {getattr(cfg, 'cache_dir', '?')!r}")

print()
print("  split_lang 包里的 detector：")
try:
    import split_lang
    sp = Path(split_lang.__file__).parent
    print(f"    包目录: {sp}")
    for f in sorted(sp.rglob("*.py")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        for i, l in enumerate(txt.split("\n"), 1):
            if "LangDetector" in l or "detect(" in l or "import" in l and "lang" in l.lower():
                print(f"    {f.name}:{i}  {l.strip()[:90]}")
except Exception as e:  # noqa: BLE001
    print(f"    失败: {type(e).__name__}: {e}")
