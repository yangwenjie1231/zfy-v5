# -*- coding: utf-8 -*-
"""验证 G2PW 降级补丁：缺 onnxruntime 时不再中断合成。

## 验证方式
1. 打补丁后确认 chinese2.py 含 try/except 与 ZFY_G2PW 开关
2. **模拟 onnxruntime 缺失**（把 import 钩子改成抛 ImportError），
   确认 chinese2 仍能 import 成功且 is_g2pw 降级为 False
3. 真实合成一小段文本，确认能出音素（不崩）
"""
from __future__ import annotations

import builtins
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

root = Path(r"D:\A\zfy-v5\GPT-SoVITS")
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "GPT_SoVITS"))
os.chdir(root)

print("=" * 76)
print("1. 补丁是否已应用")
print("=" * 76)
p = root / "GPT_SoVITS" / "text" / "chinese2.py"
src = p.read_text(encoding="utf-8", errors="replace")
print(f"  含 ZFY_V5FULLFT_PATCH : {'ZFY_V5FULLFT_PATCH' in src}")
print(f"  含 ZFY_G2PW 开关      : {'ZFY_G2PW' in src}")
print(f"  含 try/except 降级    : {'_zfy_g2pw_err' in src}")

print()
print("=" * 76)
print("2. 模拟 onnxruntime 缺失，确认 chinese2 仍可 import")
print("=" * 76)
_real_import = builtins.__import__


def _fake_import(name, *a, **k):
    if name == "onnxruntime" or name.startswith("onnxruntime."):
        raise ImportError("simulated: no onnxruntime")
    return _real_import(name, *a, **k)


builtins.__import__ = _fake_import
# 清掉可能已缓存的模块
for m in list(sys.modules):
    if m.startswith(("text.chinese2", "text.g2pw", "onnxruntime")):
        del sys.modules[m]

try:
    import text.chinese2 as c2
    print(f"  ✓ chinese2 导入成功")
    print(f"    is_g2pw = {c2.is_g2pw}  （期望 False = 已降级）")
except Exception as e:  # noqa: BLE001
    print(f"  ✗ chinese2 导入失败: {type(e).__name__}: {str(e)[:130]}")
    c2 = None
finally:
    builtins.__import__ = _real_import

print()
print("=" * 76)
print("3. 降级后还能不能正常出音素")
print("=" * 76)
if c2 is not None:
    from text import cleaned_text_to_sequence
    from text.cleaner import clean_text
    for t in ["今天天气不错。", "再好的武器，也得朝夕相处磨合上几天。"]:
        try:
            phones, word2ph, norm = clean_text(t, "zh")
            ids = cleaned_text_to_sequence(phones)
            print(f"  {t}")
            print(f"    音素 {len(phones)} 个, id {len(ids)} 个 -> {ids[:12]}…")
        except Exception as e:  # noqa: BLE001
            print(f"  {t}")
            print(f"    **失败** {type(e).__name__}: {str(e)[:130]}")
else:
    print("  跳过（上一步失败）")

print()
print("=" * 76)
print("4. 结论")
print("=" * 76)
print("""
  降级后 G2PW 关闭 -> 走 jieba 词性 + 单音字表。
  代价：多音字（长/重/行/了）可能读错。
  收益：不需要 onnxruntime，也不用每次加载 635MB 的 g2pW.onnx。

  若想开启：pip install onnxruntime
  （g2pW.onnx 已在仓库里，装完自动启用）
""")
