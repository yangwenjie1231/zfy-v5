# -*- coding: utf-8 -*-
"""复现并定位 `TypeError: unsupported operand type(s) for -: 'float' and 'str'`。

## 报错
    File "AR/modules/lr_schedulers.py", line 30, in __init__
        self._warmup_rate = (peak_lr - init_lr) / warmup_steps
    TypeError: unsupported operand type(s) for -: 'float' and 'str'

## 链路
    s1_train.py:86   config = load_yaml_config(args.config_file)
    AR/utils/io.py:9     config = yaml.full_load(f)      # 用 PyYAML 解析
    t2s_lightning_module.py:139  init_lr=self.config["optimizer"]["lr_init"]

## 假设
`train_gpt.py` 的 `_yaml_val()` 对 float 直接 `str(v)`。
Python 的 `str(0.00001)` = `'1e-05'`。
而 **YAML 1.1 规范里 `1e-05` 不是合法 float**（必须写成 `1.0e-05` 或 `0.00001`），
于是 PyYAML 把它解析成**字符串** `'1e-05'` -> 报错。

本脚本：
  1. 复现：用 yaml.full_load 解析 `1e-05` / `0.00001`
  2. 打印 train_gpt.py 实际生成的 YAML
  3. 给出修法
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml

print("=" * 76)
print("1. 复现：PyYAML 怎么解析这些写法")
print("=" * 76)
CASES = ["0.00001", "1e-05", "1.0e-05", "0.01", "1e-4", "1.0e-4",
         "0.0001", "2000", "true", "false"]
for s in CASES:
    doc = f"v: {s}"
    try:
        val = yaml.full_load(doc)["v"]
        print(f"  'v: {s}'".ljust(22) + f"-> {type(val).__name__:8s} {val!r}")
    except Exception as e:  # noqa: BLE001
        print(f"  'v: {s}'".ljust(22) + f"-> 解析失败 {type(e).__name__}")

print()
print("=" * 76)
print("2. Python 的 str(float) 会产出什么")
print("=" * 76)
for v in (0.00001, 0.01, 0.0001, 1e-5, 1e-4, 1e-7, 1234.0):
    print(f"  str({v!r}) = {str(v)!r}")
print("""
  ⚠️ 关键：str(0.00001) = '1e-05'，而 YAML 1.1 **不认** `1e-05` 这种写法
     （必须写成 `1.0e-05` 或 `0.00001`），PyYAML 会当成**字符串**。
""")

print()
print("=" * 76)
print("3. 定位：train_gpt.py 的 _yaml_val 对 float 的处理")
print("=" * 76)
import inspect  # noqa: E402
import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "tg", r"D:\A\zfy-v5\train_gpt.py")
tg = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(tg)
except SystemExit:
    pass
except Exception as e:  # noqa: BLE001
    print(f"  （导入 train_gpt 时 {type(e).__name__}，继续）")

if hasattr(tg, "_yaml_val"):
    print("  _yaml_val 源码:")
    for l in inspect.getsource(tg._yaml_val).split("\n"):
        print(f"    {l}")
    print()
    print("  对关键值的输出:")
    for v in (0.00001, 0.01, 0.0001, 2000, True, "abc", 0.0):
        out = tg._yaml_val(v)
        try:
            back = yaml.full_load(f"v: {out}")["v"]
            t = type(back).__name__
        except Exception:  # noqa: BLE001
            t = "解析失败"
        flag = "  <-- 类型变了！" if t != type(v).__name__ else ""
        print(f"    {v!r:12} -> '{out}'  -> 回读 {t}{flag}")

print()
print("=" * 76)
print("4. 修法")
print("=" * 76)
print("""
  `_yaml_val` 对 float 要保证输出是 **YAML 能识别的十进制形式**：

      if isinstance(v, float):
          # repr() 在 Python 3 里对 float 给出可 round-trip 的形式，
          # 但 1e-05 仍不被 YAML 1.1 接受 -> 用 f-string 展开
          s = f"{v:.10f}".rstrip("0").rstrip(".") or "0"
          return s

  更稳的做法：**直接用 yaml.safe_dump 生成**，让 PyYAML 自己决定格式：

      import yaml
      path.write_text(yaml.safe_dump(cfg, allow_unicode=True,
                                     sort_keys=False), encoding="utf-8")

  注意 `sort_keys=False` —— 否则 section 顺序会被打乱（不影响功能但难读）。

  我最初手写 YAML 是为了"避免依赖 pyyaml 的格式细节"，
  但这恰恰引入了 bug：**手写 YAML 要自己保证标量格式合法**，
  而 float 的科学计数法是经典陷阱。
""")
