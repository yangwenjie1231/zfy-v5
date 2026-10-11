# -*- coding: utf-8 -*-
"""校验生成的 .ipynb：结构合法 + 每个 code cell 语法正确。

不能只生成就交出去 —— notebook 里的代码格如果有语法错，
用户在云上跑到那一步才发现，浪费时间。
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NB = Path(r"D:\A\zfy-v5\zfy_v5_fullft.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

print("=" * 70)
print("1. notebook 结构")
print("=" * 70)
print(f"  nbformat     : {nb.get('nbformat')}.{nb.get('nbformat_minor')}")
print(f"  cells        : {len(nb['cells'])}")
md = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
cd = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"  markdown     : {len(md)}")
print(f"  code         : {len(cd)}")

# 结构必需字段
errs = []
for i, c in enumerate(nb["cells"]):
    if "cell_type" not in c:
        errs.append(f"cell {i} 缺 cell_type")
    if "source" not in c:
        errs.append(f"cell {i} 缺 source")
    if c["cell_type"] == "code" and "outputs" not in c:
        errs.append(f"cell {i} (code) 缺 outputs")
print(f"  结构问题     : {errs if errs else '无'}")

print()
print("=" * 70)
print("2. 每个 code cell 的语法")
print("=" * 70)
bad = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    try:
        ast.parse(src)
        first = src.strip().split("\n")[0][:52]
        print(f"  OK   cell {i:2d}  {first}")
    except SyntaxError as e:
        print(f"  FAIL cell {i:2d}  行 {e.lineno}: {e.msg}")
        print(f"       代码: {(e.text or '').rstrip()[:70]}")
        bad.append((i, e))

print()
print("=" * 70)
print("3. 内容完整性")
print("=" * 70)
allmd = "\n".join("".join(c["source"]) for c in md)
allcode = "\n".join("".join(c["source"]) for c in cd)
CHECKS = [
    ("含 clone 指引", "git" in allcode.lower() and "clone" in allcode.lower()),
    # v3 策略：**不碰** transformers/numpy/hf-hub（与云上 vllm 零冲突）
    ("明确训练不需要 transformers", "不需要" in allmd and "transformers" in allmd),
    ("明确不降 transformers", "不要去降" in allmd or "不碰 transformers" in allmd),
    ("说明 vllm 冲突", "vllm" in allmd),
    ("含 peft 不装的说明", "peft" in allmd and "全量微调不用" in allmd),
    ("含 gradio/pandas 可用的说明", "gr.Warning" in allmd or "模块级" in allmd),
    ("含 PYTHONPATH 设置", "PYTHONPATH" in allcode),
    ("含冒烟测试", "smoke" in allcode),
    ("含全量训练", "tmp_full.json" in allcode),
    ("含 fp16 开关说明", "fp16_run" in allcode or "hipFFT" in allcode),
    ("含 nccl/gloo 说明", "nccl" in allmd.lower() or "gloo" in allmd.lower()),
    ("含评测指引", "eval_retrain" in allmd or "评测" in allmd),
    ("含 LFS 检查", "lfs" in allcode.lower()),
    ("含 5-wav32k 说明", "5-wav32k" in allmd),
    ("含独立进程验证", "subprocess" in allcode and "独立进程" in allmd),
    ("含重启内核提示", "重启" in allmd),
]

# 单独检查：不能把 transformers 放进 pip 安装列表
# （注意 x_transformers 是另一个包，确实需要，不能误判）
import re as _re
_installs = []
for c in cd:
    src = "".join(c["source"])
    for m in _re.finditer(r'pip\(\s*"([^"]+)"', src):
        _installs.append(m.group(1))
    for m in _re.finditer(r'pip\(([^)]*)\)', src, _re.S):
        for q in _re.finditer(r'"([^"]+)"', m.group(1)):
            _installs.append(q.group(1))
_bad = [p for p in _installs
        if p == "transformers" or p.startswith("transformers")
        or p == "huggingface-hub" or p == "huggingface_hub"]
CHECKS.append(("不安装 transformers / hf-hub", not _bad))
for name, ok in CHECKS:
    print(f"  {'OK  ' if ok else 'FAIL'} {name}")

print()
print("=" * 70)
print("结论")
print("=" * 70)
if not bad and not errs:
    print("  ✓ notebook 结构合法，所有 code cell 语法正确")
    print(f"  ✓ 文件: {NB}  ({NB.stat().st_size/1024:.1f} KB)")
else:
    print(f"  ✗ 有 {len(bad)} 个语法错误 / {len(errs)} 个结构问题")
    sys.exit(1)
