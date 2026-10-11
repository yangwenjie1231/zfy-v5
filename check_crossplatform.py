# -*- coding: utf-8 -*-
"""检查脚本里所有**平台相关的路径拼接**（Windows 反斜杠 / 硬编码盘符）。

## 为什么需要
同一个 bug 我犯了两次：
  1. `verify_ckpt_atomic.py` 硬编码 `D:\\A\\zfy-v5\\GPT-SoVITS`
     -> 云端 FileNotFoundError
  2. 修了硬编码后，探针里又写 `GSV + r"\\GPT_SoVITS"`
     -> Linux 上反斜杠是普通字符 -> 拼出无效路径 -> 探针**静默失败**
        （只看到 8 条断言全是 None，没有任何错误信息）

**第 2 类更危险**：脚本"跑完了"，只是结果全错。

## 本脚本扫描
  1. 硬编码盘符（`D:\\`、`C:\\`）
  2. `+ r"\\..."` / `+ "\\..."` 形式的路径拼接（应用 os.path.join）
  3. `.split("\\")` 之类的分隔符硬编码
  4. 字符串里的 `\\` 路径分隔符
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent

# 允许的例外（明确的本地专用脚本）
ALLOW_HARDCODED = {
    "check_crossplatform.py",   # 自身含示例文本

    "make_v2pro_baseline.py",       # 本地合成用的驱动脚本
    "make_listen_v5_vs_v2pro.py",   # 本地盲听包
    "rebuild_listen_fixed.py",      # 本地盲听包
    "eval_from_wav.py",             # 本地评估（读本地 dist/）
    "compare_resample.py",
    "diag_breathiness.py",
    "verify_tail_rms.py",
    "find_heldout.py", "find_heldout_v2.py", "find_heldout_exact.py",
    "audit_heldout.py", "debug_heldout.py", "pick_heldout_clean.py",
    "eval_objective.py", "zfy_paths.py",   # zfy_paths 里 DEFAULTS 就是本地路径
    "explain_gpt_loss.py",
    "diag_gpt_training.py",
}

DRIVE = re.compile(r'["\']([A-Za-z]):[\\/]')
BACKSLASH_JOIN = re.compile(r'\+\s*r?["\']\\\\')
BS_IN_STRING = re.compile(r'["\'][^"\']*\\\\[A-Za-z_][^"\']*["\']')

print("=" * 90)
print("扫描平台相关的路径拼接")
print("=" * 90)

issues: list[tuple[str, int, str, str]] = []
py_files = sorted(ROOT.glob("*.py")) + sorted((ROOT / "_tools").glob("*.py")) \
    if (ROOT / "_tools").is_dir() else sorted(ROOT.glob("*.py"))

for p in py_files:
    if p.name in ALLOW_HARDCODED:
        continue
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").split("\n")
    except OSError:
        continue
    for i, l in enumerate(lines, 1):
        if l.strip().startswith("#"):
            continue
        # 硬编码盘符
        for m in DRIVE.finditer(l):
            issues.append((p.name, i, "硬编码盘符", l.strip()[:88]))
            break
        # + r"\" 路径拼接
        if BACKSLASH_JOIN.search(l):
            issues.append((p.name, i, "反斜杠拼接（应用 os.path.join）",
                           l.strip()[:88]))
        # 字符串里的反斜杠路径
        elif BS_IN_STRING.search(l) and "re." not in l and "regex" not in l.lower():
            issues.append((p.name, i, "字符串含反斜杠路径", l.strip()[:88]))

if issues:
    print()
    by_kind: dict[str, list] = {}
    for name, ln, kind, text in issues:
        by_kind.setdefault(kind, []).append((name, ln, text))
    for kind, items in by_kind.items():
        print(f"\n  【{kind}】 {len(items)} 处")
        for name, ln, text in items[:14]:
            print(f"    {name}:{ln}")
            print(f"      {text}")
        if len(items) > 14:
            print(f"    …还有 {len(items)-14} 处")
else:
    print("\n  ✓ 未发现平台相关的路径拼接问题")

print()
print("=" * 90)
print("结论")
print("=" * 90)
print(f"""
  扫描了 {len(py_files)} 个脚本（跳过 {len(ALLOW_HARDCODED)} 个本地专用脚本）
  发现 {len(issues)} 处问题

  ## 两类 bug 的区别
    ① 硬编码盘符        -> 云端**立刻报错**（FileNotFoundError），容易发现
    ② 反斜杠路径拼接    -> 云端**静默失败**（路径无效但不抛异常），
                          只看到结果全是 None / 空 —— **更危险**

  ②的典型表现就是 `verify_ckpt_atomic.py` 在云端那 8 条断言全 None。

  ## 正确写法
    ✗ sys.path[:0] = [GSV, GSV + r"\\GPT_SoVITS"]
    ✗ os.chdir(GSV + "\\\\GPT_SoVITS")
    ✓ import os; _GS = os.path.join(GSV, "GPT_SoVITS")
    ✓ from zfy_paths import find_gpt_sovits; GSV = find_gpt_sovits()
""")
print("=" * 90)
