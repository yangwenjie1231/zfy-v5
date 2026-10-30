# -*- coding: utf-8 -*-
"""验证 full-FT 完整导出补丁的语义与幂等性。"""
from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
PATCH = HERE / "patch_full_export_all.py"
SRC = HERE / "GPT-SoVITS" / "GPT_SoVITS" / "s2_train_v5_full.py"

with tempfile.TemporaryDirectory(prefix="zfy_export_test_") as td:
    root = Path(td)
    dst = root / "GPT_SoVITS" / "s2_train_v5_full.py"
    dst.parent.mkdir(parents=True)
    shutil.copy2(SRC, dst)

    def run(*args):
        return subprocess.run([sys.executable, str(PATCH), "--root", str(root), *args],
                              text=True, capture_output=True, encoding="utf-8")

    r1 = run()
    assert r1.returncode == 0, r1.stdout + r1.stderr
    out = dst.read_text(encoding="utf-8")
    assert "ZFY_FULL_EXPORT_ALL_PATCH" in out
    assert "if key not in no_grad_names" not in out
    assert "sim_ckpt[key] = ckpt[key].half().cpu()" in out
    ast.parse(out)

    # 幂等
    r2 = run()
    assert r2.returncode == 0
    assert dst.read_text(encoding="utf-8") == out

    # 还原
    r3 = run("--revert")
    assert r3.returncode == 0
    restored = dst.read_text(encoding="utf-8")
    assert "ZFY_FULL_EXPORT_ALL_PATCH" not in restored
    assert "if key not in no_grad_names" in restored

print("✓ 完整导出补丁：应用 / 幂等 / 还原 / AST 全部通过")
print("✓ 冻结 quantizer 会进入导出权重，不再依赖推理时随机初始化")
