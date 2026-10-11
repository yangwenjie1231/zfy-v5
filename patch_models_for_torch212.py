# -*- coding: utf-8 -*-
"""给 `module/models.py` 打补丁：`torch.cuda.amp` 兼容 torch 2.12+。

## 问题

`GPT_SoVITS/module/models.py:23` 是**无保护的硬 import**：

    from torch.cuda.amp import autocast

`torch.cuda.amp` 自 torch 2.4 起弃用，官方计划移除。
目标环境是 **torch 2.12.0**（跨了 5 个大版本），若已移除则：

  · `models.py` 导入失败 → **训练和推理都崩**
  · 而且它 import 在模块顶层，任何 `from module.models import ...` 都会触发

我的 `s2_train_v5_full.py` 已加 try/except 回退，但 `models.py` 没有。

## 补丁内容

把硬 import 改成带回退的形式：

    try:
        from torch.amp import autocast          # 新 API（需 device_type）
    except ImportError:
        try:
            from torch.cuda.amp import autocast
        except ImportError:
            autocast = None

并把三处 `with autocast(enabled=False):` 改成兼容调用 ——
新 API 要求 `device_type` 参数：

    with autocast("cuda", enabled=False):       # 新版
    with autocast(enabled=False):               # 旧版

做法：定义一个统一包装函数 `_autocast_disabled()` 返回合适的上下文管理器，
把三处 `with autocast(enabled=False):` 替换成它。

## 安全性

  · 原地修改，改前自动备份为 `models.py.bak_torch212`
  · 幂等：已打过补丁则跳过
  · 改后 `py_compile` 校验

用法：
    python patch_models_for_torch212.py            # 打补丁
    python patch_models_for_torch212.py --revert   # 还原
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
TARGETS = [
    GSV / "GPT_SoVITS" / "module" / "models.py",
    GSV / "GPT_SoVITS" / "module" / "models_onnx.py",
]
BAK_SUFFIX = ".bak_torch212"
MARK = "# [torch212-patch]"

OLD_IMPORT = "from torch.cuda.amp import autocast"

NEW_IMPORT = f'''{MARK} torch 2.12+ 兼容：优先用 torch.amp，回退 torch.cuda.amp
try:
    from torch.amp import autocast as _autocast_new

    def autocast(enabled=True, **kw):
        """兼容包装：新 API 需要 device_type 位置参数。"""
        return _autocast_new(device_type="cuda", enabled=enabled, **kw)
except ImportError:
    try:
        from torch.cuda.amp import autocast
    except ImportError:  # 两者都没有（极旧/极新 torch）
        import contextlib

        def autocast(enabled=True, **kw):
            return contextlib.nullcontext()'''


def patch_file(p: Path) -> bool:
    if not p.exists():
        print(f"  [跳过] 不存在: {p}")
        return False
    src = p.read_text(encoding="utf-8")
    if MARK in src:
        print(f"  [跳过] 已打过补丁: {p.name}")
        return False
    if OLD_IMPORT not in src:
        print(f"  [跳过] 未找到目标 import: {p.name}")
        return False

    bak = p.with_suffix(p.suffix + BAK_SUFFIX)
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"  备份 -> {bak.name}")

    new = src.replace(OLD_IMPORT, NEW_IMPORT, 1)
    p.write_text(new, encoding="utf-8")
    print(f"  已打补丁: {p.name}")
    return True


def revert_file(p: Path) -> bool:
    bak = p.with_suffix(p.suffix + BAK_SUFFIX)
    if not bak.exists():
        print(f"  [跳过] 无备份: {p.name}")
        return False
    shutil.copy2(bak, p)
    print(f"  已还原: {p.name}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--root", default=str(GSV),
                    help="GPT-SoVITS 根目录（云上路径不同时用）")
    args = ap.parse_args()

    global TARGETS
    root = Path(args.root)
    if root != GSV:
        TARGETS = [root / "GPT_SoVITS" / "module" / "models.py",
                   root / "GPT_SoVITS" / "module" / "models_onnx.py"]

    print("=" * 74)
    print("还原" if args.revert else "打补丁: torch.cuda.amp -> torch.amp")
    print("=" * 74)
    n = 0
    for p in TARGETS:
        ok = revert_file(p) if args.revert else patch_file(p)
        n += bool(ok)

    # 语法校验
    if not args.revert:
        import py_compile
        print()
        for p in TARGETS:
            if not p.exists():
                continue
            try:
                py_compile.compile(str(p), doraise=True)
                print(f"  OK   {p.name} 语法正确")
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {p.name}: {e}")
                return 1
    print(f"\n完成，改动 {n} 个文件")
    if not args.revert:
        print("""
  注意：本机（torch 2.7.1）也能用这个补丁 —— torch.amp 从 2.4 起就有。
  补丁是幂等的，重复运行会跳过。
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
