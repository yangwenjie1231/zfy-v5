# -*- coding: utf-8 -*-
"""给 s1_train.py（GPT 训练）打 ROCm 补丁。

## 问题

`s1_train.py` 里：

    strategy=DDPStrategy(
        process_group_backend="nccl" if platform.system() != "Windows" else "gloo"
    )

Linux 上永远选 `nccl` —— 但 **ROCm 不支持 NCCL**，会报
`ncclInternalError` / `Distributed package doesn't have NCCL built in`。
（和 `s2_train_v3_lora.py` 是同一个毛病。）

## 补丁

改成检测 `torch.version.hip`：

    ("gloo" if (platform.system() == "Windows"
                or getattr(torch.version, "hip", None)) else "nccl")

幂等：已打过就跳过。改完做 py_compile 校验。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MARK = "# [zfy-rocm]"
# 只替换表达式本身，**绝不在行尾加注释** ——
# 原行是 `strategy=DDPStrategy(process_group_backend=... "gloo")`，
# 那个收尾的 `)` 属于 DDPStrategy(。若在行尾加 `# 注释`，
# 会把它一起注释掉，导致 `'(' was never closed`。
# （第一版就是这么错的，幸好有 py_compile 兜住。）
OLD_PAT = re.compile(
    r'process_group_backend\s*=\s*"nccl"\s*if\s*platform\.system\(\)\s*!=\s*"Windows"\s*else\s*"gloo"'
)
NEW = ('process_group_backend=("gloo" if (platform.system() == "Windows" '
       'or getattr(torch.version, "hip", None)) else "nccl")')
ALREADY = 'getattr(torch.version, "hip"'


def patch_file(p: Path) -> tuple[bool, str]:
    src = p.read_text(encoding="utf-8", errors="replace")
    if ALREADY in src and "process_group_backend" in src:
        return False, "已打过补丁"
    if not OLD_PAT.search(src):
        return False, "未找到目标表达式（版本可能不同）"
    new = OLD_PAT.sub(NEW, src, count=1)
    bak = p.with_suffix(p.suffix + ".bak_rocm")
    if not bak.exists():
        bak.write_text(src, encoding="utf-8")
    p.write_text(new, encoding="utf-8")
    return True, "已打补丁"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    root = Path(args.root) if args.root else None
    if root is None:
        here = Path.cwd()
        for c in (here, *here.parents):
            if (c / "GPT_SoVITS" / "s1_train.py").exists():
                root = c
                break
    if root is None:
        print("找不到检出根，用 --root 指定")
        return 1

    print("=" * 74)
    print("s1_train.py ROCm 补丁（nccl -> gloo）")
    print("=" * 74)

    ok_any = False
    for rel in ("GPT_SoVITS/s1_train.py",):
        p = root / rel
        if not p.exists():
            print(f"  {rel}: 不存在")
            continue
        changed, msg = patch_file(p)
        print(f"  {rel}: {msg}")
        if changed:
            ok_any = True
            import py_compile
            try:
                py_compile.compile(str(p), doraise=True)
                print("     语法检查通过")
            except py_compile.PyCompileError as e:
                print(f"     !! 语法错误: {e}")
                return 1

    if not ok_any:
        print("  （无需改动）")
    print()
    print("  说明：ROCm 上必须用 gloo；已改为按 torch.version.hip 自动判断。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
