# -*- coding: utf-8 -*-
"""跨平台根目录探测 —— 让 `_tools/` 下的脚本在本地 Windows 与云端 Linux 都能跑。

## 为什么需要
`verify_ckpt_atomic.py` 在云端崩了：
    FileNotFoundError: [Errno 2] No such file or directory: 'D:\\A\\zfy-v5\\GPT-SoVITS'
因为它硬编码了本地路径，而云端是 `/mnt/workspace/zfy-v5-fullft-pkg`。

**这类 bug 很隐蔽**：本地测试全过，一上云就崩。

## 用法
    from zfy_paths import find_root, find_gpt_sovits, find_tools

    ROOT = find_root()                    # 仓库根（含 GPT-SoVITS/）
    GSV  = find_gpt_sovits()              # 含 GPT_SoVITS/utils.py 的目录
    TOOLS = find_tools()                  # _tools/ 目录

    # 或者命令行覆盖
    ROOT = find_root(cli_arg=args.root)

## 探测顺序
  1. 显式传入的参数
  2. 环境变量 `ZFY_ROOT`
  3. 从当前目录**向上**找（遇到 `GPT-SoVITS/` 或 `GPT_SoVITS/` 就停）
  4. 已知的默认位置（云端 / 本地）
  5. 当前工作目录（兜底）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 已知的默认位置（按优先级）
DEFAULTS = [
    Path("/mnt/workspace/zfy-v5-fullft-pkg"),        # 云端
    Path(r"D:\A\zfy-v5\GPT-SoVITS"),                 # 本地（v5 检出）
    Path(r"D:\A\zfy-v5"),                            # 本地（v5 仓库）
    Path(r"D:\A\voiceclone\GPT-SoVITS"),             # 本地（生产 v2Pro）
]


def _looks_like_root(p: Path) -> bool:
    """root = 含 GPT-SoVITS/ 或 GPT_SoVITS/ 的目录。"""
    return ((p / "GPT-SoVITS").is_dir() or (p / "GPT_SoVITS").is_dir()
            or (p / "GPT_SoVITS" / "utils.py").exists())


def find_root(cli_arg: str | Path | None = None,
              start: Path | None = None) -> Path:
    # 1) 显式参数
    if cli_arg:
        p = Path(cli_arg).expanduser().resolve()
        if p.exists():
            return p
    # 2) 环境变量
    env = os.environ.get("ZFY_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
    # 3) 从 start（或 cwd）向上找
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if _looks_like_root(p):
            return p
    # 4) 默认位置
    for p in DEFAULTS:
        if p.exists() and _looks_like_root(p):
            return p
    # 5) 兜底
    return (start or Path.cwd()).resolve()


def find_gpt_sovits(cli_arg: str | Path | None = None) -> Path:
    """返回**含 `GPT_SoVITS/utils.py`** 的目录。

    两种仓库布局：
      A. root/GPT-SoVITS/GPT_SoVITS/utils.py   （v5 检出，嵌套）
      B. root/GPT_SoVITS/utils.py              （扁平）
    """
    root = find_root(cli_arg)
    for cand in (root / "GPT-SoVITS", root):
        if (cand / "GPT_SoVITS" / "utils.py").exists():
            return cand.resolve()
        if (cand / "utils.py").exists() and cand.name == "GPT_SoVITS":
            return cand.parent.resolve()
    return root.resolve()


def find_tools() -> Path:
    """返回 `_tools/` 目录（可能不存在）。"""
    root = find_root()
    for cand in (root / "_tools", root / "GPT-SoVITS" / "_tools",
                 Path(__file__).resolve().parent):
        if cand.is_dir():
            return cand.resolve()
    return Path(__file__).resolve().parent


def add_to_path(root: Path | None = None) -> Path:
    """把 root 与 root/GPT_SoVITS 加到 sys.path 前部，返回 root。

    很多脚本都要这几行：
        sys.path[:0] = [ROOT, os.path.join(ROOT, "GPT_SoVITS")]
    统一到这里，避免每处都写错。
    """
    r = (root or find_root()).resolve()
    for p in (str(r / "GPT_SoVITS"), str(r)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return r


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"  cwd        : {Path.cwd()}")
    print(f"  find_root  : {find_root()}")
    print(f"  gpt_sovits : {find_gpt_sovits()}")
    print(f"  tools      : {find_tools()}")
    r = find_gpt_sovits()
    ok = (r / "GPT_SoVITS" / "utils.py").exists()
    print(f"  utils.py   : {'✓ 存在' if ok else '✗ 找不到'}")
