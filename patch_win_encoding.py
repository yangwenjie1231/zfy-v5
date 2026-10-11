# -*- coding: utf-8 -*-
"""修 Windows 下的两个编码/换行问题（GPT 训练必需）。

## 问题 1：`open()` 没指定 encoding（**主因**）
    # GPT_SoVITS/AR/utils/io.py:7-10
    def load_yaml_config(path):
        with open(path) as f:          # <- Windows 默认 GBK
            config = yaml.full_load(f)

而 `train_gpt.py` 生成的 YAML 第一行是**中文注释**（UTF-8）：
    # 由 train_gpt.py 生成
-> `UnicodeDecodeError: 'gbk' codec can't decode byte 0xb1`

**这是技能里记录过的同类坑**（`english.py` 读 cmudict.rep 也是这问题）。

同类隐患：`save_config_to_yaml` 的 `open(path, "w")` 也是默认编码。

## 问题 2：生成的 YAML 带 CRLF
实测前 40 字节：
    23 20 e7 94 b1 20 ... 0d 0a
    #     由  ...            \r\n
`train_gpt.py` 用 `Path.write_text()`，Windows 下默认 `newline=None` ->
`\n` 被转成 `\r\n`。虽然 PyYAML 能容忍 CRLF，
但**与 Linux 上生成的不一致**，容易在比对时误判。

## 修法
  1. 给 `io.py` 的 `open()` 补 `encoding="utf-8"`
     （`load_yaml_config` + `save_config_to_yaml` 两处）
  2. 给 `train_gpt.py` 的 `write_yaml` 补 `newline="\n"`

## 用法
    python _tools/patch_win_encoding.py --root .
    python _tools/patch_win_encoding.py --root . --check
    python _tools/patch_win_encoding.py --root . --revert
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MARK = "ZFY_WIN_ENCODING_PATCH"

# ---------------------------------------------------------------- 补丁 1
IO_REL = "GPT-SoVITS/GPT_SoVITS/AR/utils/io.py"
IO_ALT = "GPT_SoVITS/AR/utils/io.py"

IO_OLD = """def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config


def save_config_to_yaml(config, path):
    assert path.endswith(".yaml")
    with open(path, "w") as f:
        f.write(yaml.dump(config))
"""

IO_NEW = '''def load_yaml_config(path):
    # ---- ZFY_WIN_ENCODING_PATCH ----
    # 原实现 `open(path)` 在 Windows 下默认 **GBK**，而 YAML 里含
    # UTF-8 中文注释（train_gpt.py 生成的第一行 `# 由 ... 生成`）->
    #     UnicodeDecodeError: 'gbk' codec can't decode byte 0xb1
    # 这是技能里记录过的同类坑（english.py 读 cmudict.rep 也是这问题）。
    with open(path, encoding="utf-8") as f:
        config = yaml.full_load(f)
    return config


def save_config_to_yaml(config, path):
    assert path.endswith(".yaml")
    # 同上：写入也要显式 utf-8 + \n
    with open(path, "w", encoding="utf-8", newline="\\n") as f:
        f.write(yaml.dump(config, allow_unicode=True))
'''

# ---------------------------------------------------------------- 补丁 2
TG_REL = "train_gpt.py"

TG_OLD_1 = '''    path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")'''

TG_NEW_1 = '''    # ---- ZFY_WIN_ENCODING_PATCH ----
    # ⚠️ `newline="\\n"`：Windows 下 `Path.write_text` 的 `newline=None`
    #    会把 `\\n` 转成 `\\r\\n`（实测生成文件头是 `... 0d 0a`）。
    #    虽然 PyYAML 容忍 CRLF，但与 Linux 上生成的不一致，
    #    跨平台比对 YAML 时会误判"文件不同"。
    path.write_text("\\n".join(lines) + "\\n", encoding="utf-8",
                    newline="\\n")'''


def patch_one(root: Path, rels: list[str], old: str, new: str,
              check: bool, revert: bool, tag: str) -> int:
    p = None
    for r in rels:
        if (root / r).exists():
            p = root / r
            break
    if p is None:
        print(f"  !! 找不到 {rels}")
        return 1
    src = p.read_text(encoding="utf-8", errors="replace")
    bak = p.with_suffix(p.suffix + ".bak_zfy_enc")

    if revert:
        if bak.exists():
            shutil.copy2(bak, p)
            print(f"  已还原 {p.name}")
            return 0
        print(f"  未打过补丁 {p.name}")
        return 0

    if MARK in src:
        print(f"  已打过补丁: {p.name}")
        return 0

    if old not in src:
        print(f"  !! {p.name} 找不到目标片段（{tag}）")
        return 1

    src = src.replace(old, new, 1)
    if check:
        print(f"  [{tag}] 将修改 {p.name}")
        return 0

    import ast
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"  !! {p.name} 打完语法错误（已放弃）: 第 {e.lineno} 行 {e.msg}")
        return 1

    if not bak.exists():
        shutil.copy2(p, bak)
    p.write_text(src, encoding="utf-8", newline="\n")
    print(f"  [{tag}] {p.name} 已应用")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    print("=" * 76)
    print(f"修 Windows 编码/换行: {root}")
    print("=" * 76)

    rc = 0
    rc |= patch_one(root, [IO_REL, IO_ALT], IO_OLD, IO_NEW,
                    args.check, args.revert, "1/2 io.py")
    print()
    rc |= patch_one(root, [TG_REL], TG_OLD_1, TG_NEW_1,
                    args.check, args.revert, "2/2 train_gpt.py")

    print()
    print("=" * 76)
    if rc == 0 and not args.check and not args.revert:
        print("""
  补丁完成（2 处）。

  ## 修了什么
    1. `AR/utils/io.py` 的 `open()` 补 `encoding="utf-8"`
       -> 解决 `UnicodeDecodeError: 'gbk' codec can't decode byte 0xb1`
          （YAML 第一行含 UTF-8 中文注释，Windows 默认 GBK 读不了）
    2. `train_gpt.py` 的 `write_yaml` 补 `newline="\\n"`
       -> 避免 CRLF（实测文件头 `... 0d 0a`）

  ## 然后重跑
      python train_gpt.py --base both

  ## 如果还报编码错
  可能是 `s1_train.py` 或 `AR/data/*.py` 也有裸 `open()`。查：
      grep -rn 'open(' GPT-SoVITS/GPT_SoVITS/AR/ --include=*.py | grep -v encoding
""")
    elif rc != 0:
        print("  有改动未应用，见上面的 !! 行")
    print("=" * 76)
    return rc


if __name__ == "__main__":
    sys.exit(main())
