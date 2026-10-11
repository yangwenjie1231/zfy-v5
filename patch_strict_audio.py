# -*- coding: utf-8 -*-
"""给 data_utils.py 加"数据加载失败不许静默"的防护。

## 要解决的问题

`GPT_SoVITS/module/data_utils.py` 的 `__getitem__` 里有 4 处：

    except:
        traceback.print_exc()
        spec = torch.zeros(1025, 100)          # 全零
        wav  = torch.zeros(1, 100*hop_length)  # 全零
        ssl  = torch.zeros(1, 768, 100)        # 全零
        text = text[-1:]
        print("load audio or ssl error!!!!!!", audiopath)

异常被吞掉，返回全零张量。后果：

  · 训练**照常"成功"**，日志只有一行打印
  · 20 条样本全失败时，一秒钟就跑完 2 个 epoch
  · 产出的权重看起来正常（735.7 MB），实际学的是垃圾
  · 我因此误判过一次"冒烟通过"

## 本补丁

1. 加一个模块级失败计数器，每条失败打印**累计次数**
   （一眼看出是偶发还是系统性）
2. 环境变量 `ZFY_STRICT_AUDIO=1` 时**直接抛异常**，
   带完整路径与 traceback，训练立刻停

这样调试时不会再把"数据全坏"误读成"训练成功"。

## 用法

    python patch_strict_audio.py --root /mnt/workspace/zfy-v5-fullft-pkg

然后调试跑：
    ZFY_STRICT_AUDIO=1 python -s GPT_SoVITS/s2_train_v5_full.py --config tmp_smoke.json

幂等：重复运行不会重复打补丁。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MARK = "ZFY_AUDIO_GUARD"
TARGET_LINE = 'print("load audio or ssl error!!!!!!", audiopath)'

# 模块级计数器（插在文件头部 import 之后）
COUNTER = '''

# [ZFY_AUDIO_GUARD] 数据加载失败计数器（补丁加，见 patch_strict_audio.py）
_ZFY_AUDIO_FAIL = 0
'''

RAISE_BLOCK = '''if os.environ.get("ZFY_STRICT_AUDIO") == "1":
    raise RuntimeError(
        "[ZFY_AUDIO_GUARD] 第 %d 次数据加载失败，拒绝用全零张量训练: %s"
        % (_ZFY_AUDIO_FAIL, audiopath)
    )'''


def patch_file(p: Path) -> tuple[bool, int]:
    src = p.read_text(encoding="utf-8", errors="replace")
    if MARK in src:
        return False, 0

    lines = src.split("\n")
    out: list[str] = []
    n = 0
    for line in lines:
        out.append(line)
        if line.strip() == TARGET_LINE:
            indent = line[: len(line) - len(line.lstrip())]
            # 累计计数
            out.append(f"{indent}_ZFY_AUDIO_FAIL += 1")
            out.append(f"{indent}if _ZFY_AUDIO_FAIL <= 3 or _ZFY_AUDIO_FAIL % 50 == 0:")
            out.append(f'{indent}    print("[ZFY_AUDIO_GUARD] 累计失败 %d 次: %s"'
                       f" % (_ZFY_AUDIO_FAIL, audiopath))")
            # 严格模式直接抛
            for sub in RAISE_BLOCK.split("\n"):
                out.append(f"{indent}{sub}" if sub else "")
            n += 1

    if n == 0:
        return False, 0

    new = "\n".join(out)

    # 插入计数器定义：放在最后一个顶层 import 之后
    imp_lines = [i for i, l in enumerate(new.split("\n"))
                 if re.match(r"^(import |from )\S", l)]
    if imp_lines:
        nl = new.split("\n")
        pos = imp_lines[-1] + 1
        nl.insert(pos, COUNTER.rstrip("\n"))
        new = "\n".join(nl)

    # 备份
    bak = p.with_suffix(p.suffix + ".bak_zfyguard")
    if not bak.exists():
        bak.write_text(src, encoding="utf-8")

    p.write_text(new, encoding="utf-8")
    return True, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    root = Path(args.root) if args.root else None
    if root is None:
        here = Path.cwd()
        for c in (here, *here.parents):
            if (c / "GPT_SoVITS" / "module" / "data_utils.py").exists():
                root = c
                break
    if root is None:
        print("找不到检出根，用 --root 指定")
        return 1

    target = root / "GPT_SoVITS" / "module" / "data_utils.py"
    print("=" * 72)
    print("数据加载防护补丁")
    print("=" * 72)
    print(f"目标: {target}")
    if not target.exists():
        print("  文件不存在")
        return 1

    changed, n = patch_file(target)
    if changed:
        print(f"  已打补丁：{n} 处失败分支已加计数 + 严格模式")
        print(f"  备份: {target.name}.bak_zfyguard")
    else:
        print("  已是补丁状态（或未找到目标行），跳过")

    # 语法检查
    import py_compile
    try:
        py_compile.compile(str(target), doraise=True)
        print("  语法检查通过")
    except py_compile.PyCompileError as e:
        print(f"  !! 语法错误: {e}")
        return 1

    print()
    print("用法：调试时设 ZFY_STRICT_AUDIO=1，任何加载失败都会立刻中断训练")
    print("      ZFY_STRICT_AUDIO=1 python -s GPT_SoVITS/s2_train_v5_full.py "
          "--config tmp_smoke.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
