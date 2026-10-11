# -*- coding: utf-8 -*-
"""诊断并修复 numpy ABI 冲突。

## 症状

    ValueError: numpy.dtype size changed, may indicate binary incompatibility.
                Expected 96 from C header, got 88 from PyObject

## 判读（方向很关键）

  "Expected 96 from C header" = **扩展是按 numpy 2.x 编译的**（dtype 结构 96 字节）
  "got 88 from PyObject"       = **运行时 numpy 是 1.x**（88 字节）

所以是 **numpy 被降级了**，导致 scipy/pandas 等按 numpy2 编译的 C 扩展
ABI 不匹配。

## 为什么会降级

我的安装清单里有包声明了 `numpy<2`（很可能是 `torchmetrics<=1.5`
拉来的某个旧依赖，或 `librosa==0.10.2` 的 numba 链）。
pip 为了满足它把 numpy 从 2.3.5 降到 1.x。

## 本脚本

  1. 报告当前 numpy 版本
  2. **找出**所有声明 numpy<2 的已装包（定位真凶）
  3. 检查 llvmlite/numba 等 ABI 敏感包
  4. 把 numpy 恢复到 2.x 并验证

**只恢复 numpy，不碰 torch**（ROCm 版 torch 重装代价很大）。

用法（云上）：
    python fix_numpy_abi.py            # 诊断 + 修复
    python fix_numpy_abi.py --dry-run  # 只看诊断
"""
from __future__ import annotations

import argparse
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def hdr(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


hdr("1. 当前 numpy")
try:
    import numpy as np
    ver = np.__version__
    print(f"  numpy: {ver}")
    print(f"  dtype 结构大小推断: "
          f"{'1.x (88字节)' if ver.startswith('1.') else '2.x (96字节)'}")
except Exception as e:  # noqa: BLE001
    print(f"  numpy 导入失败: {type(e).__name__}: {e}")
    ver = None

hdr("2. 找出声明 numpy<2 的包（真凶）")
import importlib.metadata as md  # noqa: E402

culprits = []
for dist in md.distributions():
    name = dist.metadata["Name"] or "?"
    try:
        reqs = dist.requires or []
    except Exception:  # noqa: BLE001
        continue
    for r in reqs:
        rl = r.lower().replace(" ", "")
        if not rl.startswith("numpy"):
            continue
        # 只关心上界 <2 的
        if "<2" in rl or "< 2" in rl:
            culprits.append((name, dist.version, r))
if culprits:
    print(f"  发现 {len(culprits)} 个包声明 numpy<2：")
    for n, v, r in culprits:
        print(f"    {n}=={v}")
        print(f"        requires: {r}")
else:
    print("  未发现显式 numpy<2 声明")
    print("  （可能是 pip 解析传递依赖时的隐式降级）")

hdr("3. ABI 敏感包版本")
for m in ("llvmlite", "numba", "scipy", "pandas", "scikit-learn", "soundfile"):
    try:
        mod = __import__(m)
        print(f"  {m:16s} {getattr(mod, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"  {m:16s} 导入失败 {type(e).__name__}: {str(e)[:60]}")

hdr("4. 判断与修复")
need_fix = bool(ver and ver.startswith("1."))
if not need_fix:
    print("  numpy 已是 2.x —— 那不是降级问题。")
    print("  另一种可能：某个包是**按 numpy 1.x 编译**后装的（Expected 88）。")
    print("  请把完整 traceback 贴出来，我据此定位具体是哪个 import 失败。")
else:
    print("  numpy 是 1.x -> 需要恢复到 2.x")
    print()
    print("  将执行（不碰 torch）：")
    print('    pip install "numpy>=2,<2.4"')
    print()
    print("  说明：numpy 2.x 与本机实测一致（本机 2.1.3 正常）；")
    print("        <2.4 上界是为了兼容 numba/llvmlite 的已知要求。")

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--upper", default="2.4",
                help="numpy 上界（默认 2.4，兼容 numba）")
args, _ = ap.parse_known_args()

if need_fix and not args.dry_run:
    hdr("5. 恢复 numpy 到 2.x")
    cmd = [sys.executable, "-m", "pip", "install", "--quiet",
           f"numpy>=2,<{args.upper}"]
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print("  !! 失败：")
        print(r.stderr[-2000:])
    else:
        print("  安装完成")

    hdr("6. 验证")
    # 必须在新进程里验证（当前进程已加载旧 numpy）
    code = (
        "import numpy, scipy, soundfile\n"
        "print('numpy  ', numpy.__version__)\n"
        "print('scipy  ', scipy.__version__)\n"
        "print('soundfile', soundfile.__version__)\n"
        "import librosa\n"
        "print('librosa', librosa.__version__)\n"
        "print('OK: C 扩展 ABI 一致')\n"
    )
    r2 = subprocess.run([sys.executable, "-c", code],
                        capture_output=True, text=True)
    print(r2.stdout)
    if r2.returncode != 0:
        print("  !! 仍有问题：")
        print(r2.stderr[-2500:])
    else:
        print("  ✓ numpy ABI 已恢复")
        print()
        print("  注意：**重启 Notebook 内核**后再跑后续 cell，")
        print("        当前进程里已加载的旧 numpy 不会自动更新。")
elif need_fix and args.dry_run:
    print("\n(dry-run，未执行修复)")
else:
    hdr("5. 无需修复")
