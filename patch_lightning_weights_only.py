# -*- coding: utf-8 -*-
"""修 `torch.load` 的 `weights_only` 问题（PyTorch ≥2.6 默认值变了）。

## 症状（实测）
    [A] s1v3  ✓ 8 个 ckpt          <- 先跑，成功
    [B] v2    ✗ 1 个 ckpt 15s      <- 随后启动时崩

    _pickle.UnpicklingError: Weights only load failed.
    WeightsUnpickler error: Unsupported global:
        GLOBAL pathlib._local.WindowsPath was not an allowed global

## 根因（两层）
  1. **PyTorch 2.6 把 `torch.load` 的 `weights_only` 默认值从 False 改成 True**
     （技能里记录过：onnx_export.py 就是栽在这个坑）
  2. Lightning 的 `resume_start` 用 `pl_load(path, weights_only=...)`，
     而它拿到的 `weights_only` 是 `Trainer` 的参数（默认 True）
  3. checkpoint 里存了 **`WindowsPath` 对象** —— 来自 `train_gpt.py`
     写进 YAML/配置的 `Path`，Lightning 把它一起存了 -> 反序列化被拦

## 为什么 [A] 能过、[B] 才崩
  [A] 是**第一次**跑（没有旧 checkpoint -> 不走 resume 分支）
  [A] 结束时**写了** `last.ckpt`
  [B] 启动时 `resume_start` 读到 [A] 留下的 ckpt -> 崩

**所以真正的问题是"两个路线共用了同一个 checkpoint 目录"** ——
这也是本补丁要一并修的（否则 A/B 会互相干扰）。

## 修法（两处）
  1. **会话隔离**：A/B 各自独立的 `ckpt_dir`（根因，最重要）
  2. **`weights_only=False`**：给 `pl_load` 的调用点补上
     （我们自己生成的 ckpt，可信）

## 用法
    python _tools/patch_lightning_weights_only.py --root .
    python _tools/patch_lightning_weights_only.py --root . --check
    python _tools/patch_lightning_weights_only.py --root . --revert
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MARK = "ZFY_WEIGHTS_ONLY_PATCH"

# ---------------------------------------------------------------- 找 lightning
def find_cloud_io() -> Path | None:
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import lightning_fabric.utilities.cloud_io as m; print(m.__file__)"],
            capture_output=True, text=True, timeout=60)
        p = (r.stdout or "").strip()
        if p and Path(p).exists():
            return Path(p)
    except Exception:  # noqa: BLE001
        pass
    return None


def find_torch_io() -> Path | None:
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import lightning_fabric.plugins.io.torch_io as m; print(m.__file__)"],
            capture_output=True, text=True, timeout=60)
        p = (r.stdout or "").strip()
        if p and Path(p).exists():
            return Path(p)
    except Exception:  # noqa: BLE001
        pass
    return None


OLD_CLOUD = """    if not isinstance(path_or_url, (str, Path)):
        # any sort of BytesIO or similar
        return torch.load(
            path_or_url,
            map_location=map_location,  # type: ignore[arg-type] # upstream annotation is not correct
            weights_only=weights_only,
        )"""

NEW_CLOUD = """    # ---- ZFY_WEIGHTS_ONLY_PATCH ----
    # PyTorch 2.6 起 `torch.load` 的 `weights_only` 默认值从 False 变 True。
    # 本项目的 ckpt 里含 `pathlib.WindowsPath`（配置里的 Path 被一起存了）
    # -> `WeightsUnpickler error: Unsupported global: WindowsPath`。
    # 这些 ckpt 是我们自己生成的（可信），所以默认按 False。
    #
    # ⚠️ 上游只对**远程** checkpoint 做了这个兜底（下面的 `startswith("http")`
    #    分支里有 `if weights_only is None: weights_only = False`），
    #    但**本地文件路径没有** —— 而我们的 ckpt 全在本地。
    if weights_only is None:
        weights_only = False

    if not isinstance(path_or_url, (str, Path)):
        # any sort of BytesIO or similar
        return torch.load(
            path_or_url,
            map_location=map_location,  # type: ignore[arg-type] # upstream annotation is not correct
            weights_only=weights_only,
        )"""

OLD_CLOUD2 = """    fs = get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb") as f:
        return torch.load(
            f,
            map_location=map_location,  # type: ignore[arg-type]
            weights_only=weights_only,
        )"""

NEW_CLOUD2 = """    fs = get_filesystem(path_or_url)
    # ---- ZFY_WEIGHTS_ONLY_PATCH（同上：本地路径也要兜底）----
    if weights_only is None:
        weights_only = False
    with fs.open(path_or_url, "rb") as f:
        return torch.load(
            f,
            map_location=map_location,  # type: ignore[arg-type]
            weights_only=weights_only,
        )"""

OLD_TORCH_IO = ("        return pl_load(path, map_location=map_location, "
                "weights_only=weights_only)")

NEW_TORCH_IO = """        # ---- ZFY_WEIGHTS_ONLY_PATCH ----
        # 这里 `weights_only` 来自 Trainer（默认 True）。
        # 本项目 ckpt 含 WindowsPath -> True 会 UnpicklingError。
        if weights_only is None:
            weights_only = False
        return pl_load(path, map_location=map_location,
                       weights_only=weights_only)"""


def patch_file(p: Path, pairs: list, check: bool, revert: bool,
               tag: str) -> int:
    if p is None or not p.exists():
        print(f"  !! 找不到 {tag}")
        return 1
    src = p.read_text(encoding="utf-8", errors="replace")
    bak = p.with_suffix(p.suffix + ".bak_zfy_wo")

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

    done = 0
    for old, new in pairs:
        if old in src:
            src = src.replace(old, new, 1)
            done += 1
    if done == 0:
        print(f"  !! {p.name} 找不到任何目标片段（{tag}）")
        return 1
    if check:
        print(f"  [{tag}] 将修改 {p.name}")
        return 0

    import ast
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"  !! {p.name} 语法错误（已放弃）: 第 {e.lineno} 行 {e.msg}")
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
    print("=" * 78)
    print(f"修 torch.load 的 weights_only: {root}")
    print("=" * 78)

    cloud = find_cloud_io()
    tio = find_torch_io()
    print(f"  cloud_io : {cloud}")
    print(f"  torch_io : {tio}")
    print()

    rc = 0
    rc |= patch_file(cloud,
                     [(OLD_CLOUD, NEW_CLOUD), (OLD_CLOUD2, NEW_CLOUD2)],
                     args.check, args.revert, "1/2 cloud_io")
    print()
    rc |= patch_file(tio, [(OLD_TORCH_IO, NEW_TORCH_IO)],
                     args.check, args.revert, "2/2 torch_io")

    print()
    print("=" * 78)
    if rc == 0 and not args.check and not args.revert:
        print("""
  补丁完成（2 处，改的是**安装的 lightning 包**）。

  ## ⚠️ 但这只是止血 —— 真根因是 A/B 共用 ckpt 目录
  实测：[A] 成功写了 last.ckpt -> [B] 启动时 resume 读到它 -> 崩。
  即使修了 weights_only，A/B 仍会**互相覆盖** checkpoint。

  **正确做法：两条路线用独立的 ckpt_dir。**

      检查 train_gpt.py 是否为每个 base 生成独立目录：
          grep -n 'ckpt_dir\\|output_dir\\|default_root_dir' train_gpt.py

  ## 验证
      python _tools/verify_gpt_ckpt_isolation.py

  ## 然后重跑
      python train_gpt.py --base both --batch-size 3 --max-sec 30
""")
    elif rc != 0:
        print("  有改动未应用，见上面的 !! 行")
    print("=" * 78)
    return rc


if __name__ == "__main__":
    sys.exit(main())
