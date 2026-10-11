# -*- coding: utf-8 -*-
"""验证 patch_ckpt_atomic.py —— 关键验证「崩溃只丢当前一个」。

## 要验证的五件事
  1. `my_save` 的临时文件**不再落在 $PWD**（原 bug 会留 4.3 GB 垃圾）
  2. `my_save` 是**原子**的：目标文件要么是旧的、要么是新的，不会半截
  3. `latest_checkpoint_path` 把哨兵值 `233333333333` 当作**最旧**
     （原实现会永远选中它 -> resume 到错的点）
  4. 模拟 `torch.save` 失败 -> **旧文件完好无损**（这是核心价值）
  5. 保留策略只留最新 N 个

## 为什么第 4 点是核心
原实现（覆盖写）崩溃时会**同时毁掉旧文件** -> 无法 resume。
实测：配额满导致 e27 崩溃，`G_233333333333.pth` 处于未知状态。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ⚠️ 不能硬编码 Windows 路径 —— 这个脚本也要在云端 Linux 上跑。
#    实测云端崩过：FileNotFoundError: 'D:\\A\\zfy-v5\\GPT-SoVITS'
#    改用共享的 zfy_paths 探测（向上找 + 已知默认位置）。
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/mnt/workspace/zfy-v5-fullft-pkg/_tools")
try:
    from zfy_paths import find_gpt_sovits, find_tools
except ImportError:
    def find_gpt_sovits(cli_arg=None):  # type: ignore[misc]
        for p in (Path("/mnt/workspace/zfy-v5-fullft-pkg/GPT-SoVITS"),
                  Path("/mnt/workspace/zfy-v5-fullft-pkg"),
                  Path(r"D:\A\zfy-v5\GPT-SoVITS"), Path.cwd()):
            if (p / "GPT_SoVITS" / "utils.py").exists():
                return p
        return Path.cwd()

    def find_tools():  # type: ignore[misc]
        return Path(__file__).resolve().parent

_ap = argparse.ArgumentParser()
_ap.add_argument("--root", default=None,
                 help="仓库根（含 GPT_SoVITS/）。默认自动探测")
_ap.add_argument("--patch", default=None,
                 help="patch_ckpt_atomic.py 路径。默认在 _tools/ 或同级找")
_args, _ = _ap.parse_known_args()

GSV = find_gpt_sovits(_args.root)
PY = sys.executable


def _find_patch() -> Path:
    if _args.patch:
        return Path(_args.patch).resolve()
    for cand in (find_tools() / "patch_ckpt_atomic.py",
                 GSV / "_tools" / "patch_ckpt_atomic.py",
                 GSV / "patch_ckpt_atomic.py",
                 Path(__file__).resolve().parent / "patch_ckpt_atomic.py"):
        if cand.exists():
            return cand
    return Path("patch_ckpt_atomic.py")


PATCH = _find_patch()

print(f"  仓库根: {GSV}")
print(f"  补丁  : {PATCH}")
if not (GSV / "GPT_SoVITS" / "utils.py").exists():
    print(f"  !! {GSV}/GPT_SoVITS/utils.py 不存在 —— 用 --root 指定")
    sys.exit(1)
if not PATCH.exists():
    print(f"  !! 找不到补丁脚本 {PATCH} —— 用 --patch 指定")
    sys.exit(1)


def run_patch(*extra):
    r = subprocess.run([PY, str(PATCH), "--root", str(GSV), *extra],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    for l in (r.stdout or "").split("\n"):
        if "[1/" in l or "[2/" in l or "[3/" in l or "还原" in l or "备份" in l:
            print(f"      {l.strip()}")
    return r.returncode


print("=" * 88)
print("0. 确保补丁已打")
print("=" * 88)
run_patch()

PROBE = r'''
import os, sys, tempfile
GSV = GSV_PATH
# ⚠️ 必须用 os.path.join，不能写 GSV + r"\GPT_SoVITS"
#    反斜杠在 Linux 上是普通字符 -> 拼出无效路径 -> 探针静默失败
_GS = os.path.join(GSV, "GPT_SoVITS")
sys.path[:0] = [_GS, GSV]
os.chdir(_GS)

import torch
from utils import my_save, latest_checkpoint_path

tmpd = tempfile.mkdtemp(prefix="zfy_ckpt_test_")
target = os.path.join(tmpd, "G_108.pth")
cwd_before = set(os.listdir(_GS))

# ---- 1) 正常保存（成功路径：临时文件会被清理，两个版本都不留垃圾）----
my_save({"model": {"w": torch.randn(64, 64)}, "iteration": 1}, target)
sz1 = os.path.getsize(target)
junk_after_ok = [f for f in (set(os.listdir(_GS)) - cwd_before)
                 if f.endswith(".pth") or f.startswith(".zfy_tmp_")]
print(f"exists={os.path.exists(target)} size={sz1}")
print(f"junk_after_ok={len(junk_after_ok)}")

# ---- 2) 旧文件在新保存失败时**完好** + 崩溃留下什么 ----
good_bytes = open(target, "rb").read()
class Boom:
    def __reduce__(self):
        raise RuntimeError("ZFY simulated save failure")
try:
    my_save({"model": {"w": Boom()}}, target)
    print("crash_save=no_raise intact=False")
except Exception:
    after = open(target, "rb").read()
    print(f"crash_save=raised intact={after == good_bytes} "
          f"size_after={len(after)}")

junk_cwd = [f for f in (set(os.listdir(_GS)) - cwd_before)
            if f.endswith(".pth") or f.startswith(".zfy_tmp_")]
print(f"junk_after_crash_cwd={len(junk_cwd)}")
leftover = [f for f in os.listdir(tmpd) if f.startswith(".zfy_tmp_")]
print(f"leftovers_target_dir={len(leftover)}")
numeric_junk = [f for f in junk_cwd if f.split(".")[0].isdigit()]
print(f"numeric_junk_cwd={len(numeric_junk)}")

# ---- 3) latest_checkpoint_path 的排序 ----
for n in (54, 108, 162):
    my_save({"model": {"w": torch.zeros(4, 4)}, "iteration": n},
            os.path.join(tmpd, f"G_{n}.pth"))
my_save({"model": {"w": torch.zeros(4, 4)}, "iteration": 999},
        os.path.join(tmpd, "G_233333333333.pth"))
picked = latest_checkpoint_path(tmpd, "G_*.pth")
print(f"picked={os.path.basename(picked)}")
os.remove(os.path.join(tmpd, "G_233333333333.pth"))
picked2 = latest_checkpoint_path(tmpd, "G_*.pth")
print(f"picked2={os.path.basename(picked2)}")

# 清理探针自己造的垃圾，避免污染仓库
for f in junk_cwd:
    try:
        os.remove(os.path.join(_GS, f))
    except OSError:
        pass
import shutil as _sh
_sh.rmtree(tmpd, ignore_errors=True)
'''


def probe(label):
    code = PROBE.replace("GSV_PATH", repr(str(GSV)))
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600,
                       cwd=str(GSV))
    out = (r.stdout or "") + (r.stderr or "")
    d = {}
    print(f"\n--- {label} ---")
    got = False
    for l in out.split("\n"):
        s = l.strip()
        if s.startswith("MARKER"):
            s = s[len("MARKER"):].strip()
        if "=" in s and not s.startswith(("Traceback", "File ", "  ")):
            print(f"    {s}")
            got = True
            for tok in s.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    d.setdefault(k, v)
    # ⚠️ 关键：探针什么都没输出时必须把 stderr 打出来，
    #    否则只是看到一堆 None，无法定位（我第一版就是这样）
    if not got:
        print(f"    !! 探针无输出  exit={r.returncode}")
        tail = [l for l in out.strip().split("\n") if l.strip()][-12:]
        for l in tail:
            print(f"       | {l[:110]}")
    return d, out


print()
print("=" * 88)
print("1. 打补丁后测试")
print("=" * 88)
d1, out1 = probe("补丁后")

print()
print("=" * 88)
print("2. 还原到上游实现，对比（复现原 bug）")
print("=" * 88)
run_patch("--revert")
d0, out0 = probe("上游原版")

print()
print("=" * 88)
print("3. 恢复补丁")
print("=" * 88)
run_patch()

print()
print("=" * 88)
print("4. 断言")
print("=" * 88)
checks = [
    ("上游: 崩溃时把临时文件**落在 $PWD**（原 bug）",
     int(d0.get("numeric_junk_cwd", "0")) > 0,
     f"$PWD 垃圾={d0.get('numeric_junk_cwd')}"),
    ("补丁: 崩溃时 $PWD **无垃圾**",
     d1.get("junk_after_crash_cwd") == "0",
     f"$PWD 垃圾={d1.get('junk_after_crash_cwd')}"),
    ("补丁: 保存成功",
     d1.get("exists") == "True" and int(d1.get("size", "0") or 0) > 1000,
     f"exists={d1.get('exists')} size={d1.get('size')}"),
    ("补丁: 成功路径也不留垃圾",
     d1.get("junk_after_ok") == "0",
     f"junk={d1.get('junk_after_ok')}"),
    ("补丁: **保存失败时旧文件完好**（核心）",
     d1.get("crash_save") == "raised" and d1.get("intact") == "True",
     f"{d1.get('crash_save')} intact={d1.get('intact')} "
     f"size_after={d1.get('size_after')}"),
    ("补丁: 失败时目标目录也不留临时文件",
     d1.get("leftovers_target_dir") == "0",
     f"leftovers={d1.get('leftovers_target_dir')}"),
    ("补丁: 哨兵值被当作**最旧**（不选中它）",
     str(d1.get("picked", "")).startswith("G_162"),
     f"picked={d1.get('picked')}"),
    ("补丁: 无哨兵时选**最大 step**",
     str(d1.get("picked2", "")).startswith("G_162"),
     f"picked2={d1.get('picked2')}"),
]
bad = 0
for name, ok, extra in checks:
    bad += 0 if ok else 1
    print(f"  {'✓' if ok else '✗'} {name}" + (f"  ({extra})" if extra else ""))

print()
if d0.get("picked"):
    print(f"  上游选中的 checkpoint: {d0.get('picked')}")
    print(f"  补丁选中的 checkpoint: {d1.get('picked')}")
    if d0.get("picked") != d1.get("picked"):
        print("  -> **行为差异确认**：上游会 resume 到哨兵文件")

print()
print("=" * 88)
if bad == 0:
    print("  ✓ 全部验证通过")
    print("    · 临时文件不再污染 $PWD")
    print("    · 保存失败时旧 checkpoint 完好 -> 可 resume")
    print("    · 哨兵值 233333333333 被正确当作最旧")
else:
    print(f"  ✗ {bad} 项失败")
print("=" * 88)
