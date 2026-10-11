# -*- coding: utf-8 -*-
"""验证 fix_weight_header.py：造一个假 zip 头文件，测补头后能否被正确识别。

## 为什么要这样测
本地没有 v5 全量微调的权重（都在云上），所以用**合成样本**验证：
  1. 造一个以 "PK" 开头的假文件（模拟 my_save 的产物）
  2. 补头成 b"07"
  3. 用**真实的 get_sovits_version_from_path_fast** 判定，确认返回 v5dev

这样能确认修法真的能让推理端识别正确，而不是"看起来对"。
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
sys.path.insert(0, str(GSV))
sys.path.insert(0, str(GSV / "GPT_SoVITS"))

spec = importlib.util.spec_from_file_location(
    "pc", GSV / "GPT_SoVITS" / "process_ckpt.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)

get_ver = pc.get_sovits_version_from_path_fast
V2B = pc.model_version2byte

print("=" * 74)
print("1. model_version2byte 表（源码里读出来的）")
print("=" * 74)
for k, v in V2B.items():
    print(f"  {k:12s} -> {v!r}")

print()
print("=" * 74)
print("2. 造样本：模拟 my_save 的产物（PK 头，736 MB 量级）")
print("=" * 74)
tmp = Path(tempfile.mkdtemp(prefix="hdr_test_"))
try:
    # 用真实 torch.save 造一个 zip，再补到 ~736MB 让大小落在 v3 区间
    import torch
    p_pk = tmp / "fake_fullft.pth"
    torch.save({"weight": {"dummy": torch.zeros(4)}, "config": {}}, str(p_pk))
    real = p_pk.stat().st_size
    # 用稀疏填充把大小撑到 736 MB（不占磁盘）
    with p_pk.open("r+b") as f:
        f.seek(736 * 1024 * 1024 - 1)
        f.write(b"\0")
    print(f"  文件: {p_pk.name}")
    print(f"  头  : {p_pk.read_bytes()[:2]!r}")
    print(f"  大小: {p_pk.stat().st_size/1e6:.1f} MB （真实内容 {real} 字节 + 稀疏填充）")

    print()
    print("=" * 74)
    print("3. 补头前 —— 用真实判定函数看结果")
    print("=" * 74)
    try:
        r = get_ver(str(p_pk))
        print(f"  get_sovits_version_from_path_fast -> {r}")
        print(f"  => 解析出 model_version = {r[1]!r}")
        if r[1] == "v3":
            print("     **复现了误判成 v3**（正是云端 BigVGAN 报错的原因）")
    except Exception as e:  # noqa: BLE001
        print(f"  抛出 {type(e).__name__}: {str(e)[:150]}")
        print("  （也说明头不对时行为异常）")

    print()
    print("=" * 74)
    print("4. 补头 b'07'（v5dev）")
    print("=" * 74)
    data = p_pk.read_bytes()
    p_fixed = tmp / "fixed.pth"
    p_fixed.write_bytes(V2B["v5dev"] + data[2:])
    print(f"  新头: {p_fixed.read_bytes()[:2]!r}")
    print(f"  大小: {p_fixed.stat().st_size/1e6:.1f} MB （与原来一致）")

    print()
    print("=" * 74)
    print("5. 补头后 —— 判定结果")
    print("=" * 74)
    try:
        r2 = get_ver(str(p_fixed))
        print(f"  get_sovits_version_from_path_fast -> {r2}")
        print(f"  => model_version = {r2[1]!r}  "
              f"{'✓ 正确' if r2[1] == 'v5dev' else '✗ 仍不对'}")
    except Exception as e:  # noqa: BLE001
        print(f"  抛出 {type(e).__name__}: {str(e)[:150]}")

    print()
    print("=" * 74)
    print("6. 关键：补头后 torch.load 还能读吗")
    print("=" * 74)
    try:
        obj = torch.load(str(p_fixed), map_location="cpu", weights_only=False)
        print(f"  torch.load 成功，顶层键: {list(obj.keys())}")
        print("  => 只换前 2 字节不影响 zip 结构 ✓")
    except Exception as e:  # noqa: BLE001
        print(f"  torch.load 失败: {type(e).__name__}: {str(e)[:200]}")
        print("  !! 说明补头方式有问题")

    print()
    print("=" * 74)
    print("7. 用真实的 fix_weight_header.py 跑一遍")
    print("=" * 74)
    d2 = tmp / "batch"
    d2.mkdir()
    shutil.copy2(p_pk, d2 / "v5_full_e32_e8_s432_l0.pth")
    shutil.copy2(p_pk, d2 / "v5_full_e32_e16_s864_l0.pth")
    import subprocess
    r3 = subprocess.run(
        [sys.executable, r"D:\A\zfy-v5\fix_weight_header.py",
         "--dir", str(d2), "--apply"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r3.stdout[-1200:])
    for f in sorted(d2.glob("*.pth")):
        print(f"  {f.name} 头={f.read_bytes()[:2]!r}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
