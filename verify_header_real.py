# -*- coding: utf-8 -*-
"""用**真实权重**验证补头不会破坏 torch.load。

## 上一版测试的缺陷
我用稀疏零填充伪造了一个 736 MB 的文件，那会**破坏 zip 的中央目录**
（在文件末尾），所以 torch.load 报 unpickling stack underflow ——
那是伪造文件的错，不是补头的错。

## 本次用真实文件
拿生产 v2Pro 权重（134.9 MB，真实 zip）做三步：
  1. 原样 torch.load          -> 应成功
  2. 头 b"05" 换成 b"07" 后 load -> 应仍成功（只动 2 字节）
  3. 换回 b"05" 后 load        -> 应仍成功
并顺便确认 get_sovits_version_from_path_fast 的判定随之变化。

官方 my_save2 就是 `data = byte + data[2:]`（process_ckpt.py:37），
所以理论上可行；这里做的是**实测确认**。
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
sys.path.insert(0, str(GSV))
sys.path.insert(0, str(GSV / "GPT_SoVITS"))

spec = importlib.util.spec_from_file_location(
    "pc", GSV / "GPT_SoVITS" / "process_ckpt.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)
get_ver = pc.get_sovits_version_from_path_fast

REAL = Path(r"D:\A\voiceclone\zfh-models-v2pro\torch_weights"
            r"\zfh_v2pro_e12_s384.pth")

print("=" * 74)
print(f"真实权重: {REAL.name}")
print("=" * 74)
if not REAL.exists():
    print("  不存在，中止")
    sys.exit(1)
print(f"  大小: {REAL.stat().st_size/1e6:.1f} MB")

tmp = Path(tempfile.mkdtemp(prefix="hdr_real_"))
try:
    work = tmp / "t.pth"
    shutil.copy2(REAL, work)

    def head():
        return work.read_bytes()[:2]

    def try_load(tag: str):
        try:
            obj = torch.load(str(work), map_location="cpu", weights_only=False)
            keys = list(obj.keys()) if isinstance(obj, dict) else "?"
            w = obj.get("weight") if isinstance(obj, dict) else None
            n = len(w) if isinstance(w, dict) else "?"
            print(f"  {tag:22s} torch.load OK   顶层键={keys}  weight 条目={n}")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  {tag:22s} torch.load **失败** {type(e).__name__}: "
                  f"{str(e)[:120]}")
            return False

    def verdict(tag: str):
        r = get_ver(str(work))
        print(f"  {tag:22s} 判定 -> {r}  model_version={r[1]!r}")
        return r[1]

    print()
    print("--- 1) 原样（头 b'05' = v2Pro）---")
    print(f"  头 = {head()!r}")
    ok1 = try_load("原始")
    v1 = verdict("原始")

    print()
    print("--- 2) 换成 b'07'（v5dev）---")
    data = work.read_bytes()
    work.write_bytes(b"07" + data[2:])
    print(f"  头 = {head()!r}   大小不变: "
          f"{work.stat().st_size == REAL.stat().st_size}")
    ok2 = try_load("改头后")
    v2 = verdict("改头后")

    print()
    print("--- 3) 换回 b'05' ---")
    data2 = work.read_bytes()
    work.write_bytes(b"05" + data2[2:])
    print(f"  头 = {head()!r}")
    ok3 = try_load("还原后")
    v3 = verdict("还原后")

    print()
    print("=" * 74)
    print("结论")
    print("=" * 74)
    all_load = ok1 and ok2 and ok3
    print(f"  torch.load 三次都成功: {all_load}")
    print(f"  判定随头变化: {v1!r} -> {v2!r} -> {v3!r}")
    if all_load and v2 == "v5dev" and v1 == "v2Pro":
        print()
        print("  ✓ 补头方案可行：只改前 2 字节，不影响 zip 结构，")
        print("    且能让推理端正确识别版本")
    else:
        print()
        print("  ✗ 有问题，需要另找方案")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
