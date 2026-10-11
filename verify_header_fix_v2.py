# -*- coding: utf-8 -*-
"""用**正确的方法**验证补头安全性（上一版测试方法错了）。

## 上一版的错误
我用 `torch.load` 验证补头后的文件，结果**原始文件也失败**：
    UnpicklingError: unpickling stack underflow
这不是补头的锅 —— 而是**所有带版本头的 SoVITS 权重都无法直接 torch.load**。

原因：权重前 2 字节是版本标记（b"05"=v2Pro / b"07"=v5dev），
不是 zip 的 "PK"。正确读法是 `process_ckpt.load_sovits_new()`，
它会把 "PK" 补回去再 load。

> 这与 skill 里记录的坑一致：v2Pro ONNX 导出时踩过同一个。

## 本脚本
  1. 打印 load_sovits_new 的实现，确认它确实补 PK
  2. 用**真实 v2Pro 权重**做三步：原样 / 改成 b"07" / 还原 b"05"
     每步都用 load_sovits_new 读，并跑真实版本判定
  3. 检查 default_configs 里 v5dev 的底模路径是否存在
     （因为 b"07" 的第三位 if_lora_v3=True 会触发底模存在性检查）
"""
from __future__ import annotations

import importlib.util
import inspect
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

print("=" * 78)
print("1. load_sovits_new 的实现（确认它补 PK）")
print("=" * 78)
print(inspect.getsource(pc.load_sovits_new))

print("=" * 78)
print("2. head2version 第三位 = if_lora_v3")
print("=" * 78)
for k, v in pc.head2version.items():
    print(f"  {k!r} -> version={v[0]!r:6s} model_version={v[1]!r:12s} "
          f"if_lora_v3={v[2]}")

REAL = Path(r"D:\A\voiceclone\zfh-models-v2pro\torch_weights"
            r"\zfh_v2pro_e12_s384.pth")

print()
print("=" * 78)
print("3. 真实权重三步测试（用 load_sovits_new，不是 torch.load）")
print("=" * 78)
print(f"  文件: {REAL.name}  {REAL.stat().st_size/1e6:.1f} MB")

tmp = Path(tempfile.mkdtemp(prefix="hdr2_"))
results = []
try:
    work = tmp / "t.pth"
    shutil.copy2(REAL, work)

    def probe(tag: str):
        head = work.read_bytes()[:2]
        ver = pc.get_sovits_version_from_path_fast(str(work))
        try:
            d = pc.load_sovits_new(str(work))
            nw = len(d.get("weight", {}))
            info = d.get("info", "?")
            load_ok, detail = True, f"weight {nw} 条  info={info!r}"
        except Exception as e:  # noqa: BLE001
            load_ok, detail = False, f"{type(e).__name__}: {str(e)[:90]}"
        results.append((tag, head, ver[1], load_ok))
        print(f"  {tag:10s} 头={head!r}  判定={ver[1]!r:10s} "
              f"load_sovits_new={'OK ' if load_ok else 'FAIL'}  {detail}")

    probe("原样")
    data = work.read_bytes()
    work.write_bytes(b"07" + data[2:])
    probe("改b'07'")
    data2 = work.read_bytes()
    work.write_bytes(b"05" + data2[2:])
    probe("还原b'05'")

    print()
    print("=" * 78)
    print("4. default_configs：v5dev 的底模路径是否存在")
    print("=" * 78)
    print("  （b'07' 的 if_lora_v3=True，init_vits_weights 会检查底模是否存在）")
    import os
    os.chdir(GSV)
    try:
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config
        cfg = TTS_Config(str(GSV / "GPT_SoVITS" / "configs" / "tts_infer.yaml"))
        for v in ("v5dev", "v5turbo", "v2Pro", "v4"):
            if v not in cfg.default_configs:
                print(f"  {v:9s} 不在 default_configs 里")
                continue
            p = cfg.default_configs[v].get("vits_weights_path", "")
            ex = os.path.exists(p) if p else False
            print(f"  {v:9s} vits_weights_path = {p}")
            print(f"  {'':9s} 存在: {ex}  "
                  f"{'' if ex else '**缺失 -> if_lora_v3 检查会抛 FileNotFoundError**'}")
    except Exception as e:  # noqa: BLE001
        print(f"  读取 default_configs 失败: {type(e).__name__}: {str(e)[:200]}")

    print()
    print("=" * 78)
    print("5. 结论")
    print("=" * 78)
    ok = all(r[3] for r in results)
    print(f"  load_sovits_new 三次都成功: {ok}")
    for tag, head, mv, lok in results:
        print(f"    {tag:10s} 头={head!r} -> {mv!r:10s} load={'OK' if lok else 'FAIL'}")
    if ok and results[1][2] == "v5dev" and results[0][2] == "v2Pro":
        print()
        print("  ✓ 补头方案可行：只改前 2 字节，load_sovits_new 照常可读，")
        print("    且版本判定从 v2Pro 正确变为 v5dev")
        print("  ✓ 上一版测试的 torch.load 失败是**方法错误**，不是方案问题")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
