# -*- coding: utf-8 -*-
"""用**已有音频**真实测试 make_listen_any.py（不跑模型）。

找一个含 >=2 组、每组 >=8 个 wav 的已有实验目录，复制成 eval-dir 结构，
然后跑盲听集生成，验证：
  1. 每句独立打乱
  2. 响度对齐是否生效（各段 RMS 是否接近）
  3. 答案文件正确
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PY = r"C:\Users\A\miniconda3\envs\voice\python.exe"
V5 = Path(r"D:\A\zfy-v5")

# 找已有的合成目录
pool = []
for d in (V5 / "out").rglob("*"):
    if d.is_dir():
        n = len(list(d.glob("*.wav")))
        if n >= 8:
            pool.append((n, d))
pool.sort(key=lambda t: -t[0])
print("可用的已有音频目录（前 6）:")
for n, d in pool[:6]:
    print(f"  {n:3d} 个  {d.relative_to(V5)}")

if len(pool) < 2:
    print("\n不足 2 个目录，无法测试")
    sys.exit(0)

# 取前两个目录，复制前 8 个 wav 成 eval-dir 结构
tmp = Path(tempfile.mkdtemp(prefix="listen_test_"))
ev = tmp / "eval"
try:
    tags = []
    for i, (n, d) in enumerate(pool[:2]):
        tag = f"cand{i}"
        tags.append(tag)
        dst = ev / tag
        dst.mkdir(parents=True, exist_ok=True)
        for j, w in enumerate(sorted(d.glob("*.wav"))[:8]):
            shutil.copy2(w, dst / f"{j:02d}.wav")
        print(f"  复制 {d.name} -> {tag}  (8 个)")

    print()
    r = subprocess.run(
        [PY, str(V5 / "make_listen_any.py"),
         "--eval-dir", str(ev), "--tags", ",".join(tags),
         "--n", "4", "--seed", "7"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(V5))
    print(r.stdout[-2500:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-800:])
        sys.exit(1)

    # 验证 1：答案文件
    out = ev.parent / "listen_any"
    key = json.loads((out / "answer_key.json").read_text(encoding="utf-8"))
    print("答案（前 4 句的顺序）:")
    for k, v in list(key["order"].items())[:4]:
        print(f"  {k}: {v}")

    # 验证 2：顺序是否真的被打乱（两段以上应有多种排列）
    orders = [tuple(v) for v in key["order"].values()]
    print(f"\n不同排列数: {len(set(orders))} / {len(orders)}  "
          f"{'OK 已打乱' if len(set(orders)) > 1 else '!! 未打乱'}")

    # 验证 3：响度对齐（比较各段 RMS）
    print("\n响度对齐检查（把拼接文件按段切回来量 RMS）:")
    SR = 48000
    for fn in sorted(out.glob("*.wav"))[:2]:
        x, sr = sf.read(str(fn), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        n_tags = len(key["order"][fn.name])
        seg_len = (len(x) - int(0.5 * SR)) // n_tags
        print(f"  {fn.name}  总长 {len(x)/SR:.2f}s  {n_tags} 段")
        for i in range(n_tags):
            a = int(0.5 * SR) + i * seg_len
            seg = x[a:a + seg_len]
            # 去掉段尾静音再量
            act = seg[:int(len(seg) * 0.8)]
            rms = float(np.sqrt(np.mean(act ** 2)) + 1e-12)
            print(f"    段{i}: RMS {20*np.log10(rms):7.2f} dBFS")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
