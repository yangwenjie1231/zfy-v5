# -*- coding: utf-8 -*-
"""决定性诊断：为什么 `load audio or ssl error`。

## 背景

冒烟训练满屏报错，且「一秒钟出结果」。根因是 data_utils.py:122 的
`except:` 把异常吞掉，返回**全零张量**：

    spec = torch.zeros(1025, 100)
    wav  = torch.zeros(1, 100 * hop_length)
    ssl  = torch.zeros(1, 768, 100)

训练"成功"了，但学的是垃圾。

## 两个候选原因（本脚本一次分清）

  A. 特征文件是 LFS 指针（未真正拉取）
     -> logs/zfy_v5/5-wav32k/*.wav 只有 130 字节

  B. **ffmpeg 命令行程序缺失**
     -> load_audio 用 ffmpeg.input(...).run(cmd=["ffmpeg", ...])
        需要系统里的 ffmpeg 可执行文件，不是 python 的 ffmpeg-python 包

## 本脚本

  1. 查 ffmpeg 可执行文件
  2. 查特征文件体积（找出指针）
  3. **绕过 except，打印真实异常**（最关键）
  4. 给出对应修复

用法（云上，检出根）：
    python _tools/diag_load_audio.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
if not (ROOT / "GPT_SoVITS").is_dir():
    for c in (Path("/mnt/workspace/zfy-v5-fullft-pkg"),
              Path("/mnt/workspace/zfy-v5")):
        if (c / "GPT_SoVITS").is_dir():
            ROOT = c
            break

print("=" * 78)
print("1. ffmpeg 可执行文件（load_audio 依赖它，不是 python 包）")
print("=" * 78)
exe = shutil.which("ffmpeg")
print(f"  which ffmpeg : {exe or '**未找到**'}")
if exe:
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True,
                           text=True, timeout=30)
        print(f"  版本: {r.stdout.split(chr(10))[0][:80]}")
    except Exception as e:  # noqa: BLE001
        print(f"  执行失败: {e}")
else:
    print("  !! 需要安装：apt-get install -y ffmpeg")

print()
print("=" * 78)
print("2. 特征文件体积（130 字节 = LFS 指针）")
print("=" * 78)
OPT = ROOT / "logs" / "zfy_v5"
for sub in ("4-cnhubert", "5-wav32k", "3-bert"):
    d = OPT / sub
    if not d.is_dir():
        print(f"  {sub:14s} **目录不存在**")
        continue
    files = [f for f in d.iterdir() if f.is_file()]
    small = [f for f in files if f.stat().st_size < 1024]
    sizes = sorted(f.stat().st_size for f in files)
    med = sizes[len(sizes) // 2] if sizes else 0
    print(f"  {sub:14s} {len(files):4d} 个   指针(<1KB): {len(small):4d}   "
          f"中位 {med:>9,d} 字节")

for name in ("2-name2text.txt", "6-name2semantic.tsv"):
    p = OPT / name
    print(f"  {name:14s} {p.stat().st_size if p.exists() else '缺失'} 字节")

print()
print("=" * 78)
print("3. 绕过 except，打印真实异常（最关键）")
print("=" * 78)

# 找一个真实存在的 wav
wav_dir = OPT / "5-wav32k"
cand = None
if wav_dir.is_dir():
    for f in sorted(wav_dir.iterdir()):
        if f.is_file():
            cand = f
            break

if cand is None:
    print("  找不到 wav，跳过")
else:
    print(f"  测试文件: {cand.name}")
    print(f"  体积: {cand.stat().st_size} 字节")
    head = cand.read_bytes()[:40]
    if head.startswith(b"version https://git-lfs"):
        print(f"  内容: **LFS 指针**")
        print(f"    {head.decode('utf-8','replace').split(chr(10))[0]}")
    else:
        print(f"  文件头: {head[:16].hex()}")

    # 复现 load_audio，但**不吞异常**
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "GPT_SoVITS"))
    sys.path.insert(0, str(ROOT / "tools"))
    os.chdir(ROOT)
    try:
        import ffmpeg as ffmpeg_py
        out, _ = (
            ffmpeg_py.input(str(cand), threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=32000)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True,
                 capture_stderr=True)
        )
        print(f"  解码成功: {len(out)} 字节 PCM")
        n = len(out) // 4
        print(f"    -> {n} 个 float32 样本 = {n/32000:.2f} 秒")
    except FileNotFoundError as e:
        print(f"  **ffmpeg 可执行文件缺失**: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"  **解码失败**: {type(e).__name__}")
        msg = str(e)
        print(f"    {msg[:600]}")

print()
print("=" * 78)
print("结论与修复")
print("=" * 78)
print("""
  根据上面第 1 与第 3 节判断：

  【若 ffmpeg 未找到】
      apt-get update && apt-get install -y ffmpeg
      # 或 conda install -y -c conda-forge ffmpeg

  【若特征文件是 130 字节指针】
      cd """ + str(ROOT) + """
      git lfs install --local
      rm -rf .git/lfs/objects .git/lfs/incomplete
      git checkout --force -- logs/
      git lfs pull
      # 验证：ls -la logs/zfy_v5/5-wav32k/ | head
      #   应为几十~几百 KB，不是 130 字节

  【修好后必须重跑冒烟】
      数据是垃圾时训练也会"成功"（因为异常被吞、喂的是全零张量），
      所以冒烟通过不等于数据正常。修完要重跑并确认没有
      'load audio or ssl error' 输出。
""")
