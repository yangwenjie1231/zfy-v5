# -*- coding: utf-8 -*-
"""诊断：为什么 v5 全量微调权重被判定成 v3（从而去加载 BigVGAN）。

## 现象
云端合成报：
    HFValidationError: Repo id must be in the form 'repo_name' or
    'namespace/repo_name':
    '.../pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x'

说明 TTS.py:713 的 `BigVGAN.from_pretrained(本地目录)` 被调用了。
而那一支的条件是 `version == "v3"`（TTS.py:705）。

## 关键机制
`get_sovits_version_from_path_fast(sovits_path)` **不看我们传的 version**，
而是：
  1. 先算文件 hash，查 hash_pretrained_dict
  2. 再读**前 2 字节**作为版本头：`head2version[version]`
  3. 头不是 "PK" 且不在表里 -> 落到按**文件大小**猜

所以问题可能是：
  · 我们训练出的 .pth 头部字节不是 "PK"（torch.save 的 zip 头），
    于是走了 head2version 查表
  · 或者被当成 v3（750MB 那个尺寸区间）

## 本脚本
  1. 读本地 v5 全量微调权重的**前 2 字节**，查 head2version
  2. 读对照：底模 s2Gv5dev.pth / v2Pro 权重的头
  3. 看文件大小落在哪个版本区间
  4. 复现 get_sovits_version_from_path_fast 的判定
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
sys.path.insert(0, str(GSV))
sys.path.insert(0, str(GSV / "GPT_SoVITS"))

# 从 process_ckpt.py 里读表，保证与运行时一致
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "pc", GSV / "GPT_SoVITS" / "process_ckpt.py")
pc = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(pc)
except Exception as e:  # noqa: BLE001
    print(f"  导入 process_ckpt 失败: {type(e).__name__}: {e}")

print("=" * 78)
print("1. process_ckpt.py 里的版本表")
print("=" * 78)
h2v = getattr(pc, "head2version", None)
if h2v:
    for k, v in h2v.items():
        print(f"  头 {k!r:8s} -> {v}")
else:
    print("  找不到 head2version（可能定义在别处）")
    import re
    txt = (GSV / "GPT_SoVITS" / "process_ckpt.py").read_text(
        encoding="utf-8", errors="replace")
    m = re.search(r"head2version\s*=\s*\{(.*?)\}", txt, re.S)
    if m:
        print("  源码里的表:")
        for line in m.group(1).strip().split("\n"):
            print(f"    {line.strip()}")

print()
print("=" * 78)
print("2. 各权重的头 2 字节")
print("=" * 78)
CAND = [
    ("v5 底模 s2Gv5dev.pth",
     GSV / "GPT_SoVITS" / "pretrained_models" / "gsv-v5-pretrained"
     / "s2Gv5dev.pth"),
    ("v5 底模 s2Gv5turbo.pth",
     GSV / "GPT_SoVITS" / "pretrained_models" / "gsv-v5-pretrained"
     / "s2Gv5turbo.pth"),
    ("生产 v2Pro 权重",
     Path(r"D:\A\voiceclone\zfh-models-v2pro\torch_weights"
          r"\zfh_v2pro_e12_s384.pth")),
    ("本地 v5 LoRA r32 e8",
     GSV / "SoVITS_weights_v5" / "zfy_v5_e8_s432_l32.pth"),
]
for label, p in CAND:
    if not p.exists():
        print(f"  {label:32s} 不存在")
        continue
    with p.open("rb") as f:
        head = f.read(2)
    size = p.stat().st_size
    verdict = "?"
    if h2v and head in h2v:
        verdict = f"head2version -> {h2v[head]}"
    elif head == b"PK":
        verdict = "是 zip（PK），走按大小判定"
    else:
        verdict = "**未知头，不在表里**"
    print(f"  {label:32s} 头={head!r:8s} 大小={size/1e6:8.1f} MB  {verdict}")

print()
print("=" * 78)
print("3. 按文件大小判定（源码注释给的区间）")
print("=" * 78)
print("""
  v1: ~82,942 KB
  v2: ~83,014 KB
  v3: ~750 MB
  （源码注释，实际还有更多分支）
""")

print("=" * 78)
print("4. 结论与修法")
print("=" * 78)
print("""
  关键：`get_sovits_version_from_path_fast` **完全不看我们传的 version**，
  它靠 ① hash 表 ② 头 2 字节 ③ 文件大小 来判定。

  如果我们的 .pth 头不是已知的版本字节，就会落到按大小猜 ——
  而 v5 全量微调的权重约 736 MB，与 v3 的 ~750 MB 接近，
  **很可能被误判成 v3**，于是去加载 24kHz 的 BigVGAN。

  修法（按优先级）：
    A. 让权重带上正确的版本头
       —— `process_ckpt.savee()` 会写版本字节；我们用的是
          `if_save_every_weights: true`，应该已经写了头。
          需要确认写的是哪个版本（v5dev -> b"07"？）
    B. 若头不对，用 `sovits_version` 显式覆盖
       —— TTS_Config 里可能有 version 字段能强制指定
    C. 最稳：不用 TTS_infer_pack，改用 inference_webui 的路径
       —— 它同样依赖这个判定，不解决问题
""")
