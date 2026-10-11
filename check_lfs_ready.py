# -*- coding: utf-8 -*-
"""训练前检查：模型权重是真文件还是 LFS 指针。

## 为什么需要

冒烟训练失败在最后一步：
    RuntimeError: PytorchStreamReader failed reading zip archive:
    failed finding central directory

原因是 `s2Gv5dev.pth` 是 134 字节的 LFS 指针文本，不是 766 MB 的权重。
`torch.load` 试图把这段文本当 zip 读，于是报"找不到中央目录"。

这个检查很便宜（只读文件头 200 字节 + stat 体积），
但能**在花几分钟加载模型之前**就报出问题。

## 判据

  · 文件 < 1 KB 且以 "version https://git-lfs" 开头  -> LFS 指针（未拉取）
  · 文件体积与预期差距 > 20%                          -> 可能损坏
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
if not (ROOT / "GPT_SoVITS").is_dir():
    for c in (Path("/mnt/workspace/zfy-v5-fullft-pkg"),
              Path("/mnt/workspace/zfy-v5"),
              Path(r"D:\A\zfy-v5\GPT-SoVITS")):
        if (c / "GPT_SoVITS").is_dir():
            ROOT = c
            break

# 训练必需的大文件 + 预期体积（MB）
REQUIRED = [
    ("GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth", 766),
    ("GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth", 58),
    ("GPT_SoVITS/pretrained_models/s1v3.ckpt", 155),
    ("GPT_SoVITS/pretrained_models/chinese-hubert-base/pytorch_model.bin", 189),
    ("GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin", 651),
]

# 特征文件（也走 LFS，但训练时按目录批量读，单独抽查几个）
FEATURE_DIRS = [
    ("logs/zfy_v5/4-cnhubert", ".pt"),
    ("logs/zfy_v5/5-wav32k", ""),
]

print("=" * 74)
print("LFS 完整性检查")
print("=" * 74)
print(f"检出根: {ROOT}\n")

pointers = []
missing = []
ok = []

for rel, exp_mb in REQUIRED:
    p = ROOT / rel
    name = Path(rel).name
    if not p.exists():
        print(f"  缺失   {name}")
        missing.append(rel)
        continue
    sz = p.stat().st_size
    head = p.read_bytes()[:60]
    is_ptr = sz < 1024 and head.startswith(b"version https://git-lfs")
    if is_ptr:
        print(f"  指针   {name:34s} {sz:>6d} 字节  (应为 {exp_mb} MB)")
        pointers.append(rel)
    else:
        ratio = sz / (exp_mb * 1e6)
        flag = "OK  " if ratio > 0.8 else "偏小"
        print(f"  {flag} {name:34s} {sz/1e6:8.1f} MB  (预期 {exp_mb} MB)")
        if ratio <= 0.8:
            pointers.append(rel)
        else:
            ok.append(rel)

print()
print("特征文件抽查（各目录第一个文件）:")
for rel, ext in FEATURE_DIRS:
    d = ROOT / rel
    if not d.is_dir():
        print(f"  缺失   {rel}")
        missing.append(rel)
        continue
    files = sorted(f for f in d.iterdir() if f.is_file())
    if not files:
        print(f"  空目录 {rel}")
        missing.append(rel)
        continue
    f = files[0]
    sz = f.stat().st_size
    is_ptr = sz < 1024 and f.read_bytes()[:60].startswith(b"version https://git-lfs")
    print(f"  {'指针' if is_ptr else 'OK  '} {rel:32s} {len(files):4d} 个, "
          f"首个 {sz/1024:8.1f} KB")
    if is_ptr:
        pointers.append(f"{rel}/{f.name}")

print()
print("=" * 74)
if pointers:
    print(f"发现 {len(pointers)} 个 LFS 指针（内容未拉取）")
    print("=" * 74)
    print("""
  修复（在检出根执行）：
      git lfs install --local
      git lfs pull

  验证体积：
      ls -la GPT_SoVITS/pretrained_models/gsv-v5-pretrained/
      # s2Gv5dev.pth 应为 ~766 MB（不是 134 字节）

  若 git lfs pull 无效（远程对象缺失），重新 clone（**不要**加
  GIT_LFS_SKIP_SMUDGE）：
      cd /mnt/workspace
      rm -rf zfy-v5-fullft-pkg
      git lfs install
      git clone <url> zfy-v5-fullft-pkg
""")
    sys.exit(1)
elif missing:
    print(f"缺少 {len(missing)} 个文件/目录:")
    for m in missing:
        print(f"  {m}")
    sys.exit(1)
else:
    print(f"✓ 全部 {len(ok)} 个模型文件 + 特征目录完整，可以训练")
    sys.exit(0)
