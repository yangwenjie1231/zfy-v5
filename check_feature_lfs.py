# -*- coding: utf-8 -*-
"""检查 logs/zfy_v5/ 里的特征文件在远程是不是 LFS 指针。

## 为什么

冒烟训练满屏 `load audio or ssl error!!!!!!`，且瞬间跑完 ——
因为 data_utils.py:122 的 `except:` 把异常吞掉，返回**全零张量**：

    spec = torch.zeros(1025, 100)
    wav  = torch.zeros(1, 100 * hop_length)
    ssl  = torch.zeros(1, 768, 100)

所以训练"成功"但学的是垃圾。

## 怀疑

4-cnhubert/*.pt 与 5-wav32k/*.wav 在远程可能也是 LFS 指针。
我之前只查了 5 个模型权重文件，**没查特征文件** —— 这是检查的盲区。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

tok = (Path.home() / ".modelscope" / "credentials" / "git_token") \
    .read_text(encoding="utf-8").strip()
url = ("https://oauth2:" + tok + "@www.modelscope.cn/datasets/"
       "yangwenjie1231/zfy-v5-fullft-pkg.git")

tmp = Path(tempfile.mkdtemp(prefix="ms_feat_"))
try:
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                    url, "r"], cwd=str(tmp), capture_output=True, env=env)
    repo = tmp / "r"

    out = subprocess.run(["git", "ls-tree", "-r", "-l", "HEAD"],
                         cwd=str(repo), capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout

    # 按目录统计：小文件(<1KB)视为 LFS 指针
    print("=" * 78)
    print("远程各目录的文件体积分布")
    print("=" * 78)

    dirs = {}
    for line in out.split("\n"):
        parts = line.split()
        if len(parts) < 5:
            continue
        size_s, path = parts[3], parts[4]
        if not size_s.isdigit():
            continue
        size = int(size_s)
        if "/" not in path:
            continue
        # 取前 3 层作为分组
        key = "/".join(path.split("/")[:3])
        dirs.setdefault(key, []).append(size)

    KEY_DIRS = [
        "logs/zfy_v5/2-name2text.txt",
        "logs/zfy_v5/3-bert",
        "logs/zfy_v5/4-cnhubert",
        "logs/zfy_v5/5-wav32k",
        "logs/zfy_v5/6-name2semantic.tsv",
        "data/zfy_v5",
    ]
    for k in sorted(dirs):
        if not any(k.startswith(kd.rsplit("/", 1)[0]) for kd in KEY_DIRS):
            continue
        sizes = dirs[k]
        ptr = sum(1 for s in sizes if s < 1024)
        print(f"  {k:32s} {len(sizes):5d} 个   指针(<1KB): {ptr:5d}   "
              f"中位 {sorted(sizes)[len(sizes)//2]:>9,d} 字节")

    print()
    print("=" * 78)
    print("逐个抽查关键路径")
    print("=" * 78)
    info = {}
    for line in out.split("\n"):
        parts = line.split()
        if len(parts) >= 5 and parts[3].isdigit():
            info[parts[4]] = int(parts[3])

    SAMPLES = [
        "logs/zfy_v5/2-name2text.txt",
        "logs/zfy_v5/6-name2semantic.tsv",
        "data/zfy_v5/zfy_v5.list",
    ]
    for s in SAMPLES:
        sz = info.get(s)
        print(f"  {s:44s} {sz if sz is not None else '?':>10} 字节")

    for d in ("logs/zfy_v5/3-bert", "logs/zfy_v5/4-cnhubert",
              "logs/zfy_v5/5-wav32k"):
        files = [(p, s) for p, s in info.items() if p.startswith(d + "/")]
        if not files:
            print(f"  {d:44s} 无文件")
            continue
        ptr = sum(1 for _, s in files if s < 1024)
        ex_p, ex_s = files[0]
        print(f"  {d:44s} {len(files)} 个，指针 {ptr}")
        print(f"      例: {Path(ex_p).name[:44]:46s} {ex_s} 字节")

    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    feat_ptr = 0
    feat_total = 0
    for p, s in info.items():
        if p.startswith(("logs/zfy_v5/3-bert/", "logs/zfy_v5/4-cnhubert/",
                         "logs/zfy_v5/5-wav32k/")):
            feat_total += 1
            if s < 1024:
                feat_ptr += 1
    print(f"  特征文件: {feat_total} 个，其中 LFS 指针 {feat_ptr} 个")
    if feat_ptr:
        print()
        print("  => 特征文件也是 LFS 指针！这就是 'load audio or ssl error' 的根因。")
        print("     训练读到指针文本 -> 异常 -> 被 except 吞掉 -> 返回全零张量。")
    else:
        print()
        print("  => 特征文件不是指针。那问题在别处（如 ffmpeg 二进制缺失）。")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
