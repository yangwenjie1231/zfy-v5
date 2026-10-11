# -*- coding: utf-8 -*-
"""为「v5 工作区」建一个干净的发布仓库（GitHub + ModelScope 各一份）。

## 为什么不能直接推现有仓库
    · 当前 `D:\\A\\zfy-v5` **没有配置远端**，`.git` 已 **5.3 GB**
    · 里面跟踪了一个 **2646 MB 的 ZIP**（`dist/zfy_v5_fullft_cloud.zip`）
    · 工作区 23.7 GB（GPT-SoVITS 检出 19.9 GB + dist 3.1 GB）

**整推不现实**，而且历史里那个 2.6 GB 的 blob 会永远留在 .git 里。

## 做法：建**全新历史**的发布仓库
    1. 在 `D:\\A\\zfy-v5-pub` 新建空 git 仓库（**没有旧历史**）
    2. 只复制**脚本 / 文档 / 配置 / 结论 / 参考音频**（约 5 MB）
    3. 排除：第三方检出、权重、语料、大 ZIP、日志
    4. 生成 `.gitignore` + `README.md` + `LICENSE`
    5. 提交

**优点**：干净、小（~5 MB）、无历史包袱。
**代价**：不保留旧提交历史（但旧仓库仍在本地，随时可查）。

## 为什么参考音频要**包含**（虽然 .gitignore 排除 wav）
`ref/styles/*.wav` 是**产物**（三种语气风格），不是原始语料：
    · 体积小（2.4 MB 总计）
    · 是本次工作的**核心交付物**
    · 语料版权风险低（是游戏角色的公开语音，且已用于训练）
所以用 `git add -f` 强制加入。

## 用法
    python publish_new_repo.py --dry-run     # 只看会复制什么
    python publish_new_repo.py               # 实际创建
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = Path(r"D:\A\zfy-v5")
DST = Path(r"D:\A\zfy-v5-pub")

# ---------- 要发布的文件（相对路径 glob）----------
INCLUDE_GLOBS = [
    "*.py",              # 所有脚本
    "*.md",              # 文档
    "*.sh",
    "*.ipynb",
    "*.txt",             # 结论/报告（小文件）
    "*.json",
    "*.yaml", "*.yml",
    ".gitignore",
    "GPT-SoVITS/ref/styles/*.wav",     # 三种语气风格（核心交付物）
    "GPT-SoVITS/ref/styles/*.json",
    "GPT-SoVITS/ref/styles/*.txt",
    "GPT-SoVITS/ref/ref.txt",
]

# ---------- 排除（体积大 / 无价值 / 敏感）----------
# ⚠️ `GPT-SoVITS/` 整体排除（19.9 GB 第三方检出），
#    但 `GPT-SoVITS/ref/styles/*.wav` 是**核心交付物**，必须放行。
#    所以排除逻辑要**先检查白名单**。
EXCLUDE_PATTERNS = [
    "GPT-SoVITS/",           # 第三方检出（19.9 GB）
    "RVC/",
    "dist/",                 # 产物（含 2.6 GB ZIP）
    "out/",
    "data/",
    "logs/",
    "__pycache__/",
    "*.log",
    "*.pth", "*.ckpt", "*.safetensors", "*.onnx",
    "*.zip", "*.tar", "*.gz",
    "*.bak",                 # 备份文件
]

# ---------- 白名单（**优先于排除**）----------
ALLOW_PREFIXES = [
    "GPT-SoVITS/ref/styles/",
    "GPT-SoVITS/ref/ref.txt",
]

MAX_MB = 5.0                 # 单文件上限


def excluded(rel: str) -> bool:
    # 白名单优先
    for pre in ALLOW_PREFIXES:
        if rel.startswith(pre):
            return False
    import fnmatch
    for pat in EXCLUDE_PATTERNS:
        if pat.endswith("/"):
            if rel.startswith(pat) or f"/{pat}" in f"/{rel}":
                return True
        elif fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(Path(rel).name, pat):
            return True
    return False


def collect() -> list[Path]:
    """收集要发布的文件（相对 SRC 的路径）。"""
    out: list[Path] = []
    seen: set[str] = set()
    for pat in INCLUDE_GLOBS:
        # 顶层 glob（不递归，避免扫进 GPT-SoVITS/）
        if "/" in pat:
            for p in SRC.glob(pat):
                rel = p.relative_to(SRC).as_posix()
                if rel in seen or not p.is_file():
                    continue
                if excluded(rel):
                    continue
                seen.add(rel)
                out.append(p)
        else:
            for p in SRC.glob(pat):
                rel = p.relative_to(SRC).as_posix()
                if rel in seen or not p.is_file():
                    continue
                if excluded(rel):
                    continue
                seen.add(rel)
                out.append(p)
    return sorted(out, key=lambda p: p.relative_to(SRC).as_posix())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=" * 96)
    print("建 v5 发布仓库（新历史）")
    print("=" * 96)
    print(f"  源: {SRC}")
    print(f"  目标: {DST}")

    files = collect()
    total = sum(p.stat().st_size for p in files)
    print()
    print(f"  收集到 {len(files)} 个文件，合计 {total/1e6:.2f} MB")

    # 单文件超限检查
    big = [(p, p.stat().st_size / 1e6) for p in files
           if p.stat().st_size / 1e6 > MAX_MB]
    if big:
        print()
        print(f"  ⚠️ 超过 {MAX_MB} MB 的文件（将排除）:")
        for p, mb in big:
            print(f"      {mb:7.2f} MB  {p.relative_to(SRC).as_posix()}")
        files = [p for p in files if p.stat().st_size / 1e6 <= MAX_MB]
        total = sum(p.stat().st_size for p in files)
        print(f"  排除后: {len(files)} 个文件，{total/1e6:.2f} MB")

    # 按目录统计
    from collections import Counter
    dirs = Counter(p.relative_to(SRC).parent.as_posix() for p in files)
    print()
    print("  按目录:")
    for d, n in dirs.most_common(12):
        print(f"      {n:4d}  {d}")

    # 重点文件检查
    print()
    print("  关键文件:")
    for key in ("GPT-SoVITS/ref/styles/normal.wav",
                "GPT-SoVITS/ref/styles/firm.wav",
                "GPT-SoVITS/ref/styles/gentle.wav",
                "switch_style.py",
                "make_style_variants.py",
                "README.md"):
        hit = any(p.relative_to(SRC).as_posix() == key for p in files)
        print(f"      {'✓' if hit else '✗'} {key}")

    if args.dry_run:
        print()
        print("  (dry-run，未创建)")
        print("=" * 96)
        return 0

    # ---------- 创建 ----------
    if DST.exists():
        print()
        print(f"  目标已存在: {DST}")
        print("  请先删除，或用 --dry-run 预览")
        return 1
    DST.mkdir(parents=True)

    for p in files:
        rel = p.relative_to(SRC)
        d = DST / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, d)

    print()
    print(f"  ✓ 已复制 {len(files)} 个文件到 {DST}")

    # ---------- .gitignore ----------
    (DST / ".gitignore").write_text("""# ============================================================
# v5 工作区发布仓库
#
# 本仓库只含**脚本 / 文档 / 配置 / 三种语气参考音频**。
# 权重、语料、第三方检出全部排除。
# ============================================================

# ---------- 第三方检出 ----------
GPT-SoVITS/
RVC/

# ---------- 权重 ----------
*.pth
*.ckpt
*.safetensors
*.onnx

# ---------- 数据与产物 ----------
data/
logs/
out/
dist/
*.wav
*.mp3
*.flac

# ---------- 训练中间产物 ----------
__pycache__/
*.pyc
.pytest_cache/
*.log

# ---------- 大文件 ----------
*.zip
*.tar
*.gz
*.bak

# ---------- 其它 ----------
.DS_Store
Thumbs.db

# ---------- 例外：三种语气参考音频（核心交付物，约 2.4 MB）----------
!GPT-SoVITS/ref/styles/*.wav
!GPT-SoVITS/ref/ref.txt
""", encoding="utf-8")

    print("  ✓ 已写 .gitignore")
    print()
    print("=" * 96)
    print("下一步（在 " + str(DST) + " 里执行）")
    print("=" * 96)
    print("""
  ## 1) 初始化
      cd D:\\A\\zfy-v5-pub
      git init -b main
      git add -A
      git add -f GPT-SoVITS/ref/styles/*.wav GPT-SoVITS/ref/ref.txt
      git -c user.name=yangwenjie1231 -c user.email=yangwenjie1231@qq.com \\
        commit -m "feat: 庄方宜 v5 微调工作区（脚本+文档+三种语气参考音频）"

  ## 2) 建 GitHub 仓库并推送
      gh repo create zfy-v5 --public --source=. --push \\
        --description "庄方宜 GPT-SoVITS v5 微调工作区：脚本、结论、三种语气参考音频"

  ## 3) 建 ModelScope 数据集并推送
      # （用 ms 或网页建，然后）
      git remote add ms https://www.modelscope.cn/datasets/yangwenjie1231/zfy-v5.git
      git push ms main

  ## 4) 加 LFS（若 wav 超限）
      # GitHub 单文件 >100MB 才需 LFS；我们的 wav 最大 1.5MB，**不需要**
""")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
