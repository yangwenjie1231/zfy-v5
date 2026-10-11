# -*- coding: utf-8 -*-
"""用官方 SDK 修复 LFS 文件（替代不可靠的 git lfs pull）。

## 为什么不用 git lfs pull

云上 git-lfs 有两个问题：
  · `git lfs pull` 静默无输出、什么都没做（它以为对象已在缓存）
  · 下到一半截断：s2Gv5dev.pth 只有 701,571,072 / 766,385,517 字节

而 `modelscope.snapshot_download` **实测可用**（本地已验证）：
  · s2.json 1671 字节，sha256 与 LFS 指针声明一致
  · 5-wav32k/*.wav 206,764 字节，RIFF 头正确，sha256 一致

## 设计

  1. 从 git 读所有 LFS 文件的指针（oid + size）作为**校验依据**
  2. 找出工作区里不完整的（指针 / 截断 / 缺失）
  3. 用 SDK 按需下载这些文件（allow_patterns 支持列表）
  4. 逐个校验 sha256，不符则报告
  5. 支持分批（避免一次请求几千个文件）

用法（云上，检出根）：
    python _tools/sdk_repair_lfs.py --check          # 只检查
    python _tools/sdk_repair_lfs.py --fix            # 检查并修复
    python _tools/sdk_repair_lfs.py --fix --group logs   # 只修 logs/
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RID = "yangwenjie1231/zfy-v5-fullft-pkg"
PTR_RE = re.compile(rb"oid sha256:([0-9a-f]{64})\s*\n\s*size (\d+)")


def find_root() -> Path:
    here = Path.cwd()
    for c in (here, *here.parents, Path("/mnt/workspace/zfy-v5-fullft-pkg")):
        if (c / ".git").is_dir() and (c / "GPT_SoVITS").is_dir():
            return c
    return Path("/mnt/workspace/zfy-v5-fullft-pkg")


def lfs_files(repo: Path) -> list[str]:
    """LFS 跟踪的文件（用 check-attr，不依赖 git-lfs 输出格式）。"""
    r = subprocess.run(["git", "ls-files", "-z"], cwd=str(repo),
                       capture_output=True, timeout=600)
    r2 = subprocess.run(["git", "check-attr", "--stdin", "-z", "filter"],
                        cwd=str(repo), input=r.stdout or b"",
                        capture_output=True, timeout=900)
    parts = (r2.stdout or b"").decode("utf-8", "replace").split("\0")
    out = []
    for i in range(0, len(parts) - 2, 3):
        path, _a, val = parts[i], parts[i + 1], parts[i + 2]
        if val == "lfs" and path:
            out.append(path)
    return out


def declared(repo: Path, paths: list[str]) -> dict[str, tuple[str, int]]:
    """批量读指针 -> {path: (oid, size)}。"""
    if not paths:
        return {}
    inp = "".join(f"HEAD:{p}\n" for p in paths).encode()
    r = subprocess.run(["git", "cat-file", "--batch"], cwd=str(repo),
                       input=inp, capture_output=True, timeout=1800)
    blob = r.stdout or b""
    out: dict[str, tuple[str, int]] = {}
    pos = idx = 0
    while pos < len(blob) and idx < len(paths):
        nl = blob.find(b"\n", pos)
        if nl < 0:
            break
        parts = blob[pos:nl].split()
        if len(parts) < 3 or parts[1] != b"blob":
            pos = nl + 1
            idx += 1
            continue
        try:
            size = int(parts[2])
        except ValueError:
            pos = nl + 1
            idx += 1
            continue
        content = blob[nl + 1: nl + 1 + size]
        pos = nl + 1 + size + 1
        m = PTR_RE.search(content)
        if m:
            out[paths[idx]] = (m.group(1).decode(), int(m.group(2)))
        idx += 1
    return out


def sha256_of(p: Path, chunk=8 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--group", default=None,
                    help="只处理某个前缀，如 logs/ 或 GPT_SoVITS/")
    ap.add_argument("--batch", type=int, default=400,
                    help="每次请求的文件数上限")
    ap.add_argument("--verify-hash", action="store_true", default=True,
                    help="对已存在的文件算 sha256（较慢但更严）")
    ap.add_argument("--no-verify-hash", dest="verify_hash",
                    action="store_false")
    args = ap.parse_args()

    repo = Path(args.root) if args.root else find_root()
    print("=" * 78)
    print("SDK 修复 LFS 文件" + ("（含修复）" if args.fix else "（只检查）"))
    print("=" * 78)
    print(f"检出根: {repo}")
    print(f"数据集: {RID}\n")

    paths = lfs_files(repo)
    print(f"LFS 跟踪文件: {len(paths)} 个")
    if args.group:
        paths = [p for p in paths if p.startswith(args.group)]
        print(f"  过滤 {args.group} -> {len(paths)} 个")
    if not paths:
        print("  无可处理文件")
        return 1

    print("  读取指针声明 …", flush=True)
    decl = declared(repo, paths)
    print(f"  取得声明: {len(decl)} 个")
    if not decl:
        print("  !! 读不到指针，无法判定")
        return 1

    # 找出不完整的
    bad: list[tuple[str, str, int, int]] = []
    ok = 0
    for p in paths:
        if p not in decl:
            continue
        oid, exp = decl[p]
        f = repo / p
        if not f.exists():
            bad.append((p, "缺失", 0, exp))
            continue
        sz = f.stat().st_size
        if sz < 1024 and f.read_bytes()[:40].startswith(b"version https://git-lfs"):
            bad.append((p, "指针", sz, exp))
        elif sz != exp:
            bad.append((p, "截断", sz, exp))
        else:
            ok += 1

    print(f"\n  OK {ok} / {len(decl)}")
    if bad:
        from collections import Counter
        print(f"  问题分布: {dict(Counter(b[1] for b in bad))}")
        tot_missing = sum(b[3] - b[2] for b in bad if b[1] != "缺失")
        print(f"  需补下载约 {tot_missing/1e9:.2f} GB")

    if not bad:
        print("\n  ✓ 全部完整")
        return 0

    if not args.fix:
        print("\n  执行修复：python _tools/sdk_repair_lfs.py --fix")
        return 1

    print()
    print("=" * 78)
    print("用 SDK 下载缺失文件")
    print("=" * 78)
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("  modelscope 未安装：pip install modelscope")
        return 1

    todo = [b[0] for b in bad]
    print(f"  待下载 {len(todo)} 个文件，分 {args.batch} 个一批")
    staging = repo / "_sdk_download"
    staging.mkdir(exist_ok=True)

    n_batch = (len(todo) + args.batch - 1) // args.batch
    for bi in range(n_batch):
        chunk = todo[bi * args.batch: (bi + 1) * args.batch]
        print(f"\n  --- 批次 {bi+1}/{n_batch}（{len(chunk)} 个）---", flush=True)
        try:
            snapshot_download(RID, repo_type="dataset",
                              allow_patterns=chunk,
                              local_dir=str(staging))
        except Exception as e:  # noqa: BLE001
            print(f"    批次失败: {type(e).__name__}: {str(e)[:200]}")
            continue

        # 搬回工作区
        moved = 0
        for rel in chunk:
            src = staging / rel
            if not src.exists():
                continue
            dst = repo / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            moved += 1
        print(f"    搬回 {moved} 个")

    print()
    print("=" * 78)
    print("校验（sha256）")
    print("=" * 78)
    still = []
    n_ok = 0
    for rel, why, _got, exp in bad:
        f = repo / rel
        if not f.exists():
            still.append((rel, "仍缺失"))
            continue
        oid, size = decl[rel]
        if f.stat().st_size != size:
            still.append((rel, f"体积仍不符 {f.stat().st_size}/{size}"))
            continue
        if args.verify_hash:
            got = sha256_of(f)
            if got != oid:
                still.append((rel, f"sha256 不符 {got[:12]}"))
                continue
        n_ok += 1

    print(f"  修复成功 {n_ok} / {len(bad)}")
    if still:
        print(f"  仍有问题 {len(still)} 个：")
        for rel, why in still[:8]:
            print(f"    [{why}] {rel}")

    # 清理 staging
    shutil.rmtree(staging, ignore_errors=True)

    print()
    print("=" * 78)
    if not still:
        print("✓ 全部修复完成")
        print("=" * 78)
        print("""
  下一步：**重跑冒烟测试**，并确认输出里**没有**
  'load audio or ssl error!!!!!!'

  那行报错 = 数据加载失败被 except 静默替换成全零张量，
  训练会"成功"但学的是垃圾。
""")
        return 0
    print(f"仍有 {len(still)} 个文件有问题，请把输出贴出来")
    return 1


if __name__ == "__main__":
    sys.exit(main())