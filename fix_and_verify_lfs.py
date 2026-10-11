# -*- coding: utf-8 -*-
"""云端 LFS 全面校验与修复（离线可查 + 一键修）。

## 为什么需要

冒烟训练的失败路径：

    data_utils.py:122  except:
                           spec = torch.zeros(1025, 100)   # 全零
                           wav  = torch.zeros(1, 100*hop)
                           ssl  = torch.zeros(1, 768, 100)
                           print("load audio or ssl error!!!!!!", ...)

异常被**静默吞掉**，喂给模型的是全零张量。
所以「训练成功」是假象 —— 数据是垃圾。

## 关键认知

我之前只校验了 5 个模型权重文件（`.pth`/`.ckpt`/`.bin`），
**漏掉了特征文件**（`4-cnhubert/*.pt`、`5-wav32k/*.wav`、
`2-name2text.txt`）。这正是报错的来源。

## 本脚本

  --check   （默认）**离线**核对每个 LFS 文件的体积与指针声明
             指针就在 git 里（`git show HEAD:<path>`），无需联网
  --fix     清掉残缺的 LFS 缓存 -> 重新 fetch -> checkout -> 再校验

判据三态：
  · 工作区文件 < 1KB 且以 "version https://git-lfs" 开头 -> **未拉取**
  · 体积 != 指针声明的 size                                -> **截断**
  · 两者相符                                               -> OK

用法（云上，检出根）：
    python _tools/fix_and_verify_lfs.py            # 只检查
    python _tools/fix_and_verify_lfs.py --fix      # 检查并修复
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PTR_RE = re.compile(rb"oid sha256:([0-9a-f]{64})\s*\n\s*size (\d+)")


def find_root() -> Path:
    here = Path.cwd()
    for c in (here, *here.parents,
              Path("/mnt/workspace/zfy-v5-fullft-pkg")):
        if (c / ".git").is_dir() and (c / "GPT_SoVITS").is_dir():
            return c
    return Path("/mnt/workspace/zfy-v5-fullft-pkg")


def git(args, cwd, timeout=1800):
    r = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                       timeout=timeout)
    return (r.returncode,
            (r.stdout or b"").decode("utf-8", "replace"),
            (r.stderr or b"").decode("utf-8", "replace"))


def lfs_paths(repo: Path) -> list[str]:
    """所有 LFS 跟踪的文件路径。

    ⚠️ `git lfs ls-files` 每行格式是 `<oid> <marker> <path>`，
    例如 `6f93d3cd01 - GPT_SoVITS/configs/s2.json` ——
    **不能整行当路径**（我第一版就栽在这，导致全部 missing、
    结果假阳性报"全部完整"）。

    最稳的办法是 `git check-attr`：直接问 git 每个文件的 filter 属性，
    不依赖 git-lfs 的输出格式。
    """
    r = subprocess.run(["git", "ls-files", "-z"], cwd=str(repo),
                       capture_output=True, timeout=600)
    if r.returncode != 0:
        return []
    listed = r.stdout
    r2 = subprocess.run(["git", "check-attr", "--stdin", "-z", "filter"],
                        cwd=str(repo), input=listed, capture_output=True,
                        timeout=900)
    parts = (r2.stdout or b"").decode("utf-8", "replace").split("\0")
    out = []
    for i in range(0, len(parts) - 2, 3):
        path, _attr, val = parts[i], parts[i + 1], parts[i + 2]
        if val == "lfs" and path:
            out.append(path)
    return out


def declared_sizes(repo: Path, paths: list[str]) -> dict[str, tuple[str, int]]:
    """批量读指针，返回 {path: (oid, size)}。

    用 `git cat-file --batch` 一次取回全部 blob，避免 2500 次进程启动。
    输出格式（每条）：
        <sha> blob <size>\n
        <size 字节的内容>\n
    """
    if not paths:
        return {}
    inp = "".join(f"HEAD:{p}\n" for p in paths).encode()
    r = subprocess.run(["git", "cat-file", "--batch"], cwd=str(repo),
                       input=inp, capture_output=True, timeout=1800)
    blob = r.stdout or b""

    out: dict[str, tuple[str, int]] = {}
    pos = 0
    idx = 0
    n = len(blob)
    while pos < n and idx < len(paths):
        nl = blob.find(b"\n", pos)
        if nl < 0:
            break
        header = blob[pos:nl]
        parts = header.split()
        # 缺失对象：header 形如 "HEAD:path missing"（只有 2 段）
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


def check(repo: Path) -> list[tuple[str, str, int, int]]:
    """返回有问题的 [(path, 问题, 实际, 期望)]。"""
    paths = lfs_paths(repo)
    print(f"  LFS 跟踪文件: {len(paths)} 个")
    if not paths:
        return []

    print("  读取指针声明（离线，批量）…", flush=True)
    decl = declared_sizes(repo, paths)
    print(f"  取得声明体积: {len(decl)} 个")

    # 安全阀：一个都没解析出来时绝不能报"全部完整"
    # （我第一版就因此假阳性 —— 路径格式解析错了，
    #   0 个声明 -> 循环 0 次 -> 报"全部完整"，而实际全是指针）
    if not decl:
        print()
        print("  !! 无法从 git 读取任何指针声明 —— 不能判定完整性")
        print("     这通常说明文件列表解析有问题，或仓库状态异常。")
        print("     请把本输出贴出来，不要据此认为『文件完整』。")
        return [("<无法读取指针>", "检查失败", 0, 0)]

    bad: list[tuple[str, str, int, int]] = []
    n_ok = 0
    for p in paths:
        f = repo / p
        if p not in decl:
            continue
        oid, exp = decl[p]
        if not f.exists():
            bad.append((p, "文件缺失", 0, exp))
            continue
        sz = f.stat().st_size
        if sz < 1024 and f.read_bytes()[:40].startswith(b"version https://git-lfs"):
            bad.append((p, "未拉取（指针）", sz, exp))
        elif sz != exp:
            bad.append((p, "截断", sz, exp))
        else:
            n_ok += 1

    print(f"  OK {n_ok} / 共 {len(decl)}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="清除缓存并重新拉取")
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    repo = Path(args.root) if args.root else find_root()
    print("=" * 78)
    print("LFS 全面校验" + ("（含修复）" if args.fix else ""))
    print("=" * 78)
    print(f"检出根: {repo}\n")

    if not (repo / ".git").is_dir():
        print("  不是 git 仓库")
        return 1

    code, out, _ = git(["rev-parse", "--short", "HEAD"], repo)
    print(f"提交: {out.strip()}\n")

    # .gitattributes 概览
    ga = repo / ".gitattributes"
    if ga.exists():
        txt = ga.read_text(encoding="utf-8", errors="replace")
        pats = [l.split()[0] for l in txt.split("\n")
                if "filter=lfs" in l and l.strip()]
        print(f"LFS 规则 {len(pats)} 条: {', '.join(pats[:12])}"
              + (" …" if len(pats) > 12 else ""))
    print()

    print("--- 修复前 ---")
    bad = check(repo)

    if not bad:
        print("\n  ✓ 全部 LFS 文件完整")
        return 0

    # 分类统计
    from collections import Counter
    kinds = Counter(b[1] for b in bad)
    print(f"\n  问题分布: {dict(kinds)}")
    print("  举例（最多 6 个）:")
    for p, why, got, exp in bad[:6]:
        print(f"    [{why}] {p}")
        print(f"        实际 {got:>12,d}  期望 {exp:>12,d}")

    if not args.fix:
        print("""
  执行修复：
      python _tools/fix_and_verify_lfs.py --fix
""")
        return 1

    print()
    print("=" * 78)
    print("开始修复")
    print("=" * 78)

    # 1) 清掉残缺的 LFS 缓存（**关键**：否则 pull 会跳过已"存在"的对象）
    for d in (repo / ".git" / "lfs" / "objects",
              repo / ".git" / "lfs" / "incomplete"):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            print(f"  已清除 {d.relative_to(repo)}")

    code, out, err = git(["lfs", "install", "--local"], repo)
    print(f"  git lfs install --local -> {code}")

    # 2) 删掉工作区里的坏文件，强制重新检出
    n_del = 0
    for p, why, got, exp in bad:
        f = repo / p
        if f.exists():
            try:
                f.unlink()
                n_del += 1
            except OSError:
                pass
    print(f"  已删除 {n_del} 个坏文件")

    # 3) 重新拉取
    print("  git lfs fetch --all …（大文件，需耐心）", flush=True)
    code, out, err = git(["lfs", "fetch", "--all"], repo)
    print(f"    -> {code}")
    for line in (out + err).strip().split("\n")[-6:]:
        if line.strip():
            print(f"       {line.strip()[:110]}")

    print("  git lfs checkout …", flush=True)
    code, out, err = git(["lfs", "checkout"], repo)
    print(f"    -> {code}")
    for line in (out + err).strip().split("\n")[-4:]:
        if line.strip():
            print(f"       {line.strip()[:110]}")

    print("  git checkout --force …", flush=True)
    code, out, err = git(["checkout", "--force", "--", "."], repo,
                         timeout=1800)
    print(f"    -> {code}")

    print()
    print("=" * 78)
    print("修复后复检")
    print("=" * 78)
    bad2 = check(repo)

    print()
    print("=" * 78)
    if not bad2:
        print("✓ 全部 LFS 文件已完整")
        print("=" * 78)
        print("""
  下一步：**重跑冒烟测试**，并确认输出里**没有**
  'load audio or ssl error!!!!!!'

  那行报错意味着数据加载失败后被 except 静默替换成全零张量 ——
  训练照样"成功"，但学的是垃圾。
""")
        return 0

    print(f"仍有 {len(bad2)} 个文件有问题")
    print("=" * 78)
    print("""
  缓存清理 + 重拉没能修好，很可能是**远程对象本身有问题**。

  最可靠的办法：删掉重新 clone（**不要**加 GIT_LFS_SKIP_SMUDGE）

      cd /mnt/workspace
      mv zfy-v5-fullft-pkg zfy-v5-old
      git lfs install
      git clone https://oauth2:<token>@www.modelscope.cn/datasets/\\
yangwenjie1231/zfy-v5-fullft-pkg.git zfy-v5-fullft-pkg

      cd zfy-v5-fullft-pkg
      python _tools/fix_and_verify_lfs.py      # 再校验
""")
    return 1


if __name__ == "__main__":
    sys.exit(main())
