# -*- coding: utf-8 -*-
"""紧急恢复：撤销 push_single.py 误删整个仓库的提交。

## 事故

`push_single.py` 用 `git clone --no-checkout` 拿到的索引是**空的**。
之后 `git add <单个文件>` + `git commit`，提交出的树**只含那一个文件**，
把其余 2696 个文件全部删除。

远端因此从 2697 个文件变成 1 个。

## 恢复原理

上一个提交（本脚本会动态定位）是完整的。`git revert <坏提交>` 会生成
一个反向提交，把树恢复到父提交的状态 —— 不需要 LFS 内容，
因为树里引用的只是 **134 字节的指针**（真实内容仍在 LFS 存储里）。

## 为什么用 revert 而不是 force push

ModelScope 可能拒绝 force push；revert 是正常提交，历史保留。

用法：
    python recover_repo.py --dry-run
    python recover_repo.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NS, REPO = "yangwenjie1231", "zfy-v5-fullft-pkg"
EXPECT_FILES = 2697          # 出事前的文件数


def token_url(with_token=True) -> str:
    if not with_token:
        return f"https://www.modelscope.cn/datasets/{NS}/{REPO}.git"
    p = Path.home() / ".modelscope" / "credentials" / "git_token"
    t = p.read_text(encoding="utf-8").strip() if p.exists() else ""
    return (f"https://oauth2:{t}@www.modelscope.cn/datasets/{NS}/{REPO}.git"
            if t else token_url(False))


def count_files(repo: Path, rev: str = "HEAD") -> int:
    r = subprocess.run(["git", "ls-tree", "-r", "--name-only", rev],
                       cwd=str(repo), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return len([l for l in (r.stdout or "").split("\n") if l.strip()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="ms_recover_"))
    try:
        env = os.environ.copy()
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        print("=" * 76)
        print("1. 克隆（取最近若干提交，不拉 LFS 内容）")
        print("=" * 76)
        # 不用 --filter=blob:none：需要指针 blob 才能 revert
        r = subprocess.run(
            ["git", "clone", "--depth", "5", token_url(), "r"],
            cwd=str(tmp), capture_output=True, env=env, timeout=1800)
        print(f"   exit={r.returncode}")
        if r.returncode != 0:
            print("  " + r.stderr.decode("utf-8", "replace")[-500:])
            return 1
        repo = tmp / "r"

        def git(*a, timeout=1800):
            return subprocess.run(["git", *a], cwd=str(repo),
                                  capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=timeout)

        print()
        print("=" * 76)
        print("2. 历史与各提交的文件数")
        print("=" * 76)
        log = git("log", "--oneline", "-8")
        print(log.stdout)
        for line in (log.stdout or "").strip().split("\n"):
            if not line.strip():
                continue
            sha = line.split()[0]
            n = count_files(repo, sha)
            mark = "  <-- 完整" if n > 1000 else ("  <-- 破坏" if n < 100 else "")
            print(f"    {sha}  {n:5d} 个文件{mark}")

        head_n = count_files(repo, "HEAD")
        print(f"\n  HEAD 文件数: {head_n}")

        # 找最近的"完整"提交
        good = None
        for line in (git("log", "--format=%H", "-10").stdout or "").strip().split("\n"):
            if line.strip() and count_files(repo, line.strip()) > 1000:
                good = line.strip()
                break
        if good is None:
            print("  !! 找不到完整提交，需人工介入")
            return 1
        print(f"  最近的完整提交: {good[:10]}  ({count_files(repo, good)} 个文件)")

        if head_n > 1000:
            print("\n  HEAD 已是完整的，无需恢复")
            return 0

        print()
        print("=" * 76)
        print("3. revert 破坏性提交（把树恢复到完整状态）")
        print("=" * 76)
        if args.dry_run:
            print("  (dry-run，未执行)")
            return 0

        # 用 reset --mixed 到完整提交，再一次性提交（等价于 revert，更直接）
        r = git("reset", "--soft", good)
        if r.returncode != 0:
            print(f"  reset 失败: {r.stderr[-300:]}")
            return 1
        print("  已 reset --soft 到完整提交")

        st = git("status", "--porcelain")
        n_chg = len([l for l in (st.stdout or "").split("\n") if l.strip()])
        print(f"  待提交变更: {n_chg} 项")

        git("config", "user.name", "yangwenjie1231")
        git("config", "user.email", "yangwenjie1231@163.com")
        c = git("commit", "-m",
                "revert: restore full tree (push_single.py deleted all files "
                "due to empty index in --no-checkout clone)")
        if c.returncode != 0:
            print(f"  commit 失败: {(c.stdout or '')[-300:]}{(c.stderr or '')[-300:]}")
            return 1
        print("  已提交")

        n_after = count_files(repo, "HEAD")
        print(f"  新 HEAD 文件数: {n_after}")
        if n_after <= 1000:
            print("  !! 恢复后文件数仍不对，中止推送")
            return 1

        print()
        print("=" * 76)
        print("4. 推送")
        print("=" * 76)
        p = git("push", "origin", "HEAD:master")
        out = (p.stdout or "") + (p.stderr or "")
        if p.returncode != 0:
            print(f"  推送失败:\n{out[-600:]}")
            return 1
        print("  推送完成 ✅")
        for line in out.strip().split("\n")[-4:]:
            if line.strip():
                print(f"    {line.strip()[:110]}")

        print()
        print("=" * 76)
        print("5. 复验远端")
        print("=" * 76)
        git("fetch", "origin", "master")
        n_remote = count_files(repo, "FETCH_HEAD")
        print(f"  远端 master 文件数: {n_remote}  (期望 {EXPECT_FILES})")
        print("  " + ("✓ 已恢复" if n_remote > 1000 else "✗ 仍不对"))
        return 0 if n_remote > 1000 else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
