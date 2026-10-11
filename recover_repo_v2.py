# -*- coding: utf-8 -*-
"""紧急恢复 v2：撤销 push_single.py 误删整个仓库的提交。

## 事故
`push_single.py` 用 `git clone --no-checkout` 时索引是空的，
`git add <单文件>` + `git commit` 提交出的树**只含那一个文件**，
把其余 2696 个文件全删了。

远端：9ce064b 只剩 1 个文件；父提交 6d69f50 有 2697 个（完整）。

## v1 为什么失败
用了 `git reset --soft <good>` —— **--soft 只移动 HEAD，不动索引**，
索引里仍是那 1 个文件，commit 出来还是 1 个。

## v2 做法（两种，按顺序试，**推送前必须验证**）

  A. `git revert --no-edit HEAD`
     反向应用坏提交的 diff，从父提交的树恢复被删文件。

  B. `git read-tree <good>` + `git commit`
     直接把索引替换成完整提交的树，再提交。
     （不依赖工作区内容 —— 树里引用的只是 134 字节的 LFS 指针）

两法都在**本地验证 HEAD 文件数 > 1000 后**才推送。

用法：
    python recover_repo_v2.py --dry-run
    python recover_repo_v2.py
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
MIN_OK = 1000


def url(with_token=True) -> str:
    if not with_token:
        return f"https://www.modelscope.cn/datasets/{NS}/{REPO}.git"
    p = Path.home() / ".modelscope" / "credentials" / "git_token"
    t = p.read_text(encoding="utf-8").strip() if p.exists() else ""
    return (f"https://oauth2:{t}@www.modelscope.cn/datasets/{NS}/{REPO}.git"
            if t else url(False))


def count(repo: Path, rev="HEAD", *, use_index=False) -> int:
    args = (["git", "ls-files"] if use_index
            else ["git", "ls-tree", "-r", "--name-only", rev])
    r = subprocess.run(args, cwd=str(repo), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return len([l for l in (r.stdout or "").split("\n") if l.strip()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # ⚠️ 关键：把这个变量设成**进程级**环境变量。
    # 上一版只在 clone 时通过 env= 传入，后续 git revert / reset 又去
    # smudge 真实 LFS 内容 —— 实测下了 0.68 GB 还在继续（全量 3 GB，
    # 要一个多小时）。而我们**只需要 134 字节的指针**，不需要内容。
    os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
    # 双保险：直接把 lfs 过滤器清空，任何情况下都不去拉内容
    os.environ["GIT_LFS_SKIP_DOWNLOAD"] = "1"

    tmp = Path(tempfile.mkdtemp(prefix="ms_rec3_"))
    try:
        env = os.environ.copy()
        print("=" * 76)
        print("1. 克隆（正常 clone，工作区会检出；跳过 LFS 内容）")
        print("=" * 76)
        r = subprocess.run(["git", "clone", "--depth", "5", url(), "r"],
                           cwd=str(tmp), capture_output=True, env=env,
                           timeout=1800)
        if r.returncode != 0:
            print("  " + r.stderr.decode("utf-8", "replace")[-400:])
            return 1
        repo = tmp / "r"

        def git(*a, timeout=1800, check=False):
            # -c filter.lfs.* 置空：彻底禁用 LFS smudge，避免误下 3 GB
            base = ["git", "-c", "filter.lfs.smudge=",
                    "-c", "filter.lfs.process=",
                    "-c", "filter.lfs.required=false"]
            p = subprocess.run([*base, *a], cwd=str(repo),
                               capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=timeout, env=env)
            if check and p.returncode != 0:
                print(f"  !! git {' '.join(a)} -> {p.returncode}")
                print(f"     {(p.stdout or '')[-300:]}{(p.stderr or '')[-300:]}")
            return p

        head_n = count(repo)
        print(f"  HEAD 文件数: {head_n}")
        if head_n >= MIN_OK:
            print("  已是完整状态，无需恢复")
            return 0

        # 找最近的完整提交
        good = None
        for sha in (git("log", "--format=%H", "-10").stdout or "").split():
            if count(repo, sha) >= MIN_OK:
                good = sha
                break
        if not good:
            print("  !! 找不到完整提交")
            return 1
        print(f"  最近的完整提交: {good[:10]}  ({count(repo, good)} 个文件)")

        if args.dry_run:
            print("\n(dry-run)")
            return 0

        print()
        print("=" * 76)
        print("2. 恢复（方案 A: git revert）")
        print("=" * 76)
        git("config", "user.name", "yangwenjie1231")
        git("config", "user.email", "yangwenjie1231@163.com")
        rv = git("revert", "--no-edit", "HEAD")
        n_after = count(repo)
        print(f"  revert 后 HEAD 文件数: {n_after}")
        if n_after < MIN_OK:
            print(f"  方案 A 无效（{rv.returncode}），改用方案 B: read-tree")
            # 清掉可能的半成品状态
            git("revert", "--abort")
            git("reset", "--hard", "HEAD")
            # B: 把索引替换成完整提交的树，再提交
            rt = git("read-tree", good, check=True)
            n_idx = count(repo, use_index=True)
            print(f"  read-tree 后索引文件数: {n_idx}")
            if n_idx < MIN_OK:
                print("  !! 方案 B 也失败，需人工介入")
                return 1
            cm = git("commit", "-m",
                     "restore full tree (read-tree from %s)" % good[:10],
                     check=True)
            n_after = count(repo)
            print(f"  提交后 HEAD 文件数: {n_after}")

        if n_after < MIN_OK:
            print(f"  !! 恢复失败（{n_after} 个文件），已中止，未推送")
            return 1

        print()
        print("=" * 76)
        print("3. 工作区同步（让文件真的落盘）")
        print("=" * 76)
        git("reset", "--hard", "HEAD")
        on_disk = len([p for p in repo.rglob("*")
                       if p.is_file() and ".git" not in p.parts])
        print(f"  磁盘文件数: {on_disk}")
        print(f"  抽查关键文件:")
        for k in ("GPT_SoVITS/s2_train_v5_full.py",
                  "logs/zfy_v5/2-name2text.txt",
                  "_tools/sdk_repair_lfs.py",
                  "ref/zfh-e4.ckpt",
                  "_tools/cloud_synth_sweep.py"):
            p = repo / k
            sz = p.stat().st_size if p.exists() else -1
            print(f"    {'OK ' if p.exists() else '缺失'} {k:44s} {sz} 字节")

        print()
        print("=" * 76)
        print("4. 推送（已验证 HEAD 文件数 > 1000）")
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
        n_remote = count(repo, "FETCH_HEAD")
        print(f"  远端 master 文件数: {n_remote}")
        print("  " + ("✓ 已恢复" if n_remote >= MIN_OK else "✗ 仍不对"))
        return 0 if n_remote >= MIN_OK else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
