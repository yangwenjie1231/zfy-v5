# -*- coding: utf-8 -*-
"""轻量推送 v2：只推指定文件，**带推送前安全校验**。

## v1 的事故（严重）

v1 用 `git clone --no-checkout`。这种克隆的**索引是空的**，于是：

    git add <单个文件>   ->  索引里只有这 1 个文件
    git commit           ->  提交树只有 1 个文件
    git push             ->  **把其余 2696 个文件全删了**

远端从 2697 个文件变成 1 个。

## v2 的三道防线

  1. 用**正常 clone**（带 checkout，但跳过 LFS 内容），索引天然完整
  2. 提交后、推送前**校验提交树的文件数**，少于阈值直接中止
  3. 推送后再复验远端文件数

第 2 条是关键 —— v1 完全没有这个检查，才会把破坏性提交推上去。

用法：
    python push_single.py _tools/cloud_synth_sweep.py
    python push_single.py --dry-run _tools/a.py
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

V5 = Path(r"D:\A\zfy-v5")
NS, REPO = "yangwenjie1231", "zfy-v5-fullft-pkg"
MIN_FILES = 1000        # 提交树文件数下限；低于此值拒绝推送


def token_url() -> str:
    p = Path.home() / ".modelscope" / "credentials" / "git_token"
    t = p.read_text(encoding="utf-8").strip() if p.exists() else ""
    return (f"https://oauth2:{t}@www.modelscope.cn/datasets/{NS}/{REPO}.git"
            if t else f"https://www.modelscope.cn/datasets/{NS}/{REPO}.git")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="仓库内相对路径")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # 进程级禁用 LFS smudge —— 否则 checkout 会拉 3 GB 内容
    os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
    os.environ["GIT_LFS_SKIP_DOWNLOAD"] = "1"
    env = os.environ.copy()

    # 解析本地源路径
    pairs = []
    for rel in args.paths:
        src = (V5 / rel[len("_tools/"):]) if rel.startswith("_tools/") \
            else (V5 / "GPT-SoVITS" / rel)
        if not src.exists():
            print(f"  本地不存在: {src}")
            return 1
        pairs.append((src, rel))
        print(f"  {rel}  ({src.stat().st_size/1024:.1f} KB)")

    tmp = Path(tempfile.mkdtemp(prefix="ms_one2_"))
    try:
        print("\nclone（正常检出，跳过 LFS 内容）…", flush=True)
        r = subprocess.run(["git", "clone", "--depth", "1", token_url(), "r"],
                           cwd=str(tmp), capture_output=True, env=env,
                           timeout=1800)
        if r.returncode != 0:
            print("  " + r.stderr.decode("utf-8", "replace")[-400:])
            return 1
        repo = tmp / "r"

        def git(*a, timeout=1800):
            # ⚠️ 只禁 **smudge**（下载），**不能**禁 process！
            #
            # 踩过的坑：原先是
            #     -c filter.lfs.smudge= -c filter.lfs.process=
            #                -c filter.lfs.required=false
            # `filter.lfs.process` 是**双向**的 —— 禁掉它，`git add` 也
            # 不会再生成 LFS 指针，155 MB 的 .ckpt 被当普通 blob 提交，
            # ModelScope 直接拒收：
            #     ... 155315150 too large and requires the use of git lfs
            #     remote: Validation failed. Push rejected.
            #
            # 正确做法：
            #   · filter.lfs.smudge=   -> 只禁下载（避免 checkout 拉 3GB）
            #   · filter.lfs.process 保留 -> 让 clean 正常工作，生成指针
            #   · GIT_LFS_SKIP_SMUDGE=1（进程级，见上方 os.environ）
            base = ["git", "-c", "filter.lfs.smudge=",
                    "-c", "filter.lfs.required=false"]
            return subprocess.run([*base, *a], cwd=str(repo),
                                  capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=timeout, env=env)

        # 防线 1：确认克隆出来的仓库是完整的
        n0 = len([l for l in (git("ls-files").stdout or "").split("\n")
                  if l.strip()])
        print(f"  克隆后索引文件数: {n0}")
        if n0 < MIN_FILES:
            print(f"  !! 克隆结果异常（{n0} < {MIN_FILES}），中止")
            return 1

        # 复制要更新的文件
        for src, rel in pairs:
            dst = repo / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  已复制 {rel}")

        git("config", "user.name", "yangwenjie1231")
        git("config", "user.email", "yangwenjie1231@163.com")
        git("add", "--", *[rel for _, rel in pairs])

        # 防线 1.5：确认大文件**真的走了 LFS**（而不是被当普通 blob）
        # 踩过：filter.lfs.process 被禁 -> .ckpt 变成 155MB blob ->
        #       ModelScope 拒收 "too large and requires the use of git lfs"
        # 这里提前检查，避免白等一次 push。
        big_bad = []
        for _, rel in pairs:
            src_size = next(s.stat().st_size for s, r in pairs if r == rel)
            if src_size < 10 * 1024 * 1024:      # 小于 10MB 不关心
                continue
            # 看暂存区里的对象大小（LFS 指针只有 ~130 字节）
            blob = git("cat-file", "-s", f":{rel}")
            if blob.returncode != 0:
                continue
            staged = int((blob.stdout or "0").strip() or 0)
            if staged > 1024 * 1024:             # 暂存的是实体而非指针
                big_bad.append((rel, src_size, staged))
        if big_bad:
            print()
            print("  !! 以下大文件**没有走 LFS**（暂存的是实体 blob）：")
            for rel, sz, staged in big_bad:
                print(f"     {rel}")
                print(f"       原始 {sz/1e6:.1f} MB，暂存 {staged/1e6:.1f} MB")
            print("     原因通常是 filter.lfs.process 被禁用，或 .gitattributes")
            print("     里没有对应规则。ModelScope 会拒收 >100MB 的普通文件。")
            print("     检查：git check-attr filter -- <path>")
            return 1

        st = git("status", "--porcelain")
        changed = [l for l in (st.stdout or "").split("\n") if l.strip()]
        print(f"  变更 {len(changed)} 项")
        for l in changed[:8]:
            print(f"    {l}")
        if not changed:
            print("  无变更")
            return 0

        c = git("commit", "-m",
                "update " + ", ".join(Path(x).name for _, x in pairs))
        if c.returncode != 0:
            print(f"  commit 失败: {(c.stdout or '')[-200:]}{(c.stderr or '')[-300:]}")
            return 1
        print("  已提交")

        # 防线 2（最关键）：校验**提交树**的文件数
        n_tree = len([l for l in
                      (git("ls-tree", "-r", "--name-only", "HEAD").stdout or "")
                      .split("\n") if l.strip()])
        print(f"  提交树文件数: {n_tree}  (下限 {MIN_FILES})")
        if n_tree < MIN_FILES:
            print("  !! 提交树文件数异常 —— 这正是 v1 删库的原因，已中止推送")
            return 1

        if args.dry_run:
            print("\n(dry-run，未推送)")
            return 0

        # 推送
        print("  推送中…", flush=True)
        p = git("push", "origin", "HEAD:master")
        out = (p.stdout or "") + (p.stderr or "")
        if p.returncode != 0:
            print(f"  推送失败:\n{out[-600:]}")
            return 1
        print("  推送完成 ✅")
        for line in out.strip().split("\n")[-4:]:
            if line.strip():
                print(f"    {line.strip()[:110]}")

        # 防线 3：复验远端
        git("fetch", "origin", "master")
        n_remote = len([l for l in
                        (git("ls-tree", "-r", "--name-only", "FETCH_HEAD")
                         .stdout or "").split("\n") if l.strip()])
        print(f"  远端文件数: {n_remote}")
        if n_remote < MIN_FILES:
            print("  !! 远端文件数异常！立即运行 recover_repo_v2.py")
            return 1
        print("  ✓ 远端完整")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
