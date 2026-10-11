# -*- coding: utf-8 -*-
"""用 **git 直传** 把上云包推送到魔搭数据集（不再打包成 zip）。

## 为什么改

用户指出：魔搭数据集本身就是 **git 仓库**（带 LFS），
`ms upload` 底层也是走 git/LFS 提交。
我先前「先 zip 再上传」是多余的：
  · 多一层封装，notebook 里还要解压
  · 压缩包无法增量同步（改一个文件要重传整个 2.77GB）
  · git 直传可利用 LFS 分块与断点续传

## 做法

  1. 把工作区检出到一个临时目录（`git clone` 数据集仓）
  2. 用 `pack_for_cloud.collect()` 的文件清单**直接铺开**（不压缩）
  3. `git add` + `commit` + `push`

这样仓库里是**目录树**，notebook 里直接可用，无需解压。

用法：
    python push_via_git.py --dry-run     # 只看计划
    python push_via_git.py               # 实际推送
    python push_via_git.py --clean-remote  # 顺便删掉远程的旧 zip
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
sys.path.insert(0, str(Path(__file__).parent))
import pack_for_cloud as P  # noqa: E402

NS = "yangwenjie1231"
REPO = "zfy-v5-fullft-pkg"
GIT_URL = f"https://www.modelscope.cn/datasets/{NS}/{REPO}.git"


def run(cmd: list[str], cwd: Path, check=True, env=None):
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, cwd=str(cwd), env=e,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if check and r.returncode != 0:
        print(f"  命令失败: {' '.join(cmd)}")
        print(f"  stdout: {r.stdout[-800:]}")
        print(f"  stderr: {r.stderr[-800:]}")
        raise SystemExit(1)
    return r


def token_url() -> str:
    """带 token 的 URL（避免交互式登录）。"""
    tok_file = Path.home() / ".modelscope" / "credentials" / "git_token"
    if not tok_file.exists():
        return GIT_URL
    tok = tok_file.read_text(encoding="utf-8").strip()
    if not tok:
        return GIT_URL
    return f"https://oauth2:{tok}@www.modelscope.cn/datasets/{NS}/{REPO}.git"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--clean-remote", action="store_true",
                    help="删掉远程已有的 zip（改为目录树后不再需要）")
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    items = P.collect()
    total = sum(p.stat().st_size for p, _ in items)
    print(f"待推送: {len(items)} 文件 / {total/1e9:.2f} GB（不压缩）")

    # 体积分布
    from collections import defaultdict
    d = defaultdict(int)
    for p, rel in items:
        k = rel.replace("\\", "/").split("/")[0]
        d[k] += p.stat().st_size
    print("\n顶层分布:")
    for k, v in sorted(d.items(), key=lambda x: -x[1])[:8]:
        print(f"  {v/1e6:9.1f} MB  {k}")

    if args.dry_run:
        print("\n(dry-run)")
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="ms_push_"))
    print(f"\n临时目录: {tmp}")
    try:
        print("\n=== 1) clone 数据集仓（**不下载 LFS 内容**）===")
        # 关键：远程目前还挂着旧的 2.77GB zip（LFS 对象）。
        # 默认 clone 会把它**整个拉下来**再覆盖，纯属浪费。
        #   GIT_LFS_SKIP_SMUDGE=1  -> 不下载 LFS 内容，只取指针
        #   --filter=blob:none     -> 不取非 LFS 的历史 blob
        #   --depth 1              -> 只要最新一次提交
        run(["git", "clone", "--depth", "1", "--filter=blob:none",
             token_url(), str(tmp / "repo")],
            cwd=tmp, env={"GIT_LFS_SKIP_SMUDGE": "1"})
        repo = tmp / "repo"
        print("  OK（未下载旧 zip 内容）")

        print("\n=== 2) 清掉旧的 zip（改为目录树）===")
        for f in repo.glob("*.zip"):
            f.unlink()
            print(f"  删除 {f.name}")

        print("\n=== 3) 铺开文件（不压缩）===")
        n = 0
        for p, rel in items:
            dst = repo / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            n += 1
            if n % 500 == 0:
                print(f"  {n}/{len(items)} …", flush=True)
        print(f"  共 {n} 个文件")

        print("\n=== 4) git add / commit / push ===")
        run(["git", "config", "user.name", "yangwenjie1231"], cwd=repo)
        run(["git", "config", "user.email", "yangwenjie1231@163.com"], cwd=repo)
        run(["git", "add", "-A"], cwd=repo)
        st = run(["git", "status", "--porcelain"], cwd=repo)
        nchg = len([l for l in st.stdout.split("\n") if l.strip()])
        print(f"  变更 {nchg} 项")
        if nchg == 0:
            print("  无变更，跳过")
            return 0
        run(["git", "commit", "-m",
             "switch to plain directory tree (no zip); add torch2.12 patch"],
            cwd=repo)
        print("  已提交")

        print("  推送中（大文件走 LFS，可能较慢）…", flush=True)
        # 直接继承 stdio，这样能看到 LFS 的实时进度条
        # （capture_output 会把进度条吞掉，日志一片空白）
        r = subprocess.run(["git", "push", "origin", "HEAD:master"],
                           cwd=str(repo))
        if r.returncode != 0:
            print(f"\n  推送失败（退出码 {r.returncode}）")
            print(f"  仓库保留在: {repo}")
            return 1
        print("  推送完成")
        print(f"\n仓库: https://modelscope.cn/datasets/{NS}/{REPO}")
        print("notebook 里直接 clone 或 snapshot_download 即可，无需解压")
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"\n已清理 {tmp}")
        else:
            print(f"\n保留 {tmp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
