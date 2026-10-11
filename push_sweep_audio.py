#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把云端的合成音频推上远端（解决 non-fast-forward，且**不丢任何东西**）。

## 现状
云端 HEAD = `86958ee`（"v5 fullft sweep audio (20 wav)"）
远端 HEAD = `06041f2`（本机推的脚本修复）
两者分叉 -> `git push` 报 non-fast-forward。

云端还有 3 个未推送的提交：
    86958ee  v5 fullft sweep audio (20 wav)   <- 我们要的
    a207047  Merge branch 'master' ...
    ada8be7  v5 fullft sweep audio (10 wav)

## 关键：**绝不能用 --force**
`git push --force` 会**丢弃远端的所有脚本修复**（YAML 修复、LR 开关等）。
远端是唯一保存这些修复的地方（本机虽有，但万一）。

## 正确做法：rebase 后推送
    git fetch origin master
    git rebase origin/master        # 把云端 3 个提交重放到远端之上
    git push origin HEAD:master

rebase 时 `_tools/` 会冲突（两边都改过）—— 那些是**脚本**，
用 `--theirs`（远端版本）解决即可，因为远端是最新的。

## ⚠️ rebase 前必须确认：out_sweep 的 wav **没有被 rebase 破坏**
rebase 会重写提交，但**工作区文件不动**。所以 out_sweep 里的
20 个 wav 不会丢 —— 但为保险，本脚本先备份一份清单。

## 用法
    cd /mnt/workspace/zfy-v5-fullft-pkg
    python _tools/push_sweep_audio.py              # 只诊断
    python _tools/push_sweep_audio.py --apply      # 实际推送
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path("/mnt/workspace/zfy-v5-fullft-pkg")
if not (ROOT / ".git").is_dir():
    ROOT = Path.cwd()

_ap = argparse.ArgumentParser()
_ap.add_argument("--apply", action="store_true")
args = _ap.parse_args()


def git(*a, timeout=900):
    return subprocess.run(["git", *a], cwd=str(ROOT), capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=timeout)


def show(tag, r, n=8):
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    print(f"  [{tag}] exit={r.returncode}")
    for l in out.split("\n")[-n:]:
        if l.strip() and "git-lfs" not in l:
            print(f"    {l.strip()[:110]}")


print("=" * 80)
print(f"仓库: {ROOT}")
print("=" * 80)

# ---- 1) 先确认音频还在 ----
print()
print("1) 确认 out_sweep 音频还在（**推送的前提**）")
wavs = sorted((ROOT / "out_sweep").rglob("*.wav")) if (ROOT / "out_sweep").is_dir() else []
print(f"  out_sweep wav 数: {len(wavs)}")
dirs = sorted(d.name for d in (ROOT / "out_sweep").iterdir()
              if d.is_dir()) if (ROOT / "out_sweep").is_dir() else []
print(f"  目录: {dirs}")
if len(wavs) == 0:
    print("  !! 没有 wav —— 无需推送，先重新合成")
    sys.exit(1)
# 关键：e16_merged 在不在
m = [w for w in wavs if "merged" in str(w)]
print(f"  e16_merged wav 数: {len(m)}  {'✓' if m else '**缺**'}")

# ---- 2) 未推送的提交 ----
print()
print("2) 云端未推送的提交")
r = git("log", "--oneline", "origin/master..HEAD")
commits = [l for l in (r.stdout or "").split("\n") if l.strip()]
for c in commits:
    print(f"    {c}")
if not commits:
    print("    （无）")

# ---- 3) 分叉关系 ----
print()
print("3) 与远端的关系")
git("fetch", "origin", "master")
r = git("rev-list", "--left-right", "--count", "HEAD...origin/master")
print(f"  ahead/behind: {(r.stdout or '').strip()}")

# ---- 4) 检查远端是否有本机没有的东西 ----
print()
print("4) 安全性检查：远端提交是否都能在云端找到")
r = git("log", "--oneline", "-5", "origin/master")
print("  远端最新 5 个:")
for l in (r.stdout or "").split("\n"):
    if l.strip():
        print(f"    {l.strip()}")
r = git("merge-base", "HEAD", "origin/master")
mb = (r.stdout or "").strip()
print(f"  共同祖先: {mb[:12] if mb else '**无**（完全分叉）'}")

if not args.apply:
    print()
    print("=" * 80)
    print("（未加 --apply，只诊断）")
    print("=" * 80)
    print("""
  执行推送：
      python _tools/push_sweep_audio.py --apply

  它会：
    1. git fetch origin master
    2. git rebase origin/master      （把云端提交重放到远端之上）
       冲突时对 _tools/ 用 --theirs（取远端版本，因为远端最新）
    3. git push origin HEAD:master
    4. 复验 out_sweep 仍在

  ⚠️ 绝不用 --force —— 会丢弃远端的脚本修复（YAML/LR 等）
  ⚠️ rebase 只改提交历史，**工作区文件不动**，所以 20 个 wav 不会丢
""")
    sys.exit(0)

# ---- 实际执行 ----
print()
print("=" * 80)
print("开始推送")
print("=" * 80)

print()
print("  [1] git fetch origin master")
show("fetch", git("fetch", "origin", "master"))

print()
print("  [2] git rebase origin/master")
r = git("rebase", "origin/master")
show("rebase", r, n=12)
if r.returncode != 0:
    print()
    print("  rebase 有冲突，按策略解决：")
    # 冲突文件列表
    r2 = git("diff", "--name-only", "--diff-filter=U")
    conflicted = [l for l in (r2.stdout or "").split("\n") if l.strip()]
    print(f"    冲突文件 {len(conflicted)} 个")
    for f in conflicted[:10]:
        print(f"      {f}")
    # 全部取远端版本（--theirs），因为远端是最新的
    if conflicted:
        git("checkout", "--theirs", "--", *conflicted)
        git("add", "--", *conflicted)
        print("    已对全部冲突文件取远端版本（--theirs）")
    # 继续 rebase
    r3 = subprocess.run(
        ["git", "-c", "core.editor=true", "rebase", "--continue"],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=900)
    show("rebase --continue", r3, n=10)
    if r3.returncode != 0:
        print("    !! rebase 仍未完成，尝试 --skip 空提交")
        r4 = subprocess.run(["git", "rebase", "--skip"], cwd=str(ROOT),
                            capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=900)
        show("rebase --skip", r4, n=8)

print()
print("  [3] git push origin HEAD:master")
r = git("push", "origin", "HEAD:master", timeout=1800)
show("push", r, n=10)
pushed = r.returncode == 0
if not pushed and "git-lfs" in ((r.stderr or "") + (r.stdout or "")):
    print("    （含 git-lfs 提示，可能仍成功 —— 看上面 exit code）")

print()
print("  [4] 复验：out_sweep 仍在")
wavs2 = sorted((ROOT / "out_sweep").rglob("*.wav")) if (ROOT / "out_sweep").is_dir() else []
print(f"    out_sweep wav 数: {len(wavs2)}  "
      f"{'OK' if len(wavs2) >= len(wavs) else '**少了！**'}")

print()
print("=" * 80)
print("结果")
print("=" * 80)
r = git("log", "--oneline", "-3")
print("  当前 HEAD:")
for l in (r.stdout or "").split("\n"):
    if l.strip():
        print(f"    {l.strip()}")
if pushed:
    print()
    print("  ✓ 推送成功 —— 音频已在远端")
    print("""
  本机拉取（在**本机**执行）：
      python -c "from modelscope import snapshot_download as d; \\
d('yangwenjie1231/zfy-v5-fullft-pkg', repo_type='dataset', \\
  allow_patterns=['out_sweep/**'], local_dir=r'D:\\A\\zfy-v5\\dist\\sweep2')"
""")
else:
    print("  ✗ 推送失败 —— 把 [3] 的输出贴出来")
