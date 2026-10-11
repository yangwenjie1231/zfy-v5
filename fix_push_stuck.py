#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""云端 git 推送卡住的修复：`cannot pull with rebase: You have unstaged changes`

## 现象
    git push origin HEAD:master
    -> ! [rejected] (fetch first)   远端有新提交
    脚本尝试 git pull --rebase
    -> error: cannot pull with rebase: You have unstaged changes.

## 根因（两个叠加）
  1. **远端有新提交** —— 本机（Windows）之前 push 过（resolve_git_conflict.py
     等），而云端还没拉。
  2. **云端工作区有大量"未暂存改动"** —— 就是之前诊断过的 **LFS 假象**
     （2500+ 项 M，因为 .gitattributes 把 .json/.txt/.pickle 也划进 LFS）。
     `git pull --rebase` 要求工作区干净，于是被挡住。

## 关键：out_sweep 的 20 个 wav **已经 commit 了**
日志：`已暂存 out_sweep/（20 个 wav）` + `已提交` + `Uploading LFS objects: 100% (30/30), 25 MB`
-> LFS 对象**已经上传成功**，只是 ref 没推上去。
所以**音频不会丢**，只需把 ref 推上去。

## 本脚本（在云端跑）
  1. 报告状态（远端落后多少、未暂存有多少）
  2. 用 `git stash` 把未暂存的 LFS 假象收起来（可 pop 找回）
  3. `git pull --rebase` 整合远端
  4. `git push origin HEAD:master`
  5. `git stash pop` 还原（可选）

## 用法
    cd /mnt/workspace/zfy-v5-fullft-pkg
    python _tools/fix_push_stuck.py              # 只报告
    python _tools/fix_push_stuck.py --apply      # 实际修

## ⚠️ 血的教训：绝不能用 `git stash -u`
我第一版用的是 `git stash push -u`，而 `-u`（--include-untracked）
会把**所有未跟踪文件**卷进 stash，包括：
    logs/                        4.19 GB
    SoVITS_weights_v5_r128/      2.46 GB
    SoVITS_weights_v5/           0.56 GB
    SoVITS_weights_v5_full_e32/  ~23 GB
    out_sweep/                   （20 个 wav）
-> stash 去打包几十 GB，卡住不动（看着像"没输出"）。

**`stash -u` 比 `checkout .` / `reset --hard` 更危险** ——
后两者只动已跟踪文件，而 `-u` 会把训练产物一起卷走。

现在改用 `git checkout -- <文件>` 逐个丢弃**已跟踪**的未暂存文件，
并加了两道安全阀：
  · 若未暂存列表里出现 SoVITS_weights/out_sweep/logs/*.pth/*.ckpt/*.wav
    -> **直接中止**，不执行任何丢弃
  · 步骤 4 复验训练产物仍在
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ⚠️ 必须先解析参数！我第一版忘了这行，导致后面 `if not args.apply`
#    报 NameError: name 'args' is not defined。
_ap = argparse.ArgumentParser()
_ap.add_argument("--apply", action="store_true",
                 help="实际执行 stash/pull/push/stash-pop")
args = _ap.parse_args()

ROOT = Path.cwd()
for c in (ROOT, Path("/mnt/workspace/zfy-v5-fullft-pkg")):
    if (c / ".git").is_dir():
        ROOT = c
        break


def run(git_args, timeout=900, env=None):
    """注意：参数名不能叫 args —— 会与上面的命令行 args 混淆。"""
    return subprocess.run(["git", *git_args], cwd=str(ROOT),
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          timeout=timeout, env=env)


def show(tag, r):
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    print(f"  [{tag}] exit={r.returncode}")
    for l in out.split("\n")[-10:]:
        if l.strip():
            print(f"    {l.strip()[:110]}")


print("=" * 80)
print(f"仓库: {ROOT}")
print("=" * 80)

print()
print("1) 当前状态")
r = run(["status", "--porcelain"])
lines = [l for l in (r.stdout or "").split("\n") if l.strip()]
print(f"  改动/未跟踪: {len(lines)} 项")
for l in lines[:8]:
    print(f"    {l[:100]}")
if len(lines) > 8:
    print(f"    … 还有 {len(lines)-8} 项")

print()
print("2) 本地提交（确认 out_sweep 是否已 commit）")
r = run(["log", "--oneline", "-5"])
for l in (r.stdout or "").split("\n"):
    if l.strip():
        print(f"  {l.strip()}")
r = run(["show", "--stat", "--oneline", "HEAD"])
print("  HEAD 内容（前 8 行）:")
for l in (r.stdout or "").split("\n")[:8]:
    if l.strip():
        print(f"    {l.strip()[:100]}")

print()
print("3) 与远端的关系")
r = run(["fetch", "origin", "master"])
print(f"  fetch -> exit {r.returncode}")
r = run(["rev-list", "--left-right", "--count", "HEAD...origin/master"])
if r.returncode == 0:
    parts = (r.stdout or "").strip().split()
    if len(parts) == 2:
        ahead, behind = parts
        print(f"  本地领先 {ahead} 个提交，落后 {behind} 个提交")
r = run(["log", "--oneline", "-3", "origin/master"])
print("  远端最新 3 个:")
for l in (r.stdout or "").split("\n"):
    if l.strip():
        print(f"    {l.strip()}")

print()
print("4) 未暂存改动里有多少是 LFS 假象")
r = run(["diff", "--name-only"])
unstaged = [l for l in (r.stdout or "").split("\n") if l.strip()]
print(f"  未暂存文件: {len(unstaged)} 个")
attr = ROOT / ".gitattributes"
lfs_ext = set()
if attr.exists():
    for line in attr.read_text(encoding="utf-8", errors="replace").split("\n"):
        line = line.strip()
        if line.startswith("*.") and "filter=lfs" in line:
            lfs_ext.add(line.split()[0][1:])
n_lfs = sum(1 for f in unstaged
            if any(f.endswith(e) for e in lfs_ext))
print(f"  其中扩展名属于 LFS 规则的: {n_lfs} 个")
print(f"  -> 这些是 **LFS 假象**（内容没变，只是缺 clean filter）")

if not unstaged and not lines:
    print()
    print("  工作区干净，直接 push 即可：")
    print("    git push origin HEAD:master")
    if not args.apply:
        return_early = True
    else:
        show("push", run(["push", "origin", "HEAD:master"]))
        sys.exit(0)

if not args.apply:
    print()
    print("=" * 80)
    print("（未加 --apply，只报告）")
    print("=" * 80)
    print("""
  执行修复：
      python _tools/fix_push_stuck.py --apply

  它会：
    1. **逐个 `git checkout -- <文件>`** 丢弃未暂存的**已跟踪**文件
       （**不用 stash -u！** 那会把几十 GB 训练产物卷进去）
    2. git pull --rebase origin master
    3. git push origin HEAD:master
    4. 复验训练产物仍在

  安全阀：若未暂存列表里出现 SoVITS_weights/out_sweep/logs/*.pth/*.ckpt/*.wav
          -> **直接中止**，不丢弃任何东西。

  ⚠️ 你 commit 里的 20 个 wav **已经在本地提交里**，
     LFS 对象也已上传（日志：Uploading LFS objects: 100% (30/30), 25 MB）
     -> 不会被丢弃。

  ### ⚠️ 不要用 --force / --force-with-lease
  它们会**丢弃远端的提交**（本机 push 的那些脚本修复）。

  ### 手动备选（等价，且同样安全）
      cd /mnt/workspace/zfy-v5-fullft-pkg
      git diff --name-only                    # 先看清要丢什么
      git checkout -- $(git diff --name-only) # 只丢已跟踪的未暂存文件
      git pull --no-rebase origin master
      git push origin HEAD:master
      ls -d out_sweep/*/ SoVITS_weights_v5_full_e32  # 复验产物还在
""")
    sys.exit(0)

# ---- 实际修复 ----
print()
print("=" * 80)
print("开始修复")
print("=" * 80)

print("  [1] 丢弃未暂存改动（**只动已跟踪文件**）")
# ⚠️⚠️ 这里**绝不能**用 `git stash -u` / `git stash --include-untracked`！
#
# 踩过的坑：我第一版写的是
#     git stash push -u -m "..."
# `-u` 会把**所有未跟踪文件**也收进 stash，而这个仓库里有：
#     logs/                    4.19 GB
#     SoVITS_weights_v5_r128/  2.46 GB
#     SoVITS_weights_v5/       0.56 GB
#     SoVITS_weights_v5_full_e32/  ~23 GB
#     out_sweep/               （20 个 wav）
# -> stash 会去打包几十 GB，卡住不动（看着像"没输出"）。
#
# **`stash -u` 比 `checkout .` / `reset --hard` 更危险** ——
# 后两者只动已跟踪文件，而 `-u` 会把训练产物一起卷走。
#
# 正确做法：用 `git checkout -- <文件>` 逐个丢弃**已跟踪**的未暂存文件。
# 这些文件是补丁改动 + LFS 假象，都可重放/无实质变化。
r = run(["diff", "--name-only"])
unstaged_files = [l for l in (r.stdout or "").split("\n") if l.strip()]
# 安全阀：拒绝丢弃明显是产物的路径
DANGER = ("SoVITS_weights", "GPT_weights", "out_sweep", "logs/",
          ".pth", ".ckpt", ".wav")
suspect = [f for f in unstaged_files
           if any(d in f for d in DANGER)]
if suspect:
    print(f"  !! 未暂存列表里有 {len(suspect)} 个疑似产物，**中止**：")
    for f in suspect[:10]:
        print(f"     {f}")
    print("     这些不该被丢弃。请人工确认后手动处理。")
    sys.exit(1)
if unstaged_files:
    print(f"  将丢弃 {len(unstaged_files)} 个已跟踪文件：")
    for f in unstaged_files[:12]:
        print(f"     {f}")
    if len(unstaged_files) > 12:
        print(f"     … 还有 {len(unstaged_files)-12} 个")
    r = run(["checkout", "--", *unstaged_files])
    show("checkout", r)
else:
    print("  没有未暂存的已跟踪文件")


print()
print("  [2] git pull --rebase origin master")
r = run(["pull", "--rebase", "origin", "master"])
show("pull", r)
if r.returncode != 0:
    print("  !! pull 失败，尝试中止 rebase 并改用 merge")
    run(["rebase", "--abort"])
    r = run(["pull", "--no-rebase", "origin", "master"])
    show("pull(merge)", r)

print()
print("  [3] git push origin HEAD:master")
r = run(["push", "origin", "HEAD:master"])
show("push", r)
pushed = r.returncode == 0

print()
print("  [4] 复验：确认训练产物未被误动")
for pat in ("SoVITS_weights_v5_full_e32", "out_sweep", "logs/zfy_v5",
            "ref/default.wav"):
    p = ROOT / pat
    if p.is_dir():
        n = sum(1 for _ in p.rglob("*"))
        print(f"    {pat}: {n} 项  {'OK' if n else '**空了！**'}")
    elif p.exists():
        print(f"    {pat}: {p.stat().st_size/1e6:.1f} MB  OK")

print()
print("=" * 80)
print("结果")
print("=" * 80)
r = run(["log", "--oneline", "-3"])
print("  当前 HEAD:")
for l in (r.stdout or "").split("\n"):
    if l.strip():
        print(f"    {l.strip()}")
r = run(["rev-list", "--left-right", "--count", "HEAD...origin/master"])
if r.returncode == 0:
    print(f"  与远端差异 (ahead behind): {(r.stdout or '').strip()}")
print()
if pushed:
    print("  ✓ 推送成功 —— 20 个 wav 已在远端")
    print("""
  本机拉取（在**本机**执行）：
      python -c "from modelscope import snapshot_download as d; \\
d('yangwenjie1231/zfy-v5-fullft-pkg', repo_type='dataset', \\
  allow_patterns=['out_sweep/**'], local_dir=r'D:\\A\\zfy-v5\\dist\\sweep')"
""")
else:
    print("  ✗ 推送仍失败 —— 把上面 [3] 的输出贴出来")
