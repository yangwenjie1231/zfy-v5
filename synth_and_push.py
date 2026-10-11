#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成 + 推送 + 本地下载指引（一条龙）。

## 为什么写这个
合成链路已经全通（681 键全量权重加载成功），但**推送**这步之前
踩过两个坑：
  1. 云端工作区有 2500+ 个 LFS 假象 `M` -> `git add` 卡住
     -> 解法：**全新克隆**，只复制音频（`cp` 只读原目录，零风险）
  2. `--out` 用默认值会写进 `out_sweep/e16/` **覆盖旧文件**
     -> 解法：`--out out_sweep_nofreeze`

## 本脚本
  1. 合成（e8/e16/e24/e32 × 10 句 = 40 个 wav）
  2. 全新克隆 + 只复制音频 + 推送
  3. 打印本机下载命令

## 用法
    python _tools/synth_and_push.py
    python _tools/synth_and_push.py --epochs 8,16,24,32 --out out_sweep_nofreeze
    python _tools/synth_and_push.py --skip-synth      # 只推送已有音频
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

ROOT = Path("/mnt/workspace/zfy-v5-fullft-pkg")
if not (ROOT / "GPT_SoVITS").is_dir():
    ROOT = Path.cwd()

ap = argparse.ArgumentParser()
ap.add_argument("--weights-dir",
                default="SoVITS_weights_v5_full_e32_nofreeze")
ap.add_argument("--epochs", default="8,16,24,32")
ap.add_argument("--out", default="out_sweep_nofreeze")
ap.add_argument("--skip-synth", action="store_true")
ap.add_argument("--skip-push", action="store_true")
args = ap.parse_args()

OUT_DIR = ROOT / args.out
TMP_CLONE = Path("/tmp/zfy_push_synth")


def run(cmd, **kw):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", **kw)
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    for l in out.split("\n")[-12:]:
        if l.strip():
            print(f"    {l.strip()[:112]}")
    return r.returncode, out


print("=" * 90)
print("合成 + 推送")
print("=" * 90)
print(f"  仓库根   : {ROOT}")
print(f"  权重目录 : {args.weights_dir}")
print(f"  输出目录 : {args.out}")
print(f"  epochs   : {args.epochs}")

# ---------------------------------------------------------------- 1) 合成
if not args.skip_synth:
    print()
    print("=" * 90)
    print("1) 合成")
    print("=" * 90)
    # ⚠️ 关键：--out 必须与旧目录不同，否则覆盖 out_sweep/e16 等旧文件
    if args.out == "out_sweep":
        print("  !! --out 是 out_sweep —— 会覆盖旧文件！")
        print("     请用 --out out_sweep_nofreeze")
        sys.exit(1)
    rc, out = run([
        sys.executable, str(ROOT / "_tools" / "cloud_synth_sweep.py"),
        "--weights-dir", args.weights_dir,
        "--epochs", args.epochs,
        "--no-v2pro", "--out", args.out,
    ], cwd=str(ROOT), timeout=7200)
    if rc != 0:
        print(f"  !! 合成失败 exit={rc}")
        sys.exit(1)

wavs = sorted(OUT_DIR.rglob("*.wav")) if OUT_DIR.is_dir() else []
print()
print(f"  产出 wav: {len(wavs)} 个")
subs: dict[str, int] = {}
for w in wavs:
    subs[w.parent.name] = subs.get(w.parent.name, 0) + 1
for k in sorted(subs):
    print(f"    {k:16s} {subs[k]:3d} 个")

if not wavs:
    print("  !! 没有产出，无法推送")
    sys.exit(1)

# ---------------------------------------------------------------- 2) 推送
if args.skip_push:
    print()
    print("（--skip-push，跳过推送）")
    sys.exit(0)

print()
print("=" * 90)
print("2) 推送（全新克隆，绕开脏工作区）")
print("=" * 90)

# 取 origin URL（含凭据）
rc, url = run(["git", "remote", "get-url", "origin"], cwd=str(ROOT))
url = url.strip().split("\n")[-1].strip()
if not url or "://" not in url:
    print(f"  !! 拿不到 origin URL: {url!r}")
    sys.exit(1)
print(f"  origin: {re.sub(r'oauth2:[^@]+@', 'oauth2:***@', url)}")

if TMP_CLONE.exists():
    shutil.rmtree(TMP_CLONE, ignore_errors=True)

rc, _ = run(["git", "clone", "--depth", "1", url, str(TMP_CLONE)], timeout=1800)
if rc != 0 or not (TMP_CLONE / ".git").is_dir():
    print("  !! 克隆失败")
    sys.exit(1)

# 只复制音频（cp 只读原目录 -> 零风险）
dst = TMP_CLONE / args.out
dst.mkdir(parents=True, exist_ok=True)
for w in wavs:
    rel = w.relative_to(OUT_DIR)
    tgt = dst / rel
    tgt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(w, tgt)
n_copied = len(list(dst.rglob("*.wav")))
print(f"  已复制 {n_copied} 个 wav 到克隆里")

rc, _ = run(["git", "add", args.out], cwd=str(TMP_CLONE))
if rc != 0:
    print("  !! git add 失败")
    sys.exit(1)

msg = (f"v5 nofreeze sweep audio "
       f"({args.epochs} epochs, enc_p unfrozen, {n_copied} wav)")
rc, _ = run(["git", "-c", "user.name=yangwenjie1231",
             "-c", "user.email=yangwenjie1231@163.com",
             "commit", "-m", msg], cwd=str(TMP_CLONE))
if rc != 0:
    print("  ~ 无新增内容或提交失败")

rc, _ = run(["git", "push", "origin", "HEAD:master"],
            cwd=str(TMP_CLONE), timeout=3600)
pushed = rc == 0

print()
print("=" * 90)
print("3) 结果")
print("=" * 90)
if pushed:
    print("  ✓ 推送成功")
else:
    print("  ✗ 推送失败 —— 见上面的输出")
    print(f"    克隆还在 {TMP_CLONE}，可手动重试：")
    print(f"      cd {TMP_CLONE} && git push origin HEAD:master")

print()
print("=" * 90)
print("4) 本机下载（在**本机**执行）")
print("=" * 90)
print(f"""
    cd D:\\A\\zfy-v5
    & 'C:\\Users\\A\\miniconda3\\envs\\voice\\python.exe' -c "from modelscope import snapshot_download as d; d('yangwenjie1231/zfy-v5-fullft-pkg', repo_type='dataset', allow_patterns=['{args.out}/**'], local_dir=r'D:\\A\\zfy-v5\\dist\\nofreeze')"

  下载后我会做三组对比：
    · e08 vs e32         -> SoVITS 是否饱和
    · e32 vs v2Pro       -> 解冻后是否突破 0.79 平台期
    · e32 vs e16_merged  -> 解冻 vs 补底模，哪个更好（**最关键**）
""")
print("=" * 90)
