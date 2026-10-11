#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实例到期前的**紧急回传** —— 按优先级上传，可反复重跑（自动跳过已传的）。

## 为什么不能一次性推 35 GB
实测 LFS 上传速度约 5~15 MB/s（20 个 wav 共 15 MB 用了数秒）。
35 GB 需要 **40 分钟 ~ 2 小时**，实例到期前来不及。

所以按**价值**分层，先传不可再生的、再传可重训的。

## 分层（按优先级）

| 层 | 内容 | 大小 | 为什么这个优先级 |
|---|---|---|---|
| **T1** | `out_sweep_nofreeze/`（40 wav） | ~30 MB | **秒传**，且是评估的唯一输入 |
| **T1** | `ref/`（生产权重 + 参考音频） | ~280 MB | 生产资产，丢了要重下 |
| **T2** | `SoVITS_weights_v5_full_e32_nofreeze/*_e32_*` | 766 MB | **最终模型** |
| **T2** | `..._e16_*` `..._e24_*` `..._e8_*` | 2.3 GB | 饱和验证需要 |
| **T3** | `..._e32_nofreeze` 其余 28 个 | 21.5 GB | 可重训，但重训要 45 分钟 GPU |
| **T4** | `GPT_weights_gptA/` `gptB/` | 2.5 GB | 可重训（3 分钟） |
| **T5** | `SoVITS_weights_v5*/`（LoRA 早期版） | 12 GB | 已被全量版取代 |
| **T5** | `logs/zfy_v5/`（训练特征） | 0.6 GB | 可重建（约 1 小时） |

**默认只传 T1+T2（约 3.4 GB，5~10 分钟）**，加 `--tier all` 才传全部。

## 两种上传通道
  1. **`modelscope upload`**（优先）—— 专用工具，对超大文件更稳
  2. **git + LFS**（兜底）—— 仓库 `.gitattributes` 把 `.pth` 划进 LFS

## 用法
    python _tools/emergency_upload.py                  # T1+T2（推荐）
    python _tools/emergency_upload.py --tier 1         # 只传 T1
    python _tools/emergency_upload.py --tier all       # 全部
    python _tools/emergency_upload.py --list           # 只列清单和大小
    python _tools/emergency_upload.py --channel git    # 强制走 git
"""
from __future__ import annotations

import argparse
import functools
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ⚠️ 关键：管道模式下 Python 块缓冲（8KB），而本脚本要先 du() 扫描
#    **网络挂载**（/mnt/workspace 是 NFS 风格）—— 扫描慢 + 不 flush
#    -> `python script.py | head -30` 会长时间空白，看起来像卡住。
#    实测踩到：用户跑 `--list | head -30` 一直没输出。
#    修法：强制行缓冲 + 惰性求大小（带缓存）。
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:  # noqa: BLE001
    pass

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _p(msg: str = ""):
    """带 flush 的 print —— 避免管道缓冲导致"看起来卡住"。"""
    print(msg, flush=True)


ROOT = Path("/mnt/workspace/zfy-v5-fullft-pkg")
if not (ROOT / "GPT_SoVITS").is_dir():
    ROOT = Path.cwd()

REPO_ID = "yangwenjie1231/zfy-v5-fullft-pkg"

ap = argparse.ArgumentParser()
ap.add_argument("--tier", default="12", help="1/2/12/34/5/all")
ap.add_argument("--list", action="store_true")
ap.add_argument("--channel", default="auto", choices=["auto", "ms", "git"])
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()


@functools.lru_cache(maxsize=512)
def du(p_str: str) -> int:
    """目录/文件字节数（**带缓存** —— 同一路径只扫一次）。

    ⚠️ 不要去掉缓存：TIERS 构建、汇总、明细三处都会调 du，
       在 NFS 挂载上重复扫描 `logs/zfy_v5`（2488 文件）非常慢。
    """
    p = Path(p_str)
    if p.is_file():
        try:
            return p.stat().st_size
        except OSError:
            return 0
    if not p.is_dir():
        return 0
    t = 0
    try:
        for f in p.rglob("*"):
            try:
                if f.is_file() and not f.is_symlink():
                    t += f.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return t


def gb(b: int) -> str:
    return f"{b/1e9:.2f} GB" if b >= 1e9 else f"{b/1e6:.0f} MB"


# ---------------------------------------------------------------- 分层清单
W = "SoVITS_weights_v5_full_e32_nofreeze"
TIERS: dict[str, list[tuple[Path, str]]] = {}


def add_tier(name: str, paths: list[tuple[Path, str]]):
    TIERS.setdefault(name, []).extend(paths)


# T1：秒传的 + 不可再生的
t1 = []
if (ROOT / "out_sweep_nofreeze").is_dir():
    t1.append((ROOT / "out_sweep_nofreeze", "40 个合成音频（评估输入）"))
if (ROOT / "out_sweep").is_dir():
    t1.append((ROOT / "out_sweep", "旧的 20 个音频"))
if (ROOT / "ref").is_dir():
    t1.append((ROOT / "ref", "生产权重 + 参考音频"))
add_tier("1", t1)

# T2：最终模型 + 饱和验证需要的 epoch
t2 = []
wd = ROOT / W
if wd.is_dir():
    for pat, why in (("*_e32_s1728_l0.pth", "**最终模型**"),
                     ("*_e16_*.pth", "饱和验证 e16"),
                     ("*_e24_*.pth", "饱和验证 e24"),
                     ("*_e8_*.pth", "饱和验证 e08")):
        for p in sorted(wd.glob(pat)):
            t2.append((p, why))
add_tier("2", t2)

# T3：其余 epoch
t3 = []
if wd.is_dir():
    chosen = {p for p, _ in t2}
    for p in sorted(wd.glob("*.pth")):
        if p not in chosen:
            m = re.search(r"_e(\d+)_", p.stem)
            t3.append((p, f"e{m.group(1)}" if m else p.stem))
add_tier("3", t3)

# T4：GPT A/B
t4 = []
for d in ("GPT_weights_gptA", "GPT_weights_gptB"):
    if (ROOT / d).is_dir():
        t4.append((ROOT / d, f"{d}（GPT 微调）"))
add_tier("4", t4)

# T5：LoRA 早期版 + 训练特征
t5 = []
for d in ("SoVITS_weights_v5", "SoVITS_weights_v5_r128",
          "SoVITS_weights_v5_full", "logs/zfy_v5"):
    if (ROOT / d).is_dir():
        t5.append((ROOT / d, f"{d}（可重训/可重建）"))
add_tier("5", t5)

# ---------------------------------------------------------------- 选择
if args.tier == "all":
    want = ["1", "2", "3", "4", "5"]
else:
    want = list(args.tier) if args.tier else ["1", "2"]
want = [w for w in want if w in TIERS]

sel: list[tuple[Path, str, str]] = []
for t in want:
    for p, why in TIERS[t]:
        sel.append((p, why, t))

print("=" * 92)
print("实例到期前紧急回传")
print("=" * 92)
print(f"  仓库根 : {ROOT}")
print(f"  仓库   : {REPO_ID}")
print(f"  层     : {want}   条目 {len(sel)}")

print()
print("  分层清单：")
for t in sorted(TIERS):
    tot = sum(du(str(p)) for p, _ in TIERS[t])
    mark = "✓ 本次" if t in want else "  跳过"
    print(f"    T{t}  {mark}  {len(TIERS[t]):3d} 项  {gb(tot):>10s}")

total = sum(du(str(p)) for p, _, _ in sel)
print()
print(f"  **本次合计: {gb(total)}**")

print()
print("  明细（前 20）：")
shown = 0
for p, why, t in sel:
    if shown >= 20:
        print(f"    … 还有 {len(sel)-20} 项")
        break
    print(f"    T{t}  {gb(du(str(p))):>10s}  {p.name[:52]:<52s} {why}")
    shown += 1

if args.list:
    sys.exit(0)

if not sel:
    print("  !! 没有可传的项")
    sys.exit(1)

# ---------------------------------------------------------------- 通道探测
def have_ms() -> bool:
    try:
        r = subprocess.run(["modelscope", "--version"], capture_output=True,
                           text=True, timeout=30)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def have_git_token() -> bool:
    for c in (Path.home() / ".modelscope" / "credentials" / "git_token",
              Path("/root/.modelscope/credentials/git_token")):
        if c.exists():
            return True
    # 仓库 remote URL 里带凭据也算
    try:
        r = subprocess.run(["git", "remote", "get-url", "origin"],
                           cwd=str(ROOT), capture_output=True, text=True,
                           timeout=30)
        return "oauth2:" in (r.stdout or "")
    except Exception:  # noqa: BLE001
        return False


print()
print("=" * 92)
print("上传通道")
print("=" * 92)
ms_ok = have_ms()
git_ok = have_git_token()
print(f"  modelscope CLI : {'✓ 可用' if ms_ok else '✗ 不可用'}")
print(f"  git + LFS      : {'✓ 可用' if git_ok else '✗ 不可用'}")

channel = args.channel
if channel == "auto":
    # ⚠️ 对超大文件，git+LFS 的 `git add` 会对每个文件算 SHA256（766 MB × N）
    #    但 modelscope upload 是专用工具，通常更快更稳。
    #    这里优先 ms，但**只有在小批量（<=3 GB）时**才用 —— 大批量时
    #    ms upload 的断点续传不如 git 可靠。
    channel = "ms" if (ms_ok and total <= 3.5e9) else ("git" if git_ok else "ms")
print(f"  -> 选用: {channel}")

if args.dry_run:
    print("\n（--dry-run，不实际上传）")
    sys.exit(0)

# ---------------------------------------------------------------- 通道 1: ms
def upload_ms(items: list[tuple[Path, str, str]]) -> bool:
    ok_all = True
    for p, why, t in items:
        rel = p.relative_to(ROOT)
        print(f"\n  [ms] 上传 {rel}  ({gb(du(str(p)))})  {why}", flush=True)
        t0 = time.time()
        r = subprocess.run(
            ["modelscope", "upload", REPO_ID, str(p), str(rel)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=14400)
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        dt = time.time() - t0
        if r.returncode == 0:
            sp = du(str(p)) / max(dt, 1e-9) / 1e6
            print(f"       ✓ 完成 {dt:.0f}s  ({sp:.1f} MB/s)")
        else:
            print(f"       ✗ 失败 exit={r.returncode} {dt:.0f}s")
            for l in out.split("\n")[-6:]:
                if l.strip():
                    print(f"         {l.strip()[:100]}")
            ok_all = False
    return ok_all


# ---------------------------------------------------------------- 通道 2: git
def upload_git(items: list[tuple[Path, str, str]]) -> bool:
    """直接在当前仓库 git add + commit + push（不做全新克隆）。

    ## 为什么不全新克隆
    全新克隆要再占一份 24.5 GB 磁盘（配额可能不够）。
    而 `git add <具体路径>` **只暂存指定路径** ——
    工作区那 2500+ 个 LFS 假象 `M` 不会被带进来。

    ## 为什么分组提交
    每组独立 commit + push，这样中途失败时**已完成的部分保留在远端**，
    重跑时 git 会跳过已推送的对象。
    """
    ok_all = True
    # 按 tier 分组，逐组提交推送（失败时前面的已保住）
    groups: dict[str, list[Path]] = {}
    for p, why, t in items:
        groups.setdefault(t, []).append(p)

    for t in sorted(groups):
        paths = groups[t]
        rels = [str(p.relative_to(ROOT)) for p in paths]
        tot = sum(du(str(p)) for p in paths)
        print(f"\n  [git] T{t}: {len(paths)} 项  {gb(tot)}", flush=True)

        for r in rels:
            print(f"       add {r}")
        rc = subprocess.run(["git", "add", "--"] + rels, cwd=str(ROOT),
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=7200)
        if rc.returncode != 0:
            print(f"       ✗ git add 失败")
            for l in ((rc.stdout or "") + (rc.stderr or "")).split("\n")[-5:]:
                if l.strip():
                    print(f"         {l.strip()[:100]}")
            ok_all = False
            continue

        msg = f"emergency upload T{t} ({len(paths)} items, {gb(tot)})"
        rc = subprocess.run(
            ["git", "-c", "user.name=yangwenjie1231",
             "-c", "user.email=yangwenjie1231@163.com",
             "commit", "-m", msg], cwd=str(ROOT),
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=3600)
        out = (rc.stdout or "") + (rc.stderr or "")
        if rc.returncode != 0 and "nothing to commit" not in out:
            print(f"       ~ commit: {out.strip().splitlines()[-1][:90] if out.strip() else ''}")

        print(f"       push（LFS 上传中，可能较久）…", flush=True)
        t0 = time.time()
        rc = subprocess.run(["git", "push", "origin", "HEAD:master"],
                            cwd=str(ROOT), capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=28800)
        out = ((rc.stdout or "") + (rc.stderr or "")).strip()
        dt = time.time() - t0
        for l in out.split("\n")[-8:]:
            if l.strip():
                print(f"         {l.strip()[:104]}")
        if rc.returncode == 0:
            sp = tot / max(dt, 1e-9) / 1e6
            print(f"       ✓ T{t} 推送完成 {dt:.0f}s  ({sp:.1f} MB/s)")
        else:
            print(f"       ✗ T{t} 推送失败 {dt:.0f}s")
            ok_all = False
    return ok_all


print()
print("=" * 92)
print("开始上传")
print("=" * 92)
t_all = time.time()
if channel == "ms":
    ok = upload_ms(sel)
else:
    ok = upload_git(sel)
dt = time.time() - t_all

print()
print("=" * 92)
print("结果")
print("=" * 92)
print(f"  耗时     : {dt/60:.1f} 分钟")
print(f"  传输量   : {gb(total)}")
print(f"  平均速度 : {total/max(dt,1e-9)/1e6:.1f} MB/s")
print(f"  状态     : {'✓ 全部成功' if ok else '✗ 有失败项'}")
if not ok:
    print()
    print("  重跑本脚本会自动跳过已推送的对象：")
    print(f"      python _tools/emergency_upload.py --tier {args.tier}")
print()
print("  本机校验（下载清单）：")
print(f"      python -c \"from modelscope import snapshot_download as d; "
      f"d('{REPO_ID}', repo_type='dataset', "
      f"allow_patterns=['{W}/**'], local_dir=r'D:\\A\\zfy-v5\\dist\\w')\"")
print("=" * 92)
sys.exit(0 if ok else 1)
