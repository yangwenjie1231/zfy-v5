#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实例空间清理 —— 按"是否已备份"分层，默认只诊断。

## 为什么可以大胆删
远端已确认保存（提交 `bb37ceb`）：
    ✅ out_sweep_nofreeze/          40 个 wav
    ✅ ref/                          4 个（含 zfh-e4.ckpt / default.wav）
    ✅ logs/zfy_v5/               2484 个训练特征（825 x 3）
    ✅ SoVITS_weights_..._nofreeze/  4 个权重（e8/e16/e24/e32，走 LFS）

## 分层

| 层 | 内容 | 大小 | 为什么可删 |
|---|---|---|---|
| **C1** | 其余 **28 个** nofreeze 权重 | 21.5 GB | 都是**同一个 bug**（码本漂移），远端有 4 个代表性 epoch |
| **C1** | `logs/.../logs_s2_v5dev_full/G_*.pth` | 9.2 GB | 含 optimizer state，**训练已完成**不需 resume |
| **C2** | LoRA 早期版权重（3 个目录） | 12 GB | 已被全量版取代 |
| **C2** | `GPT_weights_gptA/` `gptB/` | 2.5 GB | GPT 微调，**重训只要 3 分钟** |
| **C3** | `/tmp/zfy_up*` 克隆 | 4~8 GB | 临时克隆，推送已完成 |
| **C3** | `out_sweep/`（旧的 20 wav） | 15 MB | 远端有 |
| **C4** | `.git` 的 LFS 缓存 | 5.9 GB | `git lfs prune` 可清 |
| **保留** | `GPT_SoVITS/pretrained_models/` | 2.7 GB | **底模，重训必需**（可重下但慢） |
| **保留** | `logs/zfy_v5/` | 591 MB | **训练特征**（远端有，但本地重训要用） |
| **保留** | `ref/` | 280 MB | 合成必需 |
| **保留** | 代码 + `_tools/` | 2 MB | — |

## 用法
    python _tools/free_space.py                # 只诊断
    python _tools/free_space.py --apply 1      # 删 C1（腾 ~31 GB）
    python _tools/free_space.py --apply 12     # 删 C1+C2（腾 ~45 GB）
    python _tools/free_space.py --apply all    # 全删（腾 ~55 GB）
    python _tools/free_space.py --apply 1 --dry # 演练
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
ap.add_argument("--apply", default=None, help="1/12/123/all；不填则只诊断")
ap.add_argument("--dry", action="store_true", help="演练：只打印不删")
args = ap.parse_args()


def du(p: Path) -> int:
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
    return f"{b/1e9:6.2f} GB" if b >= 1e9 else f"{b/1e6:6.0f} MB"


print("=" * 94, flush=True)
print("实例空间清理", flush=True)
print("=" * 94, flush=True)
print(f"  仓库根: {ROOT}", flush=True)

# ---------- 当前占用 ----------
print()
print("1) 当前占用（>100 MB 的项）", flush=True)
print("-" * 94, flush=True)
items = []
for d in sorted(ROOT.iterdir()):
    if d.name in (".git",):
        continue
    b = du(d)
    if b > 1e8:
        items.append((b, d, "目录" if d.is_dir() else "文件"))
# logs 子项单独列（logs 总量可能很大但内部差异大）
lg = ROOT / "logs"
if lg.is_dir():
    for d in sorted(lg.iterdir()):
        b = du(d)
        if b > 1e8:
            items.append((b, d, "logs 子项"))
items.sort(reverse=True, key=lambda t: t[0])
for b, d, tag in items[:18]:
    try:
        rel = d.relative_to(ROOT)
    except ValueError:
        rel = d
    print(f"  {gb(b)}  {str(rel)[:62]:<62s} {tag}", flush=True)

git_b = du(ROOT / ".git")
print(f"  {gb(git_b)}  .git", flush=True)

# ---------- 分层候选 ----------
print()
print("2) 清理候选（按层）", flush=True)
print("-" * 94, flush=True)

C: dict[str, list[tuple[Path, str]]] = {"1": [], "2": [], "3": [], "4": []}

# C1a: 28 个未备份的 nofreeze 权重（保留远端已有的 4 个）
wd = ROOT / "SoVITS_weights_v5_full_e32_nofreeze"
KEEP_EPOCHS = {8, 16, 24, 32}
if wd.is_dir():
    for p in sorted(wd.glob("*.pth")):
        m = re.search(r"_e(\d+)_s\d+_l\d+", p.stem)
        if m and int(m.group(1)) not in KEEP_EPOCHS:
            C["1"].append((p, f"e{m.group(1)}（同 bug，远端有 4 个代表）"))

# C1b: G_*.pth（含 optimizer，训练已完成）
for pat in ("logs/v5_full_e32_nofreeze/logs_s2_v5dev_full/G_*.pth",
            "logs/v5_full_e32/logs_s2_v5dev_full/G_*.pth",
            "logs/smoke/logs_s2_v5dev_full/G_*.pth",
            "logs/zfy_v5_full/logs_s2_v5dev_full/G_*.pth"):
    for p in ROOT.glob(pat):
        C["1"].append((p, "训练 checkpoint（含 optimizer，已完成）"))

# C2: LoRA 早期版 + GPT A/B
for d in ("SoVITS_weights_v5", "SoVITS_weights_v5_r128",
          "SoVITS_weights_v5_full", "SoVITS_weights_smoke",
          "GPT_weights_gptA", "GPT_weights_gptB"):
    p = ROOT / d
    if p.is_dir():
        why = "GPT 微调（重训 3 分钟）" if "GPT" in d else "LoRA 早期版（已取代）"
        C["2"].append((p, why))

# C3: 临时克隆 + 旧音频
for p in (Path("/tmp/zfy_up"), Path("/tmp/zfy_up2"), Path("/tmp/zfy_up3"),
          Path("/tmp/zfy_push"), Path("/tmp/zfy_push2"), Path("/tmp/zfy_push3"),
          Path("/tmp/zfy_push_synth")):
    if p.exists():
        C["3"].append((p, "临时克隆（推送已完成）"))
os_p = ROOT / "out_sweep"
if os_p.is_dir():
    C["3"].append((os_p, "旧的 20 wav（远端有）"))

# C4: .git LFS 缓存（用 prune 而不是删 .git）
C["4"].append((ROOT / ".git", "LFS 缓存（用 git lfs prune，**不删 .git**）"))

tot = {}
for t in sorted(C):
    b = sum(du(p) for p, _ in C[t])
    tot[t] = b
    mark = "✓ 本次" if (args.apply and (args.apply == "all" or t in args.apply)) else "  跳过"
    print(f"  C{t}  {mark}  {len(C[t]):3d} 项  {gb(b)}", flush=True)

print()
print("  明细：", flush=True)
for t in sorted(C):
    if not C[t]:
        continue
    print(f"  --- C{t} ---", flush=True)
    shown = 0
    for p, why in sorted(C[t], key=lambda x: -du(x[0])):
        if shown >= 8:
            print(f"      … 还有 {len(C[t])-8} 项", flush=True)
            break
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            rel = p
        print(f"      {gb(du(p))}  {str(rel)[:52]:<52s} {why}", flush=True)
        shown += 1

print()
print("  **必须保留**（重训/合成要用）", flush=True)
for k in ("GPT_SoVITS/pretrained_models", "logs/zfy_v5", "ref",
          "GPT_SoVITS", "_tools"):
    p = ROOT / k
    if p.exists():
        print(f"      {gb(du(p))}  {k}", flush=True)

if not args.apply:
    print()
    print("=" * 94, flush=True)
    print("（只诊断。加 --apply 才动手）", flush=True)
    print()
    print("  推荐：先删 C1（腾 ~31 GB，全部已备份或无用）：", flush=True)
    print("      python _tools/free_space.py --apply 1", flush=True)
    print()
    print("  还不够就加 C2（腾 ~45 GB）：", flush=True)
    print("      python _tools/free_space.py --apply 12", flush=True)
    print("=" * 94, flush=True)
    sys.exit(0)

# ---------- 执行 ----------
want = ["1", "2", "3", "4"] if args.apply == "all" else list(args.apply)
print()
print("=" * 94, flush=True)
print(f"开始清理（层 {want}）{'  [演练模式]' if args.dry else ''}", flush=True)
print("=" * 94, flush=True)

freed = 0
for t in want:
    for p, why in C.get(t, []):
        b = du(p)
        # C4 特例：.git 用 prune，不 rmtree
        if p.name == ".git" and p.parent == ROOT:
            if args.dry:
                print(f"  [演练] git lfs prune", flush=True)
                continue
            print(f"  git lfs prune …", flush=True)
            r = subprocess.run(["git", "lfs", "prune"], cwd=str(ROOT),
                               capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=1800)
            for l in ((r.stdout or "") + (r.stderr or "")).split("\n")[-4:]:
                if l.strip():
                    print(f"      {l.strip()[:90]}", flush=True)
            after = du(ROOT / ".git")
            freed += max(0, b - after)
            continue
        if args.dry:
            print(f"  [演练] 将删 {p}  ({gb(b)})", flush=True)
            freed += b
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            freed += b
            print(f"  ✓ 删除 {str(p)[:70]}  ({gb(b)})", flush=True)
        except OSError as e:
            print(f"  ✗ 删除失败 {p}: {e}", flush=True)

print()
print("=" * 94, flush=True)
print(f"释放 {gb(freed)}", flush=True)
print("  剩余占用：", flush=True)
for k in ("SoVITS_weights_v5_full_e32_nofreeze", "logs", "GPT_SoVITS",
          "ref", ".git"):
    p = ROOT / k
    if p.exists():
        print(f"    {gb(du(p))}  {k}", flush=True)
print(f"    {gb(du(ROOT))}  总计", flush=True)
try:
    st = os.statvfs(str(ROOT))
    print(f"    statvfs 可用 {st.f_bavail*st.f_frsize/1e12:.2f} TB"
          f"（**是 NAS 总池，不是配额**）", flush=True)
except Exception:  # noqa: BLE001
    pass
print("=" * 94, flush=True)
