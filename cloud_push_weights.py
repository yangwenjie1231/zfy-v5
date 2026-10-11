# -*- coding: utf-8 -*-
"""把训练好的权重回传数据集（云端执行）。

## 为什么需要选择性上传

v5 全量微调每轮都存权重，e1..e32 共 **32 个 × 736 MB = 23.5 GB**。
全传上去按实测 ~1.3 MB/s 要 5 小时，不现实。

所以：
  · 默认只传**指定 epoch**（--epochs 8,16,24,32 → 4 个，约 2.9 GB）
  · 或者先只传音频（`cloud_synth_sweep.py`），听完再传选中的那个

## 用法（云端，检出根）

    python _tools/cloud_push_weights.py --list
    python _tools/cloud_push_weights.py --epochs 8,32
    python _tools/cloud_push_weights.py --epochs 32 --tag best
    python _tools/cloud_push_weights.py --pattern "SoVITS_weights_v5_full*" --latest 3

产出：仓库内 `weights/<实验名>/<文件名>.pth`，然后本机可 clone/pull 下载。
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

RID = "yangwenjie1231/zfy-v5-fullft-pkg"


def find_root() -> Path:
    here = Path.cwd()
    for c in (here, *here.parents, Path("/mnt/workspace/zfy-v5-fullft-pkg")):
        if (c / "GPT_SoVITS").is_dir() and (c / ".git").is_dir():
            return c
    return Path("/mnt/workspace/zfy-v5-fullft-pkg")


def find_token() -> str | None:
    for k in ("MODELSCOPE_API_TOKEN", "MS_TOKEN", "MODELSCOPE_TOKEN"):
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()
    cred = Path.home() / ".modelscope" / "credentials"
    if cred.is_dir():
        for p in sorted(cred.iterdir()):
            if p.is_file():
                v = p.read_text(encoding="utf-8", errors="replace").strip()
                if v and "\n" not in v and 10 <= len(v) <= 200:
                    return v
    try:
        r = subprocess.run(["git", "remote", "get-url", "origin"],
                           cwd=str(Path.cwd()), capture_output=True,
                           text=True, timeout=30)
        m = re.search(r"oauth2:([^@]+)@", r.stdout or "")
        if m:
            return m.group(1)
    except Exception:  # noqa: BLE001
        pass
    return None


def ep_of(name: str) -> int | None:
    """从权重名解析 epoch —— **必须锚定到结尾**。

    名字形如 v5_full_e32_e8_s432_l0.pth，同时含 "e32"（实验名）和
    "e8"（真 epoch）。非锚定正则会命中前者。
    """
    m = re.search(r"_e(\d+)_s(\d+)_l\d+\.pth$", name)
    if m:
        return int(m.group(1))
    m = re.search(r"_e(\d+)_l\d+\.pth$", name)
    return int(m.group(1)) if m else None


def discover(root: Path, pattern: str) -> list[Path]:
    out = []
    for d in sorted(root.glob(pattern)):
        if d.is_dir():
            out += sorted(d.glob("*.pth"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="SoVITS_weights_v5_full*",
                    help="权重目录 glob（默认所有 v5 全量实验）")
    ap.add_argument("--epochs", default=None,
                    help="逗号分隔的 epoch 号，如 8,16,24,32")
    ap.add_argument("--latest", type=int, default=None,
                    help="只取每轮实验中最新/最大的 N 个 epoch")
    ap.add_argument("--tag", default=None,
                    help="上传到 weights/<tag>/；默认用权重所在目录名")
    ap.add_argument("--list", action="store_true", help="只列出，不上传")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = find_root()
    os.chdir(root)

    all_w = discover(root, args.pattern)
    if not all_w:
        print(f"没找到权重（pattern={args.pattern}）")
        print(f"检出根: {root}")
        dirs = sorted(p.name for p in root.glob("SoVITS_weights*") if p.is_dir())
        if dirs:
            print(f"实际存在: {dirs}")
        return 1

    print("=" * 74)
    print("可回传的权重")
    print("=" * 74)
    by_dir: dict[str, list[Path]] = {}
    for w in all_w:
        by_dir.setdefault(w.parent.name, []).append(w)
    for d, ws in by_dir.items():
        tot = sum(p.stat().st_size for p in ws) / 1e9
        print(f"  {d:34s} {len(ws):3d} 个  {tot:6.2f} GB")
        eps = sorted(x for x in (ep_of(p.name) for p in ws) if x is not None)
        if eps:
            print(f"      epochs: {eps[0]}..{eps[-1]}")

    # ---- 选择 ----
    sel = all_w
    if args.epochs:
        want = {int(x) for x in args.epochs.split(",") if x.strip()}
        sel = [p for p in sel if ep_of(p.name) in want]
    if args.latest:
        keep = []
        for d, ws in by_dir.items():
            ws2 = sorted([p for p in ws if p in sel],
                         key=lambda p: ep_of(p.name) or 0)
            keep += ws2[-args.latest:]
        sel = keep

    if not sel:
        print("\n筛选后没有权重可选")
        return 1

    tot = sum(p.stat().st_size for p in sel) / 1e9
    print()
    print("=" * 74)
    print(f"待上传 {len(sel)} 个，合计 {tot:.2f} GB"
          f"（按 1.3 MB/s 约 {tot*1000/1.3/60:.0f} 分钟）")
    print("=" * 74)
    for p in sel:
        print(f"  {p.parent.name}/{p.name}  {p.stat().st_size/1e6:.1f} MB")

    if args.list:
        return 0
    if args.dry_run:
        print("\n(dry-run)")
        return 0

    # ---- 复制进仓库 ----
    staged = []
    for p in sel:
        sub = args.tag or p.parent.name
        dst = root / "weights" / sub / p.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != p.stat().st_size:
            shutil.copy2(p, dst)
        staged.append(dst.relative_to(root))
        print(f"  已暂存 {dst.relative_to(root)}")

    # ---- git 提交推送 ----
    def run(a, **kw):
        return subprocess.run(["git", *a], cwd=str(root), capture_output=True,
                              text=True, timeout=7200, **kw)

    run(["config", "user.name", "yangwenjie1231"])
    run(["config", "user.email", "yangwenjie1231@163.com"])
    run(["add", "--", "weights"])

    st = run(["status", "--porcelain", "weights"])
    if not (st.stdout or "").strip():
        print("  无变更（权重已在仓库里）")
    else:
        c = run(["commit", "-m",
                 f"upload {len(staged)} checkpoint(s) "
                 f"({', '.join(sorted({s.parts[1] for s in staged}))})"])
        if c.returncode != 0:
            print(f"  commit 失败: {(c.stdout or '')[-200:]}{(c.stderr or '')[-300:]}")
            return 1
        print("  已提交")

    n_tree = len([l for l in
                  (run(["ls-tree", "-r", "--name-only", "HEAD"]).stdout or "")
                  .split("\n") if l.strip()])
    print(f"  提交树文件数: {n_tree}")
    if n_tree < 1000:
        print("  !! 文件数异常，中止推送（防误删，见 push_single.py 事故）")
        return 1

    token = find_token()
    urls = ([f"https://oauth2:{token}@www.modelscope.cn/datasets/{RID}.git"]
            if token else []) + ["origin"]
    print("  推送中…（大文件走 LFS，可能较慢）", flush=True)
    pushed = False
    for u in urls:
        r = run(["push", u, "HEAD:master"])
        if r.returncode == 0:
            pushed = True
            break
        out = (r.stdout or "") + (r.stderr or "")
        if "rejected" in out or "non-fast-forward" in out:
            run(["pull", "--rebase", u, "master"])
            r = run(["push", u, "HEAD:master"])
            if r.returncode == 0:
                pushed = True
                break
        print(f"    ({u.split('@')[-1][:50]} 失败: {out[-200:]})")

    if not pushed:
        print("  推送失败。手动：git push origin HEAD:master")
        return 1
    print("  推送完成 ✅")

    print()
    print("=" * 74)
    print("本机下载（在本地仓库执行）")
    print("=" * 74)
    print(f"""
  git pull
  # 或只取权重（约 {tot:.1f} GB）
  git lfs pull --include="weights/**"
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
