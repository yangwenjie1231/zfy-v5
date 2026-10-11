#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""云端 git 冲突一键解决。

## 冲突的根因（不是真的冲突）

云端跑过这几个**就地修改 tracked 文件**的补丁脚本：

    patch_v5_fullft_infer.py  -> GPT_SoVITS/TTS_infer_pack/TTS.py
                                 GPT_SoVITS/text/LangSegmenter/langsegmenter.py
                                 GPT_SoVITS/text/chinese2.py
    patch_s1_train_rocm.py    -> GPT_SoVITS/s1_train.py
    patch_strict_audio.py     -> GPT_SoVITS/module/data_utils.py
                                 GPT_SoVITS/s2_train_v5_full.py
    patch_models_for_torch212.py -> GPT_SoVITS/module/models.py

于是 `git pull` 时这些文件有**未提交的本地修改**，与远端更新冲突。

## 为什么可以放心丢弃本地修改

**补丁是可重放的** —— 它们都幂等（检测到标记就跳过），
而且原始内容就在远端。所以正确做法是：

    git checkout -- <这些文件>      # 丢弃本地补丁
    git pull                        # 拉最新
    bash/py 重新打补丁               # 重放

**关键：绝不能 `git checkout .` 或 `git reset --hard`** ——
那会把**训练产出的权重、logs、out_sweep 一起清掉**吗？
不会（它们大多未 tracked），但会丢掉 out_sweep 里已暂存的内容。
所以本脚本**只对补丁文件**做 checkout，逐个列出。

## 用法
    cd /mnt/workspace/zfy-v5-fullft-pkg
    python _tools/resolve_git_conflict.py            # 只报告
    python _tools/resolve_git_conflict.py --apply    # 实际解决
    python _tools/resolve_git_conflict.py --apply --repatch   # 解决后重放补丁
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path.cwd()
for c in (ROOT, Path("/mnt/workspace/zfy-v5-fullft-pkg")):
    if (c / ".git").is_dir() and (c / "GPT_SoVITS").is_dir():
        ROOT = c
        break


def run(args: list[str], timeout: int = 300):
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=timeout)


# 补丁会就地修改的 tracked 文件（丢弃后可重放）
PATCHED = [
    "GPT_SoVITS/TTS_infer_pack/TTS.py",
    "GPT_SoVITS/text/LangSegmenter/langsegmenter.py",
    "GPT_SoVITS/text/chinese2.py",
    "GPT_SoVITS/s1_train.py",
    "GPT_SoVITS/module/data_utils.py",
    "GPT_SoVITS/s2_train_v5_full.py",
    "GPT_SoVITS/module/models.py",
]

# 绝不能丢的（训练产物 / 权重 / 音频）
PRECIOUS = ("SoVITS_weights", "GPT_weights", "logs/", "out_sweep",
            "ref/", "*.pth", "*.ckpt", "*.wav")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--repatch", action="store_true",
                    help="解决后重放补丁")
    args = ap.parse_args()

    print("=" * 78)
    print(f"仓库: {ROOT}")
    print("=" * 78)

    # ---- 1) 现状 ----
    print()
    print("1) 当前状态")
    st = run(["status", "--porcelain"])
    lines = [l for l in (st.stdout or "").split("\n") if l.strip()]
    if not lines:
        print("  工作区干净，没有冲突")
    for l in lines[:40]:
        print(f"  {l}")
    if len(lines) > 40:
        print(f"  … 还有 {len(lines)-40} 项")

    # ---- 2) 是否有未完成的 merge/rebase ----
    print()
    print("2) 未完成的操作")
    for f, label in (("MERGE_HEAD", "merge"), ("REBASE_HEAD", "rebase"),
                     ("CHERRY_PICK_HEAD", "cherry-pick")):
        p = ROOT / ".git" / f
        print(f"  {label:12s}: {'**进行中**' if p.exists() else '无'}")

    # ---- 3) 冲突标记扫描 ----
    print()
    print("3) 冲突标记（<<<<<<<）")
    # ⚠️ 要排除**本脚本自身** —— 它的源码里含 `"<<<<<<< "` 字面量
    #    （用来检测冲突），会被自己误判成冲突文件。
    SELF = {Path(__file__).name, "resolve_git_conflict.py",
            "verify_shell_heredocs.py"}
    hits = []
    for f in ROOT.rglob("*.py"):
        if ".git" in f.parts or "__pycache__" in f.parts:
            continue
        if f.name in SELF:
            continue
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        # 行首必须是冲突标记本身，避免误报
        for line in txt.split("\n"):
            if line.startswith("<<<<<<< ") or line.startswith(">>>>>>> "):
                hits.append(f.relative_to(ROOT))
                break
    if hits:
        for h in hits[:20]:
            print(f"  {h}")
    else:
        print("  无")

    # ---- 4) 找出「补丁文件被本地改过」的 ----
    print()
    print("4) 被补丁改过、且与远端有分歧的文件")
    dirty = []
    for rel in PATCHED:
        r = run(["diff", "--name-only", "HEAD", "--", rel])
        if (r.stdout or "").strip():
            dirty.append(rel)
            print(f"  {rel}  <- 有本地修改")
        else:
            print(f"  {rel}  （无修改）")

    # 也检查未 tracked 但会挡 pull 的
    print()
    print("5) 安全检查：训练产物是否会被误删")
    # ⚠️ 用相对 ROOT 的**固定路径**，不要用带通配符的字符串去 Path()
    for pat in ("SoVITS_weights_v5_full_e32", "SoVITS_weights_v5_full",
                "out_sweep", "logs/zfy_v5", "logs/v5_full_e32",
                "ref/default.wav", "ref/zfh-e4.ckpt"):
        p = ROOT / pat
        if not p.exists():
            continue
        if p.is_dir():
            n = sum(1 for _ in p.rglob("*"))
            print(f"  {pat}: {n} 项  （**本脚本不会动它**）")
        else:
            print(f"  {pat}: {p.stat().st_size/1e6:.1f} MB  （**本脚本不会动它**）")
    # 额外列出所有 SoVITS_weights* 目录，让用户确认
    extra = sorted(d.name for d in ROOT.glob("SoVITS_weights*") if d.is_dir())
    if extra:
        print(f"  权重目录共 {len(extra)} 个: {', '.join(extra[:8])}"
              + (" …" if len(extra) > 8 else ""))

    if not args.apply:
        print()
        print("=" * 78)
        print("（未加 --apply，只报告）")
        print("=" * 78)
        print(f"""
  要解决冲突，执行：
      python _tools/resolve_git_conflict.py --apply --repatch

  它会：
    1. 中止未完成的 merge/rebase（若有）
    2. 只对上面 {len(dirty)} 个**补丁文件**执行 git checkout --
       （**绝不** git checkout . / reset --hard）
    3. git pull
    4. --repatch 时重放 4 个补丁脚本（幂等）

  ⚠️ 权重 / logs / out_sweep / ref **完全不受影响**。
""")
        return 0

    # ---- 实际解决 ----
    print()
    print("=" * 78)
    print("开始解决")
    print("=" * 78)

    # 1) 中止未完成操作
    for f, cmd in (("MERGE_HEAD", ["merge", "--abort"]),
                   ("REBASE_HEAD", ["rebase", "--abort"]),
                   ("CHERRY_PICK_HEAD", ["cherry-pick", "--abort"])):
        if (ROOT / ".git" / f).exists():
            r = run(cmd)
            print(f"  {cmd[0]} --abort -> {r.returncode}")

    # 2) 只 checkout 补丁文件
    if dirty:
        r = run(["checkout", "--", *dirty])
        print(f"  已丢弃 {len(dirty)} 个补丁文件的本地修改: "
              f"{'OK' if r.returncode == 0 else 'FAIL'}")
        for d in dirty:
            print(f"     {d}")
    else:
        print("  没有需要丢弃的补丁文件")

    # 3) pull
    print()
    print("  git pull …")
    r = run(["pull", "--no-rebase", "origin", "master"], timeout=900)
    out = (r.stdout or "") + (r.stderr or "")
    print("\n".join(f"    {l}" for l in out.strip().split("\n")[-12:]))
    if r.returncode != 0:
        print("  !! pull 失败 —— 可能还有其他未提交文件挡住")
        st = run(["status", "--porcelain"])
        print("  当前状态：")
        for l in (st.stdout or "").split("\n")[:20]:
            if l.strip():
                print(f"    {l}")
        print("""
  处理办法：
    · 若列出的是**补丁文件** -> 把它们加进本脚本的 PATCHED 列表后重跑
    · 若是**你自己改的**文件 -> 先 git stash（会保留，可 git stash pop 找回）
    · 若是**权重/logs** -> 它们通常未 tracked，不会挡 pull
""")
        return 1

    # 4) 重放补丁
    if args.repatch:
        print()
        print("  重放补丁…")
        cmds = [
            ["python", "_tools/patch_v5_fullft_infer.py", "--root", "."],
            ["python", "_tools/patch_s1_train_rocm.py", "--root", "."],
            ["python", "_tools/patch_strict_audio.py", "--root", "."],
            ["python", "_tools/patch_models_for_torch212.py", "--root", "."],
        ]
        for c in cmds:
            if not (ROOT / c[1]).exists():
                print(f"    跳过 {c[1]}（不存在）")
                continue
            r = subprocess.run(c, cwd=str(ROOT), capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=600)
            tail = [l for l in ((r.stdout or "") + (r.stderr or "")).split("\n")
                    if l.strip()][-3:]
            print(f"    {c[1]} -> exit {r.returncode}")
            for l in tail:
                print(f"       {l.strip()[:100]}")

    print()
    print("=" * 78)
    print("完成")
    print("=" * 78)
    print(f"""
  当前 HEAD: {(run(['log','-1','--oneline']).stdout or '').strip()}
  工作区剩余改动:
""")
    st = run(["status", "--porcelain"])
    for l in (st.stdout or "").split("\n")[:15]:
        if l.strip():
            print(f"    {l}")
    print("""
  下一步：
    python _tools/cloud_synth_sweep.py \\
        --weights-dir SoVITS_weights_v5_full_e32 --epochs 16 --no-v2pro
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
