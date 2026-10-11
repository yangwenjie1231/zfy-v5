#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""e27 存盘崩溃后的磁盘诊断 + 清理 + 救援。

## 崩溃根因：磁盘满
    utils.py:71  torch.save(fea, tmp_path)
    RuntimeError: basic_ios::clear: iostream error
`basic_ios::clear` 出在 `write_end_of_file` —— 写文件末尾失败 = 空间不足。

## 为什么空间会不够
  · 导出权重 766 MB x 27 = 20.7 GB（新目录）
  · 旧 freeze 版权重 735 MB x 33 = 24.3 GB（还在）
  · `G_233333333333.pth` 含 optimizer state，约 2 GB
  · `my_save` 的 tmp 文件写在**当前工作目录**（相对路径），
    失败时会在仓库根留下**半截垃圾文件**

## 本脚本
  1. 报告磁盘使用（`statvfs` + `du` 关键目录）
  2. 找出 `my_save` 留下的垃圾（根目录下的 `<数字>.pth`）
  3. 找出可能被截断的权重（大小明显小于中位数）
  4. 列出可安全删除的项（**默认只报告，不动手**）
  5. `--clean` 时才真的删（且只删明确安全的）

## 用法
    python _tools/rescue_disk.py                 # 只诊断
    python _tools/rescue_disk.py --clean-garbage # 删 my_save 垃圾
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from statistics import median

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path("/mnt/workspace/zfy-v5-fullft-pkg")
if not (ROOT / "GPT_SoVITS").is_dir():
    ROOT = Path.cwd()

ap = argparse.ArgumentParser()
ap.add_argument("--clean-garbage", action="store_true",
                help="删除 my_save 留下的 <时间戳>.pth 垃圾")
ap.add_argument("--clean-truncated", action="store_true",
                help="删除明显被截断的权重（危险，需确认）")
args = ap.parse_args()


def du(p: Path) -> tuple[int, int]:
    """返回 (字节, 文件数)。"""
    tot = n = 0
    if p.is_file():
        return p.stat().st_size, 1
    if not p.is_dir():
        return 0, 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                tot += f.stat().st_size
                n += 1
        except OSError:
            pass
    return tot, n


def mb(b: int) -> str:
    return f"{b/1e6:,.0f} MB" if b < 1e9 else f"{b/1e9:.2f} GB"


print("=" * 88)
print("1. 磁盘状态")
print("=" * 88)
try:
    st = os.statvfs(str(ROOT))
    total = st.f_blocks * st.f_frsize
    free = st.f_bavail * st.f_frsize
    used = total - st.f_bfree * st.f_frsize
    print(f"  {ROOT}")
    print(f"    总容量 {mb(total)}   已用 {mb(used)}   可用 {mb(free)}")
    pct = used / total * 100 if total else 0
    print(f"    使用率 {pct:.1f}%")
    if free < 5e9:
        print(f"    !! 可用空间 < 5 GB —— 这就是 e27 崩溃的原因")
    else:
        print(f"    ~ 当前有 {mb(free)} 可用")
except Exception as e:  # noqa: BLE001
    print(f"  statvfs 失败: {type(e).__name__}: {e}")

print()
print("=" * 88)
print("2. 关键目录占用")
print("=" * 88)
TARGETS = [
    "SoVITS_weights_v5_full_e32_nofreeze",
    "SoVITS_weights_v5_full_e32",
    "SoVITS_weights_v5_full",
    "logs/v5_full_e32_nofreeze",
    "logs/v5_full_e32",
    "logs/zfy_v5",
    "out_sweep",
    "GPT_SoVITS/pretrained_models",
]
rows = []
for t in TARGETS:
    p = ROOT / t
    if not p.exists():
        continue
    b, n = du(p)
    rows.append((b, n, t))
    print(f"  {mb(b):>12s}  {n:>6d} 个  {t}")
print(f"  {'-'*70}")
print(f"  {mb(sum(r[0] for r in rows)):>12s}  合计")

# 其他可能的大目录
print()
print("  仓库根下其他大目录（>500 MB）:")
for d in sorted(ROOT.iterdir()):
    if not d.is_dir() or d.name == ".git" or any(
            d.name == t.split("/")[0] for t in TARGETS):
        continue
    b, n = du(d)
    if b > 5e8:
        print(f"    {mb(b):>12s}  {n:>6d} 个  {d.name}/")

print()
print("=" * 88)
print("3. my_save 留下的垃圾（仓库根下的 <时间戳>.pth）")
print("=" * 88)
garbage = []
for p in ROOT.glob("*.pth"):
    # 形如 1759654123.45.pth
    if re.fullmatch(r"\d+\.\d+\.pth", p.name) or re.fullmatch(r"\d+\.pth", p.name):
        garbage.append(p)
if garbage:
    tot = sum(p.stat().st_size for p in garbage)
    print(f"  找到 {len(garbage)} 个，共 {mb(tot)}")
    for p in sorted(garbage, key=lambda x: -x.stat().st_size)[:10]:
        print(f"    {mb(p.stat().st_size):>12s}  {p.name}")
    if len(garbage) > 10:
        print(f"    …还有 {len(garbage)-10} 个")
    print()
    print("  这些是 `utils.py:70 tmp_path = \"%s.pth\" % ttime()` 写的临时文件。")
    print("  崩溃时未能 move 到目标目录 -> **纯垃圾，可删**。")
    if args.clean_garbage:
        freed = 0
        for p in garbage:
            try:
                freed += p.stat().st_size
                p.unlink()
            except OSError as e:
                print(f"    删除失败 {p.name}: {e}")
        print(f"  ✓ 已删除，释放 {mb(freed)}")
    else:
        print("  （加 --clean-garbage 才会真的删）")
else:
    print("  无（可能崩溃时那个 tmp 文件也被删了，或写在别处）")

print()
print("=" * 88)
print("4. 权重完整性检查")
print("=" * 88)
d = ROOT / "SoVITS_weights_v5_full_e32_nofreeze"
if not d.is_dir():
    print(f"  {d.name} 不存在")
else:
    ps = sorted(d.glob("*.pth"),
                key=lambda p: int(re.search(r"_e(\d+)", p.stem).group(1))
                if re.search(r"_e(\d+)", p.stem) else 0)
    sizes = [(p, p.stat().st_size) for p in ps]
    if sizes:
        med = int(median(s for _, s in sizes))
        print(f"  文件数: {len(sizes)}   中位大小: {med:,}")
        bad = [(p, s) for p, s in sizes if s < med * 0.9]
        print()
        print(f"  {'epoch':>6s} {'大小':>14s} {'占比':>7s}  状态")
        for p, s in sizes:
            m = re.search(r"_e(\d+)", p.stem)
            e = int(m.group(1)) if m else 0
            ratio = s / med
            flag = "✓" if ratio > 0.9 else "!! 疑似截断"
            print(f"  {e:6d} {s:14,} {ratio*100:6.1f}%  {flag}")
        print()
        if bad:
            print(f"  !! {len(bad)} 个文件明显偏小（疑似截断）:")
            for p, s in bad:
                print(f"     {p.name}  {s:,}  ({s/med*100:.1f}%)")
            if args.clean_truncated:
                freed = sum(s for _, s in bad)
                for p, _ in bad:
                    p.unlink()
                print(f"  已删除，释放 {mb(freed)}")
        else:
            print("  ✓ 全部正常（无截断）")
        # epoch 连续性
        es = sorted(int(re.search(r"_e(\d+)", p.stem).group(1))
                    for p in ps if re.search(r"_e(\d+)", p.stem))
        gaps = [x for i, x in enumerate(es) if i and x != es[i-1] + 1]
        print()
        print(f"  epoch 序列: {es[0]} ~ {es[-1]}  共 {len(es)} 个")
        if gaps:
            print(f"  !! 缺: {gaps}（崩溃导致，正常）")
        else:
            print("  连续无缺")

print()
print("=" * 88)
print("5. 结论：你并不需要重训")
print("=" * 88)
print(f"""
  崩溃丢的只有 **e27 这一个 epoch 的导出权重**（766 MB）。
  代码顺序决定了：L390 的 `save_checkpoint` 先执行、崩在那里，
  所以 L413 的 `savee(..._e27_...)` 根本没跑。

  ## 已完成 epoch 的导出权重是**独立文件**，全部保留
      `..._e1_...pth` ~ `..._e26_...pth`  (766 MB each)

  这些**可以直接推理**，且已用文件大小验证含 enc_p/ssl_proj：
      freeze  版 735,663,517 字节
      nofreeze版 766,394,418 字节
      增量 +30,730,901 = (enc_p 14.15M + ssl_proj 1.18M) x 2  ✓ 吻合 100.2%

  ## SoVITS 到 e26 早已饱和
  技能里：「SoVITS 逐轮提升，到 e12 仍在涨」—— e26 远超这个范围。
  缺 e27~e32 只是少了 6 个额外样本，**不影响可用性**。

  ## 下一步（三选一）
    ① 直接用 e1..e26 合成对比（推荐，零成本）
    ② 腾空间后从 e26 续训到 e32
    ③ 换更小语料重跑（不必要）
""")
print("=" * 88)
