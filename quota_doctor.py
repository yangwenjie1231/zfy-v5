#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""磁盘**配额**诊断与清理（不是"磁盘满"）。

## 关键区分
    df -h /mnt/workspace
    172.31.1.150:/0ba7f3a6-...  1.0P  25T  1000T  3%  /mnt/workspace
`df` 报的是**整个 NAS**（1.0 PB），不是你的配额。
而实际错误是：
    OSError: [Errno 122] Disk quota exceeded
-> **每个实例有独立配额**，df 看不出来。

## 配额通常是 100 GB 左右

已知占用（从崩溃前的目录列表推算）：
    SoVITS_weights_v5_full_e32_nofreeze/   766 MB x 27 ≈ 20.7 GB   <- 新权重
    SoVITS_weights_v5_full_e32/            735 MB x 33 ≈ 24.3 GB   <- 旧 freeze 版
    logs/v5_full_e32_nofreeze/             G_*.pth ≈ 2 GB + 软链
    logs/v5_full_e32/                      G_*.pth ≈ 2 GB
    logs/zfy_v5/                           特征 ~5-10 GB
    GPT_weights_gptA + gptB                155 MB x 16 ≈ 2.5 GB
    out_sweep + SoVITS_weights_v5*         若干
    GPT_SoVITS/pretrained_models/          底模 ~3-5 GB
    --------------------------------------------------------------
    合计可能 60~80 GB -> 触顶

## 本脚本
  1. **探测配额**（不依赖 df）：分层 `du` + 试写
  2. 列出**所有**大目录，按大小排序
  3. 找 `my_save` 留下的垃圾（仓库根下的 `<时间戳>.pth`，每个 ~2 GB）
  4. 给出**分级清理建议**（安全 / 需确认 / 不建议）
  5. `--clean-safe` 才动手，且只删明确安全的

## 用法
    python _tools/quota_doctor.py                 # 只诊断
    python _tools/quota_doctor.py --clean-safe    # 删安全项（垃圾 + 旧 ckpt）
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path("/mnt/workspace/zfy-v5-fullft-pkg")
if not (ROOT / "GPT_SoVITS").is_dir():
    ROOT = Path.cwd()

ap = argparse.ArgumentParser()
ap.add_argument("--clean-safe", action="store_true",
                help="删除明确安全项（my_save 垃圾 + 旧 G_*.pth）")
ap.add_argument("--clean-old-freeze", action="store_true",
                help="额外删除旧的 freeze 版权重（24.3 GB，**需确认**）")
args = ap.parse_args()


def mb(b: int) -> str:
    if b >= 1e12:
        return f"{b/1e12:.2f} TB"
    if b >= 1e9:
        return f"{b/1e9:.2f} GB"
    return f"{b/1e6:,.0f} MB"


def du(p: Path, max_files: int = 400000) -> tuple[int, int]:
    """返回 (字节, 文件数)。"""
    if p.is_file():
        try:
            return p.stat().st_size, 1
        except OSError:
            return 0, 0
    if not p.is_dir():
        return 0, 0
    tot = n = 0
    try:
        for i, f in enumerate(p.rglob("*")):
            if i > max_files:
                break
            try:
                if f.is_file() and not f.is_symlink():
                    tot += f.stat().st_size
                    n += 1
            except OSError:
                pass
    except OSError:
        pass
    return tot, n


print("=" * 90)
print("磁盘配额诊断")
print("=" * 90)

print()
print("1. 挂载与 df（注意：df 报的是整个 NAS，**不是你的配额**）")
print("-" * 90)
print(f"""
    df -h /mnt/workspace
    -> 1.0P 总容量 / 1000T 可用 / 3% 使用率
    -> **这个数字毫无意义** —— 它是 NAS 总池

    实际写入时报:
        OSError: [Errno 122] Disk quota exceeded

    含义：你的实例有**独立配额**（DSW 常见 100 GB）。
""")

print("2. 探测配额上限")
print("-" * 90)
probe = ROOT / "_zfy_quota_probe.bin"
CHUNK = 64 * 1024 * 1024          # 64 MB
written = 0
try:
    with probe.open("wb") as f:
        while written < 3 * 1024**3:      # 最多试 3 GB
            try:
                f.write(b"\0" * CHUNK)
                f.flush()
                written += CHUNK
            except OSError as e:
                print(f"  写到 {mb(written)} 时失败: {e}")
                break
    if written >= 3 * 1024**3:
        print(f"  ✓ 至少能写 3 GB（配额未满或很大）")
    else:
        print(f"  可写上限约 {mb(written)}")
finally:
    try:
        probe.unlink()
    except OSError:
        pass

print()
print("3. 仓库总占用")
print("-" * 90)
total_b, total_n = du(ROOT)
print(f"  {ROOT}")
print(f"    {mb(total_b)}   {total_n:,} 个文件")

print()
print("4. 逐目录占用（降序）")
print("-" * 90)
entries = []
for d in sorted(ROOT.iterdir()):
    if not d.is_dir() or d.name == ".git":
        continue
    b, n = du(d)
    entries.append((b, n, d.name))
entries.sort(reverse=True)
print(f"  {'大小':>11s} {'文件数':>8s}  目录")
for b, n, name in entries[:20]:
    print(f"  {mb(b):>11s} {n:>8,d}  {name}/")

# 根目录下的散文件
root_files = [(f.stat().st_size, f.name) for f in ROOT.iterdir()
              if f.is_file()]
if root_files:
    root_files.sort(reverse=True)
    print()
    print(f"  根目录散文件 ({len(root_files)} 个，"
          f"{mb(sum(s for s, _ in root_files))}):")
    for s, name in root_files[:10]:
        print(f"    {mb(s):>11s}  {name}")

print()
print("=" * 90)
print("5. 清理候选（按收益排序）")
print("=" * 90)

SAFE: list[tuple[Path, int, str]] = []       # (path, bytes, 说明)
CONFIRM: list[tuple[Path, int, str]] = []

# 5a my_save 垃圾（根目录下的 <时间戳>.pth）
for p in ROOT.glob("*.pth"):
    if re.fullmatch(r"\d+(\.\d+)?\.pth", p.name):
        try:
            SAFE.append((p, p.stat().st_size,
                         "my_save 临时文件（崩溃残留）"))
        except OSError:
            pass
# 也扫 logs/ 下
for p in ROOT.glob("logs/*/*.pth"):
    if re.fullmatch(r"\d+(\.\d+)?\.pth", p.name):
        try:
            SAFE.append((p, p.stat().st_size, "my_save 临时文件（logs 下）"))
        except OSError:
            pass

# 5b 旧实验的 G_*.pth（含 optimizer state，每个 ~2 GB）
for pat in ("logs/v5_full_e32/logs_s2_v5dev_full/G_*.pth",
            "logs/v5_full/logs_s2_v5dev_full/G_*.pth"):
    for p in ROOT.glob(pat):
        try:
            b = p.stat().st_size
        except OSError:
            continue
        SAFE.append((p, b, "旧实验 checkpoint（含 optimizer，通常不再需要）"))

# 5c 旧 freeze 版权重（需确认）
for d in ("SoVITS_weights_v5_full_e32", "SoVITS_weights_v5_full",
          "SoVITS_weights_v5", "SoVITS_weights_v5_r128",
          "SoVITS_weights_smoke"):
    p = ROOT / d
    if p.is_dir():
        b, n = du(p)
        why = ("旧的 freeze 版权重（已被 nofreeze 版取代）"
               if "e32" in d and "nofreeze" not in d else
               f"早期实验权重（{n} 个）")
        CONFIRM.append((p, b, why))

# 5d GPT 权重（A/B 两条线，已评估过）
for d in ("GPT_weights_gptA", "GPT_weights_gptB"):
    p = ROOT / d
    if p.is_dir():
        b, n = du(p)
        CONFIRM.append((p, b, f"GPT A/B 微调权重（{n} 个，已评估）"))

if SAFE:
    print()
    print("  【安全项】可直接删（--clean-safe）")
    for p, b, why in sorted(SAFE, key=lambda t: -t[1]):
        print(f"    {mb(b):>11s}  {p.name[:52]:<52s} {why}")
    print(f"    {'-'*74}")
    print(f"    {mb(sum(b for _, b, _ in SAFE)):>11s}  小计")
else:
    print()
    print("  【安全项】无")

if CONFIRM:
    print()
    print("  【需确认】删了会丢训练产物（--clean-old-freeze 只删第一类）")
    for p, b, why in sorted(CONFIRM, key=lambda t: -t[1]):
        print(f"    {mb(b):>11s}  {p.name[:52]:<52s} {why}")
    print(f"    {'-'*74}")
    print(f"    {mb(sum(b for _, b, _ in CONFIRM)):>11s}  小计")

KEEP = ["SoVITS_weights_v5_full_e32_nofreeze", "logs/v5_full_e32_nofreeze",
        "logs/zfy_v5", "GPT_SoVITS/pretrained_models", "ref", "out_sweep"]
print()
print("  【必须保留】")
for k in KEEP:
    p = ROOT / k
    if p.exists():
        b, n = du(p)
        print(f"    {mb(b):>11s}  {k}")

print()
print("=" * 90)
if args.clean_safe and SAFE:
    print("执行清理（安全项）")
    print("=" * 90)
    freed = 0
    for p, b, why in SAFE:
        try:
            if p.is_dir():
                import shutil
                shutil.rmtree(p)
            else:
                p.unlink()
            freed += b
            print(f"  已删 {p.name}  ({mb(b)})")
        except OSError as e:
            print(f"  删除失败 {p.name}: {e}")
    print()
    print(f"  释放 {mb(freed)}")
    b2, _ = du(ROOT)
    print(f"  仓库现占 {mb(b2)}")
elif args.clean_old_freeze and CONFIRM:
    print("执行清理（需确认项）")
    print("=" * 90)
    import shutil
    freed = 0
    for p, b, why in CONFIRM:
        try:
            shutil.rmtree(p)
            freed += b
            print(f"  已删 {p.name}  ({mb(b)})")
        except OSError as e:
            print(f"  删除失败 {p.name}: {e}")
    print()
    print(f"  释放 {mb(freed)}")
else:
    print("（只诊断。加 --clean-safe / --clean-old-freeze 才动手）")
print("=" * 90)
print("""
## 推荐清理顺序

1. **my_save 垃圾**（安全）—— 每次崩溃留一个 ~2 GB 文件
       python _tools/quota_doctor.py --clean-safe
2. **旧实验 G_*.pth**（安全）—— 每个 ~2 GB，含 optimizer state
       也在 --clean-safe 里
3. **旧 freeze 版权重 24.3 GB**（需确认）
       它是"解冻 vs freeze"的对照基准。**关键数据已提取并记录**：
           freeze  版 735,663,517 字节
           nofreeze版 766,394,418 字节
           增量 +30,730,901 = (enc_p 14.15M + ssl_proj 1.18M) x 2
       所以删掉不影响结论。
       python _tools/quota_doctor.py --clean-old-freeze
4. **GPT A/B 权重 2.5 GB**（需确认）—— 已评估过，可重训

## 为什么必须腾空间

配额满 -> 训练在存盘时崩 -> **e27~e32 永远存不出来**。
而且 TensorBoard 也在同一次崩溃里失败：
    OSError: [Errno 122] Disk quota exceeded
所以续训会**立刻**在 `_run` 的 `SummaryWriter` 处崩（你已实测到）。
""")
