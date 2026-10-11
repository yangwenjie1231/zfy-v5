#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练收尾：验证成果 + 加版本头 + 报告，一次做完。

## 用途
训练结束后跑一次，确认：
  1. 训练真的完成了（不是中途崩的）
  2. 导出的权重数量正确（e27~e32 = 6 个）
  3. 权重含 enc_p/ssl_proj（用文件大小，非侵入）
  4. checkpoint 保留策略生效（只留最新 2 个）
  5. 磁盘空间够不够后续操作
  6. 给权重加版本头（**必须**，否则推理当 v3 加载 24k BigVGAN）

## 用法
    python _tools/finalize_training.py              # 检查（不改盘）
    python _tools/finalize_training.py --fix-header # 检查 + 加版本头
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from statistics import median

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path("/mnt/workspace/zfy-v5-fullft-pkg")
if not (ROOT / "GPT_SoVITS").is_dir():
    root = Path.cwd()
    ROOT = root if (root / "GPT_SoVITS").is_dir() else root

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="SoVITS_weights_v5_full_e32_nofreeze")
ap.add_argument("--exp", default="logs/v5_full_e32_nofreeze/logs_s2_v5dev_full")
ap.add_argument("--log", default="train_nofreeze2.log")
ap.add_argument("--fix-header", action="store_true")
args = ap.parse_args()

WDIR = ROOT / args.dir
EDIR = ROOT / args.exp
LOG = ROOT / args.log

TOTAL_EPOCHS = 32
STEPS_PER_EPOCH = 54


def parse_epoch(stem: str) -> int:
    """从权重/ckpt 文件名里取 **真正的** epoch。

    ⚠️ 不能说 `re.search(r"_e(\\d+)_", stem)` —— 会踩配置名的坑：
        "v5_full_e32_nofreeze_e10_s540_l0"
                   ^^^^^ 先匹配到 _e32_（配置名里的），全部显示 32

    正确做法：锚定到 `_e{N}_s{step}_l{rank}` 这个完整模式。
        "v5_full_e32_nofreeze_e10_s540_l0"
                              ^^^^^^^ 精确命中
    兜底也按**最后一个** `_e{N}_` 匹配。
    """
    m = re.search(r"_e(\d+)_s\d+_l\d+", stem)
    if m:
        return int(m.group(1))
    m = re.search(r"_e(\d+)_merged", stem)
    if m:
        return int(m.group(1))
    hits = re.findall(r"_e(\d+)_", stem)
    return int(hits[-1]) if hits else 0


def ckpt_step(stem: str) -> int:
    """从 `G_1728.pth` 取 global_step。"""
    m = re.search(r"G_(\d+)", stem)
    return int(m.group(1)) if m else 0

print("=" * 90)
print("训练收尾检查")
print("=" * 90)

# ---------------------------------------------------------------- 1) 日志
print()
print("1) 训练是否正常完成")
print("-" * 90)
if LOG.exists():
    raw = LOG.read_bytes().decode("utf-8", errors="replace")
    lines = [l for l in re.split(r"[\r\n]+", raw) if l.strip()]
    starts = [int(m.group(1)) for l in lines
              for m in [re.search(r"====>\s*Epoch:\s*(\d+)", l)] if m]
    saves = [m.group(1) for l in lines
             for m in [re.search(r"saving ckpt\s+(\S+):Success", l)] if m]
    errs = [l.strip()[:100] for l in lines
            if "Disk quota exceeded" in l or "RuntimeError" in l
            or "Traceback" in l or "basic_ios" in l]
    print(f"  日志: {LOG.name}  ({len(lines)} 段)")
    if starts:
        print(f"  训练过的 epoch: {starts[0]} ~ {starts[-1]}  "
              f"（共 {len(set(starts))} 个）")
    print(f"  成功保存的权重: {len(saves)} 个")
    for s in saves[-8:]:
        print(f"    {s}")
    if errs:
        print(f"  !! 有 {len(errs)} 条错误:")
        for e in errs[-4:]:
            print(f"     {e}")
    else:
        print("  ✓ 无错误（无配额/崩溃）")
    done = 32 in starts and len(saves) >= 6
    print(f"  -> {'✓ 看起来正常完成' if done else '~ 需人工确认'}")
else:
    print(f"  找不到 {LOG}")

# ---------------------------------------------------------------- 2) 权重
print()
print("2) 导出的权重")
print("-" * 90)
if not WDIR.is_dir():
    print(f"  !! {WDIR} 不存在")
    ws = []
else:
    ws = sorted(WDIR.glob("*.pth"), key=lambda p: parse_epoch(p.stem))
    sizes = [p.stat().st_size for p in ws]
    if ws:
        med = int(median(sizes))
        print(f"  目录: {WDIR.name}")
        print(f"  数量: {len(ws)}   中位大小: {med:,} 字节")
        print()
        print(f"  {'epoch':>6s} {'大小':>14s} {'占中位':>7s}  文件")
        for p in ws:
            e = parse_epoch(p.stem)
            s = p.stat().st_size
            flag = "" if s > med * 0.9 else "  !! 偏小"
            print(f"  {e:6d} {s:14,} {s/med*100:6.1f}%  {p.name}{flag}")
        es = [parse_epoch(p.stem) for p in ws]
        miss = [e for e in range(min(es), max(es) + 1) if e not in es]
        print()
        print(f"  epoch 范围: {min(es)} ~ {max(es)}")
        if miss:
            print(f"  !! 缺失: {miss}")
        else:
            print("  ✓ 连续无缺")
    else:
        print(f"  !! {WDIR} 里没有 .pth")

# ---------------------------------------------------------------- 3) enc_p
print()
print("3) 权重是否含 enc_p/ssl_proj（用文件大小推算，非侵入）")
print("-" * 90)
# 已知数值（来自旧 freeze 版）
FREEZE_SIZE = 735_663_517
ENC_P, SSL_PROJ, BPP = 14_150_000, 1_180_000, 2
expected = (ENC_P + SSL_PROJ) * BPP
if ws:
    got = int(median([p.stat().st_size for p in ws]))
    diff = got - FREEZE_SIZE
    print(f"  freeze 版（旧，已知）: {FREEZE_SIZE:,} 字节")
    print(f"  本次权重中位        : {got:,} 字节")
    print(f"  增量                : {diff:+,} 字节 ({diff/1e6:+.2f} MB)")
    print(f"  预期增量            : {expected:,} 字节 "
          f"((enc_p {ENC_P/1e6:.2f}M + ssl_proj {SSL_PROJ/1e6:.2f}M) x {BPP})")
    ratio = diff / expected if expected else 0
    print(f"  吻合度              : {ratio*100:.2f}%")
    if 0.9 <= ratio <= 1.1:
        print("  ✓ **解冻生效** —— enc_p/ssl_proj 已在权重里")
    elif abs(diff) < expected * 0.3:
        print("  ✗ **解冻未生效**（增量接近 0）")
    else:
        print("  ~ 需用 verify_encp_in_weights.py 数键数确认")

# ---------------------------------------------------------------- 4) ckpt
print()
print("4) checkpoint 保留策略")
print("-" * 90)
if not EDIR.is_dir():
    print(f"  {EDIR} 不存在")
else:
    fs = sorted(EDIR.glob("G_*.pth"), key=lambda p: ckpt_step(p.stem))
    print(f"  数量: {len(fs)}  （ZFY_KEEP_CKPTS 默认 2）")
    for f in fs:
        n = ckpt_step(f.stem)
        ep = n // STEPS_PER_EPOCH
        print(f"    {f.name:28s} {f.stat().st_size/1e9:.2f} GB   "
              f"（约 e{ep}）")
    if len(fs) <= 2:
        print("  ✓ 保留策略生效")
    else:
        print(f"  !! 有 {len(fs)} 个，超过预期 2 个")
    if any("233333333333" in f.name for f in fs):
        print("  ~ 哨兵文件 G_233333333333.pth 还在（第 2 次保存后会被清掉）")

# ---------------------------------------------------------------- 5) 磁盘
print()
print("5) 磁盘占用")
print("-" * 90)
# ⚠️ df/statvfs 在网络挂载上报的是**整个 NAS 池**（1.0 PB），
#    不是本实例配额。配额满时的真实错误是
#        OSError: [Errno 122] Disk quota exceeded
#    所以这里只报各目录占用（那才是你能控制的），并提示配额判断方法。
try:
    _st = __import__("os").statvfs(str(ROOT))
    print(f"  statvfs 可用: {_st.f_bavail*_st.f_frsize/1e12:.2f} TB")
    print("  （**这个数字是 NAS 总池，不是你的配额** —— 配额满时会报")
    print("    'OSError: [Errno 122] Disk quota exceeded'，与 df 无关）")
except Exception as e:  # noqa: BLE001
    print(f"  statvfs 失败: {e}")
print()
print("  各目录实际占用:")


def du(p: Path) -> int:
    if p.is_file():
        try:
            return p.stat().st_size
        except OSError:
            return 0
    t = 0
    if p.is_dir():
        for f in p.rglob("*"):
            try:
                if f.is_file() and not f.is_symlink():
                    t += f.stat().st_size
            except OSError:
                pass
    return t


for name in (args.dir, args.exp, "logs/zfy_v5", "GPT_SoVITS", ".git",
             "out_sweep"):
    p = ROOT / name
    if p.exists():
        print(f"  {du(p)/1e9:8.2f} GB  {name}")

# ---------------------------------------------------------------- 6) 版本头
print()
print("6) 权重版本头")
print("-" * 90)
if ws:
    need = 0
    for p in ws:
        try:
            head = p.read_bytes()[:2]
        except OSError:
            continue
        if head == b"PK":
            need += 1
    print(f"  缺版本头的权重: {need} / {len(ws)}")
    if need == 0:
        print("  ✓ 全部有头")
    else:
        print("  ⚠️ 全量微调的权重**不带版本头**（上游只在 lora_rank 或 "
              "'Pro' 时写头）")
        print("     不加头 -> 推理会按文件大小判成 v3 -> 加载 24k BigVGAN "
              "-> 报 HFValidationError")
        if args.fix_header:
            print()
            print("  执行 fix_weight_header.py ...")
            r = subprocess.run(
                [sys.executable,
                 str(ROOT / "_tools" / "fix_weight_header.py"),
                 "--dir", args.dir, "--apply"],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", cwd=str(ROOT))
            for l in ((r.stdout or "") + (r.stderr or "")).split("\n")[-14:]:
                if l.strip():
                    print(f"    {l.strip()}")
        else:
            print("     加 --fix-header 执行")

print()
print("=" * 90)
print("下一步")
print("=" * 90)
print(f"""
  1. 加版本头（若上面显示"缺版本头"）：
       python _tools/finalize_training.py --fix-header

  2. 合成 e8 / e16 / e24 / e32 对比（验证是否饱和）：
       python _tools/cloud_synth_sweep.py \\
           --weights-dir {args.dir} --epochs 8,16,24,32 --no-v2pro

  3. 推送音频到远端，然后本机下载 + 客观评估：
       python _tools/eval_from_wav.py
""")
print("=" * 90)
