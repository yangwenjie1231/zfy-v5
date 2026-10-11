#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""看训练进度 —— 正确解析 tqdm 的 `\\r` 进度条。

## 为什么 `tail -3` 看不清
tqdm 用 `\\r`（回车，不带换行）原地刷新，所以：
  · 整个 epoch 的进度条在文件里是**一行的多个版本**
  · `tail -3` 会显示超长的一行，且中间混着旧版本
  · 且因为它不写 `\\n`，日志的"最后一行"可能是个未完成的进度条

## 本脚本
  1. 按 `\\r` 和 `\\n` 都当作行分隔（还原 tqdm 的真实显示）
  2. 提取：当前 epoch、步数、耗时、ETA
  3. 提取：已完成的 epoch 数、导出的权重文件数
  4. 提取：loss 历史
  5. 检查：配额错误、崩溃

## 用法
    python _tools/progress.py
    python _tools/progress.py --log /path/to/train.log
    python _tools/progress.py --watch        # 每 10 秒刷新（Ctrl-C 退出）
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path("/mnt/workspace/zfy-v5-fullft-pkg")
if not (ROOT / "GPT_SoVITS").is_dir():
    ROOT = Path.cwd()

ap = argparse.ArgumentParser()
ap.add_argument("--log", default=None)
ap.add_argument("--watch", action="store_true")
ap.add_argument("--interval", type=int, default=10)
args = ap.parse_args()

# 自动找日志
LOG = Path(args.log) if args.log else None
if LOG is None:
    for cand in ("train_nofreeze2.log", "train_nofreeze.log"):
        p = ROOT / cand
        if p.exists():
            LOG = p
            break
if LOG is None or not LOG.exists():
    print(f"  找不到日志（试过 {ROOT}/train_nofreeze*.log）")
    sys.exit(1)

TOTAL_EPOCHS = 32
STEPS_PER_EPOCH = 54


def parse():
    raw = LOG.read_bytes().decode("utf-8", errors="replace")
    # ⚠️ tqdm 用 \r 刷新 -> 把 \r 也当行分隔，才能还原每次刷新
    lines = [l for l in re.split(r"[\r\n]+", raw) if l.strip()]

    cur_epoch = None
    steps_done = 0
    eta = ""
    speed = ""
    last_loss = None
    losses = []          # [(epoch, loss)]
    completed = []       # 完成的 epoch
    errors = []

    re_epoch_start = re.compile(r"====>\s*Epoch:\s*(\d+)")
    re_train_line = re.compile(r"Train Epoch:\s*(\d+)\s*\[(\d+)%\]")
    re_bar = re.compile(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^\]]*)\]")
    re_loss = re.compile(r"\[([\d.]+),\s*(\d+),\s*([\d.eE+-]+)\]")
    re_saved = re.compile(r"saving ckpt\s+(\S+):Success")

    for l in lines:
        # ⚠️ 关键：**不要**在匹配到某类后就 `continue`。
        #    tqdm 的进度条与 INFO 日志常常在**同一个 segment** 里
        #    （因为 tqdm 只写 \r 不写 \n）：
        #        100%|##| 54/54 [01:18<00:00, 1.46s/it]INFO:...saving ckpt ..._e32_...:Success.
        #    我第一版在 bar 匹配后 continue -> 永远看不到 saving ckpt
        #    -> "已完成 epoch" 恒为空。
        matched_bar = False

        m = re_epoch_start.search(l)
        if m:
            cur_epoch = int(m.group(1))
            steps_done = 0

        m = re_train_line.search(l)
        if m:
            cur_epoch = int(m.group(1))

        m = re_bar.search(l)
        if m:
            steps_done = int(m.group(2))
            inside = m.group(4)
            mm = re.search(r"<([\d:]+)", inside)
            if mm:
                eta = mm.group(1)
            mm = re.search(r"([\d.]+)(s/it|it/s)", inside)
            if mm:
                speed = mm.group(0)
            matched_bar = True

        m = re_loss.search(l)
        if m:
            try:
                last_loss = float(m.group(1))
                losses.append((cur_epoch, last_loss))
            except ValueError:
                pass

        m = re_saved.search(l)
        if m:
            # 名字形如 v5_full_e32_nofreeze_e32_s1728_l0
            # 用 `_e(\d+)_s\d+_l\d+` 精确定位，否则会先匹配到配置名里的 e32
            mm = re.search(r"_e(\d+)_s\d+_l\d+", m.group(1))
            if not mm:
                mm = re.search(r"_e(\d+)_merged", m.group(1))
            if mm:
                completed.append(int(mm.group(1)))

        if not matched_bar:
            if "Disk quota exceeded" in l or "Exit 1" in l:
                errors.append(l.strip()[:110])
            if "RuntimeError" in l or "Traceback" in l:
                errors.append(l.strip()[:110])

    running = False
    if "python -s GPT_SoVITS/s2_train_v5_full.py" in raw.split("\n")[0] or True:
        # 用进程判断更可靠
        try:
            import subprocess
            r = subprocess.run(["pgrep", "-fc", "s2_train_v5_full"],
                               capture_output=True, text=True, timeout=10)
            running = int((r.stdout or "0").strip() or 0) > 0
        except Exception:  # noqa: BLE001
            running = "start training from epoch" in raw and not errors

    return {
        "cur_epoch": cur_epoch, "steps": steps_done, "eta": eta,
        "speed": speed, "last_loss": last_loss, "losses": losses,
        "completed": sorted(set(completed)), "errors": errors[-5:],
        "running": running,
    }


def render():
    d = parse()
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 78)
    print(f"训练进度   {LOG}")
    print("=" * 78)

    total_steps = TOTAL_EPOCHS * STEPS_PER_EPOCH
    ce = d["cur_epoch"]
    if ce:
        done_steps = (ce - 1) * STEPS_PER_EPOCH + d["steps"]
        pct = done_steps / total_steps * 100
        print(f"  当前 epoch : {ce} / {TOTAL_EPOCHS}")
        print(f"  本 epoch   : {d['steps']} / {STEPS_PER_EPOCH} 步")
        print(f"  总进度     : {done_steps} / {total_steps} 步  ({pct:.1f}%)")
        bar_w = 44
        filled = int(pct / 100 * bar_w)
        print(f"  [{'#'*filled}{'.'*(bar_w-filled)}]")
        if d["speed"]:
            print(f"  速度       : {d['speed']}")
        if d["eta"]:
            print(f"  本 epoch ETA: {d['eta']}")
    else:
        print("  还没开始训练（或日志格式变了）")

    print()
    print(f"  进程运行中 : {'是 ✓' if d['running'] else '否 / 未知'}")

    if d["last_loss"] is not None:
        print(f"  最新 loss  : {d['last_loss']}")
        if len(d["losses"]) >= 2:
            tail = d["losses"][-6:]
            print(f"  最近 loss  : "
                  f"{' -> '.join(f'{v:.4f}' for _, v in tail)}")
            first, last = d["losses"][0][1], d["losses"][-1][1]
            print(f"  趋势       : {first:.4f} -> {last:.4f}  "
                  f"({'下降 ✓' if last < first else '上升 / 持平'})")

    c = d["completed"]
    print()
    if c:
        print(f"  已完成 epoch: {len(c)} 个  ({c[0]}~{c[-1]})")
        missing = [e for e in range(c[0], c[-1] + 1) if e not in c]
        if missing:
            print(f"  !! 缺失      : {missing}")
        print(f"  实际权重文件: "
              f"{'缺' if not (ROOT / 'SoVITS_weights_v5_full_e32_nofreeze').is_dir() else len(list((ROOT / 'SoVITS_weights_v5_full_e32_nofreeze').glob('*.pth')))} 个")
    else:
        print("  还没有完成的 epoch 权重")

    # checkpoint 目录
    ck = ROOT / "logs/v5_full_e32_nofreeze/logs_s2_v5dev_full"
    if ck.is_dir():
        fs = sorted(ck.glob("G_*.pth"))
        tot = sum(f.stat().st_size for f in fs)
        print()
        print(f"  checkpoint : {len(fs)} 个  ({tot/1e9:.2f} GB)")
        for f in fs:
            print(f"    {f.name:28s} {f.stat().st_size/1e9:.2f} GB")

    if d["errors"]:
        print()
        print("  !! 日志里有错误:")
        for e in d["errors"]:
            print(f"     {e}")

    print()
    print("=" * 78)


if args.watch:
    try:
        while True:
            render()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n（退出监控）")
else:
    render()
