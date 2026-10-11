# -*- coding: utf-8 -*-
"""r128 重训结果的盲听对比集（48kHz）。

产出四份：
  1. `AB_v5r128_vs_v2Pro.wav`   —— 每句先 A(v5 r128 最佳) 后 B(v2Pro)
  2. `ABC_epochs_r128.wav`      —— 同句依次若干 epoch，用于选点
  3. `对比_旧r32.wav`            —— r128 vs r32(旧)，看秩提升有没有用
  4. `说明.txt`

用法：
    python make_listen_retrain.py --epochs 8,16,24 --rank 128
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")
OUT = V5 / "out" / "listen_retrain"
EVAL = V5 / "out" / "retrain_eval"
OLD_EVAL = V5 / "out" / "ref_fix"          # 上一轮 rank32 的对齐参考结果
SR = 48000
TARGET_RMS = 0.06


def load(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        k = int(round(len(x) * SR / sr))
        x = np.interp(np.linspace(0, len(x) - 1, k), np.arange(len(x)), x)
    x = x.astype(np.float32)
    r = float(np.sqrt(np.mean(x ** 2))) + 1e-9
    return x * (TARGET_RMS / r)


def concat(parts, gap=0.7):
    g = np.zeros(int(gap * SR), dtype=np.float32)
    out = [np.zeros(int(0.5 * SR), dtype=np.float32)]
    for p in parts:
        out += [p, g]
    full = np.concatenate(out)
    pk = float(np.max(np.abs(full)))
    return full * (0.95 / pk) if pk > 0.95 else full


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--epochs", default="8,16,24")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    eps = [int(x) for x in args.epochs.split(",")]
    sets = {}
    for e in eps:
        d = EVAL / f"r{args.rank}_e{e:02d}"
        ps = sorted(d.glob("*.wav")) if d.exists() else []
        if ps:
            sets[f"r{args.rank}_e{e:02d}"] = ps
    v2 = sorted((EVAL / "v2pro").glob("*.wav"))
    old = sorted((OLD_EVAL / "P1_v5_goodref").glob("*.wav"))

    made = []

    # 1) A/B：最佳 rank128 vs v2Pro
    best = None
    ev = EVAL / "eval.json"
    if ev.exists():
        data = json.loads(ev.read_text(encoding="utf-8"))
        cand = [r for r in data["rows"]
                if r["tag"].startswith(f"r{args.rank}_")]
        if cand:
            # 综合：SIM 高 + CER 低
            best = max(cand, key=lambda r: r["sim"] - r["cer"])["tag"]
    if best is None and sets:
        best = sorted(sets)[-1]

    if best and v2:
        a = sets.get(best) or sorted((EVAL / best).glob("*.wav"))
        n = min(len(a), len(v2))
        parts = []
        for i in range(n):
            parts += [load(a[i]), load(v2[i])]
        full = concat(parts)
        p = OUT / "AB_v5r128_vs_v2Pro.wav"
        sf.write(str(p), full, SR, subtype="PCM_16")
        made.append((p.name, len(full) / SR,
                     f"每句先 A(v5 {best}) 后 B(v2Pro)"))

    # 2) ABC：逐 epoch
    if len(sets) >= 2:
        keys = sorted(sets)
        n = min(len(sets[k]) for k in keys)
        parts = []
        for i in range(n):
            for k in keys:
                parts.append(load(sets[k][i]))
        full = concat(parts)
        p = OUT / "ABC_epochs_r128.wav"
        sf.write(str(p), full, SR, subtype="PCM_16")
        made.append((p.name, len(full) / SR,
                     "每句依次 " + " → ".join(keys)))

    # 3) r128 vs r32（旧）
    if sets and old:
        k0 = sorted(sets)[-1]
        a = sets[k0]
        n = min(len(a), len(old))
        parts = []
        for i in range(n):
            parts += [load(a[i]), load(old[i])]
        full = concat(parts)
        p = OUT / "对比_新r128_vs_旧r32.wav"
        sf.write(str(p), full, SR, subtype="PCM_16")
        made.append((p.name, len(full) / SR,
                     f"每句先 新({k0}) 后 旧(rank32)"))

    (OUT / "说明.txt").write_text(f"""GPT-SoVITS v5 重训（LoRA rank {args.rank}）对比
============================================================

背景：已核实 v2/v2Pro 是**全量微调**（673/678 键，is_lora=False），
而 v5 只能走 LoRA（官方 s2_train_v3_lora.py，只微调 CFM 的
to_k/to_q/to_v/to_out.0）。所以把 LoRA 能力用满：
  rank 32 -> {args.rank}，epochs 8 -> 24

本轮已修正上一轮的两个错误：
  · 参考音频用生产 default.wav（已验证与 ref_text 一致）
    上一轮用的数据集 ref.wav 前缀一致率 0%、语速 6.94 字/秒，是错配的
  · GPT 统一用 zfh-e4.ckpt（生产同款），不是底模 s1v3

音频（均 48kHz，响度已对齐）：
""" + "\n".join(f"  {n:34s} {d:6.1f}s  {desc}" for n, d, desc in made) +
        "\n\n客观指标见 out/retrain_eval/eval.json\n", encoding="utf-8")

    print()
    for n, d, desc in made:
        print(f"  {n:34s} {d:6.1f}s  {desc}")
    if best:
        print(f"\n选用的最佳检查点: {best}")
    print(f"输出: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
