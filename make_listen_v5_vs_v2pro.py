# -*- coding: utf-8 -*-
"""把三组音频做成盲听 A/B 包：v2Pro（生产）vs v5 e16_merged（全量微调修复版）。

## 为什么需要
云端那 20 个 wav 只有 e16 / e16_merged，**缺 v2Pro 基准** ——
而 v2Pro 是当前生产在用的，是唯一参照。已在本地用完全相同的参数生成。

## 三组来源
    A = v2Pro 生产版      D:\\A\\zfy-v5\\dist\\baseline_v2pro\\NN_*_v2pro.wav
    B = v5 全量 e16_merged D:\\A\\zfy-v5\\dist\\sweep2\\out_sweep\\e16_merged\\NN_*_e16_merged.wav
    C = v5 全量 e16（坏）  D:\\A\\zfy-v5\\dist\\sweep2\\out_sweep\\e16\\NN_*_e16.wav

C 只在第 1 句放（用来对比"缺 ssl_proj"的差异），其余只做 A/B。

## 盲听设计
  · 每句独立打乱 A/B 顺序（不是固定顺序）
  · 响度对齐到统一 RMS（否则"响的听起来更好"）
  · 句间留静音，便于分辨
  · 答案单独存 answer_key.txt

## 输出
    D:\\A\\zfy-v5\\dist\\listen_v5_vs_v2pro\\
        01_短_1.wav  01_短_2.wav
        02_短_1.wav  02_短_2.wav
        ...
        answer_key.txt   （听完再看）
        objective.txt    （时长/语速客观指标）
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
A_DIR = V5 / "dist" / "baseline_v2pro"                    # v2Pro
B_DIR = V5 / "dist" / "sweep2" / "out_sweep" / "e16_merged"  # v5 merged
C_DIR = V5 / "dist" / "sweep2" / "out_sweep" / "e16"         # v5 坏
OUT = V5 / "dist" / "listen_v5_vs_v2pro"

TARGET_SR = 48000
TARGET_RMS = 0.06          # 响度对齐目标（与 make_listen_any.py 一致）
HEAD_S = 0.5               # 句首静音
GAP_S = 0.7                # 句间静音
SEED = 20261011


def load_mono(p: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(p), always_2d=True)
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return x.astype(np.float64), sr


def resample_lin(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    """线性插值重采样（够用；不引入 scipy 依赖）。"""
    if sr == target:
        return x
    n = int(round(len(x) * target / sr))
    idx = np.linspace(0, len(x) - 1, n)
    return np.interp(idx, np.arange(len(x)), x)


def align_rms(x: np.ndarray, target: float = TARGET_RMS) -> np.ndarray:
    r = float(np.sqrt((x ** 2).mean()))
    if r < 1e-9:
        return x
    y = x * (target / r)
    peak = float(np.abs(y).max())
    if peak > 0.99:                      # 防削顶
        y = y * (0.99 / peak)
    return y


def find(d: Path, idx: str) -> Path | None:
    """按 NN_ 前缀找文件。"""
    if not d.is_dir():
        return None
    hits = sorted(d.glob(f"{idx}_*.wav"))
    return hits[0] if hits else None


def main() -> int:
    print("=" * 78)
    print("构造盲听 A/B 包：v2Pro（生产）vs v5 e16_merged（全量微调）")
    print("=" * 78)

    a_files = sorted(A_DIR.glob("*.wav")) if A_DIR.is_dir() else []
    b_files = sorted(B_DIR.glob("*.wav")) if B_DIR.is_dir() else []
    print(f"  v2Pro     : {len(a_files)} 个  ({A_DIR})")
    print(f"  v5 merged : {len(b_files)} 个  ({B_DIR})")
    if not a_files or not b_files:
        print("  !! 缺少来源，中止")
        return 1

    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    key_lines = ["# 盲听答案（听完再看）", ""]
    obj_lines = ["# 客观指标", ""]
    n_made = 0

    # 10 句，编号 01..10
    for i in range(1, 11):
        idx = f"{i:02d}"
        pa = find(A_DIR, idx)
        pb = find(B_DIR, idx)
        if not (pa and pb):
            print(f"  {idx}: 缺文件（v2pro={bool(pa)} merged={bool(pb)}）")
            continue

        # 读 + 重采样 + 响度对齐
        xa, sra = load_mono(pa)
        xb, srb = load_mono(pb)
        xa = align_rms(resample_lin(xa, sra, TARGET_SR))
        xb = align_rms(resample_lin(xb, srb, TARGET_SR))

        # 每句独立打乱
        items = [("v2pro", xa), ("v5merged", xb)]
        rng.shuffle(items)

        # 拼接：头静音 + [句1] + 间隔 + [句2]
        head = np.zeros(int(HEAD_S * TARGET_SR))
        gap = np.zeros(int(GAP_S * TARGET_SR))
        cat = np.concatenate([head, items[0][1], gap, items[1][1]])

        kind = pa.stem.split("_")[1] if "_" in pa.stem else "x"
        fn = OUT / f"{idx}_{kind}_AB.wav"
        sf.write(str(fn), cat, TARGET_SR)
        n_made += 1

        key_lines.append(
            f"{idx}_{kind}:  第1段 = {items[0][0]:9s}   "
            f"第2段 = {items[1][0]}")
        obj_lines.append(
            f"{idx}_{kind}:  v2pro {len(xa)/TARGET_SR:6.2f}s   "
            f"v5merged {len(xb)/TARGET_SR:6.2f}s")

    # 额外：第 1 句加一个三方对比（含坏版本）
    p_bad = find(C_DIR, "01")
    p_a = find(A_DIR, "01")
    p_b = find(B_DIR, "01")
    if p_bad and p_a and p_b:
        xa, sra = load_mono(p_a)
        xb, srb = load_mono(p_b)
        xc, src = load_mono(p_bad)
        xa = align_rms(resample_lin(xa, sra, TARGET_SR))
        xb = align_rms(resample_lin(xb, srb, TARGET_SR))
        xc = align_rms(resample_lin(xc, src, TARGET_SR))
        head = np.zeros(int(HEAD_S * TARGET_SR))
        gap = np.zeros(int(GAP_S * TARGET_SR))
        cat = np.concatenate([head, xa, gap, xb, gap, xc])
        sf.write(str(OUT / "01_短_三方对比_v2pro_v5merged_v5bad.wav"),
                 cat, TARGET_SR)
        key_lines.append("")
        key_lines.append("01_短_三方对比:  第1段 = v2pro   "
                         "第2段 = v5merged   第3段 = v5bad（缺 ssl_proj）")
        obj_lines.append(
            f"01 三方:  v2pro {len(xa)/TARGET_SR:.2f}s  "
            f"v5merged {len(xb)/TARGET_SR:.2f}s  "
            f"v5bad {len(xc)/TARGET_SR:.2f}s")

    (OUT / "answer_key.txt").write_text(
        "\n".join(key_lines) + "\n", encoding="utf-8")
    (OUT / "objective.txt").write_text(
        "\n".join(obj_lines) + "\n", encoding="utf-8")

    print()
    print(f"  已生成 {n_made} 个 A/B 文件 + 1 个三方对比")
    print(f"  输出目录: {OUT}")
    print()
    print("  用法：")
    print("    1. 用文件管理器打开上面的目录，按序号听")
    print("    2. 每个文件是「第1段 / 第2段」，听完判断哪段更像庄方宜")
    print("    3. 全部听完后打开 answer_key.txt 对答案")
    return 0


if __name__ == "__main__":
    sys.exit(main())
