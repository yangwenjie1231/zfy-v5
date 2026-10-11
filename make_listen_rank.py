# -*- coding: utf-8 -*-
"""生成 r32/r128/v2Pro 的盲听对比集（48kHz）。

三份：
  1. `AB_最佳v5_vs_v2Pro.wav`  —— 每句先 A(最佳 v5=r32_e8) 后 B(v2Pro)
  2. `ABC_秩对比.wav`           —— 同句依次 r32_e8 → r128_e20 → r128_e24
                                  （验证"提秩有没有用"）
  3. `说明.txt`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")
CMP = V5 / "out" / "r32_vs_r128"
OUT = V5 / "out" / "listen_retrain"
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
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads((CMP / "compare.json").read_text(encoding="utf-8"))["rows"]
    sims = {r["tag"]: r["sim"] for r in rows}

    def files(tag):
        return sorted((CMP / tag).glob("*.wav"))

    made = []
    v5tags = sorted([t for t in sims if t.startswith("r")],
                    key=lambda t: -sims[t])
    best = v5tags[0] if v5tags else None

    # 1) 最佳 v5 vs v2Pro
    if best:
        a, b = files(best), files("v2pro")
        n = min(len(a), len(b))
        parts = []
        for i in range(n):
            parts += [load(a[i]), load(b[i])]
        full = concat(parts)
        p = OUT / "AB_最佳v5_vs_v2Pro.wav"
        sf.write(str(p), full, SR, subtype="PCM_16")
        made.append((p.name, len(full) / SR,
                     f"每句先 A(最佳 v5 = {best}) 后 B(v2Pro)"))

    # 2) 秩对比
    pick = [t for t in ("r32_e8", "r128_e20", "r128_e24") if t in sims]
    if len(pick) >= 2:
        n = min(len(files(t)) for t in pick)
        parts = []
        for i in range(n):
            for t in pick:
                parts.append(load(files(t)[i]))
        full = concat(parts)
        p = OUT / "ABC_秩对比.wav"
        sf.write(str(p), full, SR, subtype="PCM_16")
        made.append((p.name, len(full) / SR,
                     "每句依次 " + " → ".join(pick)))

    lines = [
        "GPT-SoVITS v5 重训（LoRA rank128） vs r32 vs v2Pro",
        "=" * 60,
        "",
        "客观指标（20 句，同参考/同 GPT=zfh-e4）：",
        f"  {'配置':12s} {'CER':>8s} {'SIM':>9s}",
    ]
    for r in sorted(rows, key=lambda x: -x["sim"]):
        lines.append(f"  {r['tag']:12s} {r['cer']*100:7.1f}% {r['sim']:9.4f}")
    lines += [
        "",
        "结论：",
        "  · r32_e8 是 v5 里最好的（SIM 0.8236）",
        "  · r128 反而更差（0.8092~0.8208），配对检验显示提秩无收益",
        "  · v2Pro 全量微调 SIM 0.8888，显著高于任何 v5（t=-5.23）",
        "",
        "音频（48kHz，响度已对齐）：",
    ]
    lines += [f"  {n:28s} {d:6.1f}s  {desc}" for n, d, desc in made]
    lines += [
        "",
        "听的重点：",
        "  1) 最佳 v5 和 v2Pro 的音色差多少（SIM 差 0.065）",
        "  2) 三个秩之间听得出区别吗（若听不出，说明提秩确实无意义）",
    ]
    (OUT / "说明.txt").write_text("\n".join(lines), encoding="utf-8")

    print()
    for n, d, desc in made:
        print(f"  {n:28s} {d:6.1f}s  {desc}")
    print(f"\n输出: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
