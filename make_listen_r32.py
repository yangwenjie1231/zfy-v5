# -*- coding: utf-8 -*-
"""生成 r32 各 epoch 的盲听集，用于人工选点。

## 为什么需要

`sweep_r32_all.py` 显示 r32 的 8 个 epoch SIM 极差只有 0.0202
（0.8217 ~ 0.8419），**没有明显胜出者** —— 这个波动在噪声量级，
指标无法可靠选点。

而用户反馈"v5 最佳的效果不错"，说明**听感才是可靠判据**。
所以把 8 个 epoch 都做成对比音频，让人耳来选。

产出：
  · `r32_全部epoch.wav`   —— 8 个 epoch 依次播放（每句一轮）
                            太长，改为**每句只放前 4 个 epoch**
  · `r32_前4.wav` / `r32_后4.wav` —— 分成两份，各含 4 个 epoch

用法：
    python make_listen_r32.py --n 12
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
SWEEP = V5 / "out" / "r32_sweep"
OUT = V5 / "out" / "listen_r32"
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
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tags = sorted([d.name for d in SWEEP.iterdir() if d.is_dir()])
    if not tags:
        print(f"没有可用的 epoch 目录: {SWEEP}")
        return 1
    print(f"epoch: {tags}")

    made = []
    # 分两组，每组 4 个 epoch，避免单个文件过长
    for gi, group in enumerate([tags[:4], tags[4:8]], 1):
        if len(group) < 2:
            continue
        files = {t: sorted((SWEEP / t).glob("*.wav"))[: args.n] for t in group}
        n = min(len(v) for v in files.values())
        parts = []
        for i in range(n):
            for t in group:
                parts.append(load(files[t][i]))
        full = concat(parts)
        name = f"r32_第{gi}组_{'_'.join(t.replace('r32_', '') for t in group)}.wav"
        p = OUT / name
        sf.write(str(p), full, SR, subtype="PCM_16")
        made.append((name, len(full) / SR,
                     "每句依次 " + " → ".join(t.replace("r32_", "") for t in group)))

    rows = json.loads((SWEEP / "sweep_r32.json").read_text(encoding="utf-8"))["rows"]
    lines = [
        "v5 LoRA r32 各 epoch 盲听选点",
        "=" * 56,
        "",
        "客观指标（20 句，同参考/同 GPT）：",
        f"  {'epoch':10s} {'CER':>8s} {'SIM':>9s}",
    ]
    for r in sorted(rows, key=lambda x: x["tag"]):
        lines.append(f"  {r['tag']:10s} {r['cer']*100:7.1f}% {r['sim']:9.4f}")
    sims = [r["sim"] for r in rows]
    lines += [
        "",
        f"  SIM 极差 {max(sims)-min(sims):.4f}（{min(sims):.4f} ~ {max(sims):.4f}）",
        "  → 波动在噪声量级，指标无法可靠选点，**请以听感为准**",
        "",
        "音频（48kHz，响度已对齐）：",
    ]
    lines += [f"  {n:44s} {d:6.1f}s  {desc}" for n, d, desc in made]
    lines += [
        "",
        "听的时候注意：",
        "  · 哪个的音色最像庄方宜",
        "  · 哪个的语气/停顿最自然",
        "  · 若听不出区别，说明 epoch 选择不重要，用任意一个即可",
    ]
    (OUT / "说明.txt").write_text("\n".join(lines), encoding="utf-8")

    print()
    for n, d, desc in made:
        print(f"  {n:44s} {d:6.1f}s  {desc}")
    print(f"\n输出: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
