# -*- coding: utf-8 -*-
"""生成 v5 vs v2Pro 的盲听对比音频（本轮的核心交付物）。

已有的合成结果在 `out/fair_compare/`（同 12 句、同参考、同 seed）：
    v5_e4 / v5_e7 / v5_e8 / v2pro

本脚本把它们拼成三份便于听的成品：
  1. `AB_v5_vs_v2pro.wav` —— 每句先 A(v5_e8) 后 B(v2Pro)，用于判断"哪个更好"
  2. `ABC_epochs.wav`     —— 同一句依次 e4 / e7 / e8，用于判断"哪个 epoch 更好"
  3. `长文本对比.wav`      —— 长句单独一份（长文本最容易暴露韵律/糊）

响度统一到同一 RMS，避免音量差干扰判断。

## 采样率必须是 48kHz（我第一版做错了）

第一版把盲听集输出成 **16kHz**（奈奎斯特 8kHz）—— 这会把
v5 的 16-24kHz 内容、以及 v5 与 v2Pro 在 8-16kHz 的差异**全部抹掉**，
等于把要对比的东西先删了。本版统一输出 **48kHz**：
  · v5 原生 48kHz，直通
  · v2Pro 是 32kHz，上采样到 48kHz —— **不会凭空产生 16kHz 以上内容**，
    所以 v2Pro 的 16k 硬上限依然真实存在，对比是公平的。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")
FAIR = V5 / "out" / "fair_compare"
OUT = V5 / "out" / "listen"
SR = 48000              # 必须 48k：否则抹掉 v5 的带宽优势
TARGET_RMS = 0.06


def load_resampled(p: Path, sr_out: int = SR) -> np.ndarray:
    """读成单声道并重采样到 sr_out。

    升采样用线性插值（v2Pro 32k->48k）**不会**制造新频带内容，
    所以 v2Pro 的 16kHz 上限保持真实。
    """
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != sr_out:
        k = int(round(len(x) * sr_out / sr))
        x = np.interp(np.linspace(0, len(x) - 1, k), np.arange(len(x)), x)
    x = x.astype(np.float32)
    r = float(np.sqrt(np.mean(x ** 2))) + 1e-9
    return x * (TARGET_RMS / r)


def concat(parts, gap=0.7):
    g = np.zeros(int(gap * SR), dtype=np.float32)
    head = np.zeros(int(0.5 * SR), dtype=np.float32)
    out = [head]
    for p in parts:
        out.append(p)
        out.append(g)
    full = np.concatenate(out)
    pk = float(np.max(np.abs(full)))
    if pk > 0.95:
        full = full * (0.95 / pk)
    return full


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    texts = json.loads((FAIR / "metrics.json").read_text(encoding="utf-8"))["texts"]
    tags = {}
    for t in ("v5_e4", "v5_e7", "v5_e8", "v2pro"):
        d = FAIR / t
        ps = sorted(d.glob("*.wav")) if d.exists() else []
        if ps:
            tags[t] = ps
    print(f"可用组: {list(tags)}  句子数={len(texts)}")

    made = []

    # ---- 1) AB：v5_e8 vs v2Pro ----
    if "v5_e8" in tags and "v2pro" in tags:
        parts = []
        for a, b in zip(tags["v5_e8"], tags["v2pro"]):
            parts += [load_resampled(a), load_resampled(b)]
        full = concat(parts)
        p = OUT / "AB_v5_vs_v2pro.wav"
        sf.write(str(p), full, SR, subtype="PCM_16")
        made.append(("AB_v5_vs_v2pro.wav", len(full) / SR,
                     "每句先 A(v5_e8) 后 B(v2Pro)"))

    # ---- 2) ABC：逐 epoch ----
    ep_tags = [t for t in ("v5_e4", "v5_e7", "v5_e8") if t in tags]
    if len(ep_tags) >= 2:
        parts = []
        for i in range(len(texts)):
            for t in ep_tags:
                if i < len(tags[t]):
                    parts.append(load_resampled(tags[t][i]))
        full = concat(parts)
        p = OUT / "ABC_epochs.wav"
        sf.write(str(p), full, SR, subtype="PCM_16")
        made.append(("ABC_epochs.wav", len(full) / SR,
                     "每句依次 " + " → ".join(ep_tags)))

    # ---- 3) 长文本单独一份 ----
    long_idx = [i for i, t in enumerate(texts) if len(t) >= 20]
    if long_idx and "v5_e8" in tags and "v2pro" in tags:
        parts = []
        for i in long_idx:
            parts += [load_resampled(tags["v5_e8"][i]), load_resampled(tags["v2pro"][i])]
        full = concat(parts)
        p = OUT / "长文本对比.wav"
        sf.write(str(p), full, SR, subtype="PCM_16")
        made.append(("长文本对比.wav", len(full) / SR,
                     f"{len(long_idx)} 个长句，先 A(v5) 后 B(v2Pro)"))

    # ---- 答案与说明 ----
    lines = [
        "GPT-SoVITS v5 微调  vs  v2Pro（当前生产）",
        "=" * 56,
        "",
        "【AB_v5_vs_v2pro.wav】每句先 A 后 B",
        "  A = v5dev + LoRA(rank32) 微调 e8，输出 48kHz",
        "  B = v2Pro 微调 e12（当前生产），输出 32kHz",
        "",
        "【ABC_epochs.wav】每句依次 " + " → ".join(ep_tags),
        "  用于判断哪个 epoch 更好（听感为主）",
        "",
        "【长文本对比.wav】只放长句，先 A(v5) 后 B(v2Pro)",
        "",
        "对比要点：",
        "  1) 哪个更像庄方宜（音色相似度）？",
        "  2) 哪个更清晰、不糊？",
        "  3) 哪个语调/停顿更自然？",
        "",
        "已知客观指标（12 句配对，同参考/同 seed）：",
        "  · 音色相似度(ERes2NetV2)：v2Pro 0.8295 > v5 0.8052（显著）",
        "  · 可懂度 CER：v5 12.7% ≈ v2Pro 14.6%（不显著）",
        "  · 带宽：v5 有效 ~22.8kHz（含真实 16-24k 谐波，包络相关 +0.99）",
        "          v2Pro 硬上限 16kHz",
        "  · 12-16kHz 绝对能量：v5 -33.8dB ≈ v2Pro -32.2dB",
        "",
        "句子清单:",
    ]
    for i, t in enumerate(texts):
        lines.append(f"  {i+1:2d}. {t}")
    (OUT / "说明.txt").write_text("\n".join(lines), encoding="utf-8")

    print()
    for name, dur, desc in made:
        print(f"  {name:26s} {dur:6.1f}s  {desc}")
    print(f"\n输出目录: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
