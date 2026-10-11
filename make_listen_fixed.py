# -*- coding: utf-8 -*-
"""生成修正后的盲听对比集（48kHz）。

已定位根因：**参考音频与 ref_text 不匹配**。
  我上一轮用的 `庄方宜_TTS训练集/ref_audio/ref.wav`：
    ref_text 写「多亏你及时制止那些寨民袭击武陵城…回赠一些物资。」
    实际念「那里常年监测着周边环境的深度毒数是抵抗秦世朝的第一线」
    前缀一致率 0%，语速 6.94 字/秒（异常），文件 4.32s（README 说 6.95s）
  → 模型去"补完"参考文本，表现为**串词 + 语气怪异**，两个模型都中招。

修正后用生产参考 `zfh-models-v2pro/ref/default.wav`（前缀一致 100%）。

本脚本产出：
  1. `AB_v5_vs_v2pro_修正.wav`  —— 每句先 A(v5) 后 B(v2Pro)，都用对齐参考
  2. `参考音频修复对比.wav`      —— 同一模型下 对齐 vs 错配（证明根因）
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
OUT = V5 / "out" / "listen_fixed"
REF = V5 / "out" / "ref_fix"
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
    texts = json.loads((REF / "ref_fix_metrics.json").read_text(encoding="utf-8"))

    made = []

    # 1) 修正后的 A/B
    a = sorted((REF / "P1_v5_goodref").glob("*.wav"))
    b = sorted((REF / "P3_v2pro_goodref").glob("*.wav"))
    if a and b:
        parts = []
        for x, y in zip(a, b):
            parts += [load(x), load(y)]
        full = concat(parts)
        p = OUT / "AB_v5_vs_v2pro_修正.wav"
        sf.write(str(p), full, SR, subtype="PCM_16")
        made.append((p.name, len(full) / SR, "每句先 A(v5) 后 B(v2Pro)，均用对齐参考"))

    # 2) 参考音频修复前后（用 v2Pro，最能说明问题）
    g = sorted((REF / "P3_v2pro_goodref").glob("*.wav"))
    bg = sorted((REF / "P4_v2pro_badref").glob("*.wav"))
    if g and bg:
        parts = []
        for x, y in zip(g, bg):
            parts += [load(x), load(y)]
        full = concat(parts)
        p = OUT / "参考音频修复对比.wav"
        sf.write(str(p), full, SR, subtype="PCM_16")
        made.append((p.name, len(full) / SR,
                     "每句先 对齐参考(正确) 后 错配参考(我上轮的bug)"))

    # 3) v5 的同对比
    g5 = sorted((REF / "P1_v5_goodref").glob("*.wav"))
    b5 = sorted((REF / "P2_v5_badref").glob("*.wav"))
    if g5 and b5:
        parts = []
        for x, y in zip(g5, b5):
            parts += [load(x), load(y)]
        full = concat(parts)
        p = OUT / "参考修复对比_v5.wav"
        sf.write(str(p), full, SR, subtype="PCM_16")
        made.append((p.name, len(full) / SR, "v5：先对齐 后错配"))

    (OUT / "说明.txt").write_text("""GPT-SoVITS v5 / v2Pro —— 修正参考音频后的对比
============================================================

【根因】我上一轮用错了参考音频
  数据集里的 ref_audio/ref.wav：
    · ref_text 声明「多亏你及时制止那些寨民袭击武陵城…回赠一些物资。」
    · 实际念的是「那里常年监测着周边环境的深度毒数是抵抗秦世朝的第一线」
    · 前缀一致率 0%，语速 6.94 字/秒（正常 3-4.5），文件 4.32s（README 写 6.95s）
  → 参考音频与文本不匹配时，模型会去"补完"参考文本，
    表现为**串词**和**语气怪异**。两个模型都受影响，不是 v5 独有的问题。

  生产用的 default.wav（006_更换武器）是对的：前缀一致 100%，语速 3.77 字/秒。

【修正后的客观指标】同 12 句、同 seed、同 GPT(zfh-e4)
  组                       CER     时长    字数比    SIM
  v5   + 对齐参考          8.6%   4.59s    0.93   0.8370
  v5   + 错配参考(上轮)    42.3%   5.25s    1.26   0.7644
  v2Pro+ 对齐参考         11.9%   4.76s    0.93   0.8957
  v2Pro+ 错配参考(上轮)    46.7%   5.67s    1.42   0.8622

  字数比 1.26/1.42 就是"多念了参考文本"的直接证据（应≈1.0）。

【三份音频】
  AB_v5_vs_v2pro_修正.wav   每句先 A(v5) 后 B(v2Pro)，都用对齐参考
                            这次才是真正公平的 v5 vs v2Pro 对比
  参考音频修复对比.wav        v2Pro：先 对齐(正确) 后 错配(我上轮的bug)
  参考修复对比_v5.wav         v5：先 对齐 后 错配

【建议先听】参考音频修复对比.wav —— 差异非常明显（串词 vs 正常）。
""", encoding="utf-8")

    print()
    for n, d, desc in made:
        print(f"  {n:30s} {d:6.1f}s  {desc}")
    print(f"\n输出: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
