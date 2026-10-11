# -*- coding: utf-8 -*-
"""分析云端合成的 10 个 wav：时长是否恢复正常（验证 enc_p 修复）。

## 判据
`让我想想。` 5 个字：
  · 正常          约 1.2s
  · 缺 enc_p 时   11.40s（实测）

同时看物理指标：
  · 时长比值 = 实测 / (字数 / 4字每秒)，<0.6 视为崩
  · 语速 = 字数 / 时长，正常中文 3~5 字/秒
  · 尾音 RMS（最后 0.4s），判断是否被截断
  · 峰值 / 是否静音
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

D = Path(r"D:\A\zfy-v5\dist\sweep\out_sweep")

# 与 cloud_synth_sweep.py 的 TEXTS 对应
TEXTS = {
    "01": ("短", "让我想想。"),
    "02": ("短", "好。"),
    "03": ("中", "迎敌或是迎客，都得整装后再出发。"),
    "04": ("中", "出门前，要不要先喝杯茶提提神？"),
    "05": ("中", "嗯……我理一理。"),
    "06": ("长", "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？"),
    "07": ("长", "武陵城外有一片梨树林，每年春天都会开满白色的花。"),
    "08": ("长", "她就在那阵花香里，只是他自己不知道。"),
    "09": ("长", "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。"),
    "10": ("数字", "这件事我三点五十分再去处理。"),
}

wavs = sorted(D.rglob("*.wav"))
if not wavs:
    print(f"  在 {D} 下没找到 wav")
    sys.exit(1)

print("=" * 92)
print(f"分析 {len(wavs)} 个 wav")
print("=" * 92)
print(f"  {'文件':26s} {'时长':>7s} {'字数':>5s} {'语速':>6s} {'比值':>6s} "
      f"{'峰值':>7s} {'尾音RMS':>9s}")
print("  " + "-" * 88)

rows = []
for w in wavs:
    try:
        d, sr = sf.read(str(w), always_2d=True)
    except Exception as e:  # noqa: BLE001
        print(f"  {w.name:26s} 读取失败 {type(e).__name__}")
        continue
    if d.ndim == 2 and d.shape[1] > 1:
        d = d.mean(axis=1)
    else:
        d = d[:, 0]
    dur = len(d) / sr

    key = w.stem.split("_")[0]
    nchar = len(TEXTS.get(key, ("", ""))[1].replace("……", "…"))
    speed = nchar / dur if dur > 0 else 0
    ratio = dur / (nchar / 4.0) if nchar else 0
    peak = float(np.abs(d).max())
    tail = d[int(max(0, len(d) - 0.4 * sr)):]
    trms = float(np.sqrt((tail ** 2).mean())) if len(tail) else 0.0

    print(f"  {w.name:26s} {dur:6.2f}s {nchar:5d} {speed:5.2f} "
          f"{ratio:6.2f} {peak:7.3f} {trms:9.5f}")
    rows.append((w.name, dur, nchar, speed, ratio, peak, trms))

print()
print("=" * 92)
print("判定")
print("=" * 92)
if rows:
    durs = [r[1] for r in rows]
    ratios = [r[4] for r in rows]
    speeds = [r[3] for r in rows]
    print(f"  时长    : 最短 {min(durs):.2f}s  最长 {max(durs):.2f}s  "
          f"中位 {np.median(durs):.2f}s")
    print(f"  时长比值: 中位 {np.median(ratios):.2f}  "
          f"（正常应 0.8~1.3；<0.6 = 崩）")
    print(f"  语速    : 中位 {np.median(speeds):.2f} 字/秒  "
          f"（正常中文 3~5）")
    print()
    first = rows[0]
    print(f"  **「让我想想。」（01_短）: {first[1]:.2f}s**")
    print(f"     正常约 1.2s；缺 enc_p 时实测 11.40s")
    if first[1] < 3.0:
        print("     -> ✓ 时长已恢复正常，enc_p 修复**生效**")
    elif first[1] > 8.0:
        print("     -> ✗ 仍异常，enc_p 可能还没生效")
    else:
        print("     -> ? 介于两者之间，需人工听")

    print()
    print("  静音检查（峰值 < 0.01 视为静音）:")
    sil = [r[0] for r in rows if r[5] < 0.01]
    print(f"     {sil if sil else '无静音'}")

print()
print("=" * 92)
print("文件清单")
print("=" * 92)
for w in wavs:
    print(f"  {w}")
