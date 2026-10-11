# -*- coding: utf-8 -*-
"""分析从云端拉回的 20 个 wav —— 验证 ssl_proj 修复。

## 判据
`让我想想。`（5 字）：
  · 缺 ssl_proj 时 = 11.40s（过度生成 281 token）
  · 修复后应约 1.12s（24 token）

## 语速基准
正常中文 3~5 字/秒。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

D = Path(r"D:\A\zfy-v5\dist\sweep2\out_sweep")

TEXTS = {
    "01": "让我想想。", "02": "好。",
    "03": "迎敌或是迎客，都得整装后再出发。",
    "04": "出门前，要不要先喝杯茶提提神？",
    "05": "嗯……我理一理。",
    "06": "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？",
    "07": "武陵城外有一片梨树林，每年春天都会开满白色的花。",
    "08": "她就在那阵花香里，只是他自己不知道。",
    "09": "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。",
    "10": "这件事我三点五十分再去处理。",
}

if not D.is_dir():
    print(f"  {D} 不存在")
    sys.exit(1)


def analyze(d: Path):
    out = []
    for w in sorted(d.glob("*.wav")):
        try:
            x, sr = sf.read(str(w), always_2d=True)
        except Exception as e:  # noqa: BLE001
            print(f"    {w.name}: 读取失败 {type(e).__name__}")
            continue
        x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
        dur = len(x) / sr
        key = w.stem.split("_")[0]
        n = len(TEXTS.get(key, ""))
        out.append({
            "name": w.name, "dur": dur, "n": n,
            "speed": n / dur if dur else 0,
            "ratio": dur / (n / 4.0) if n else 0,
            "peak": float(np.abs(x).max()),
        })
    return out


print("=" * 100)
print("A/B 对比：e16（未合并，缺 ssl_proj）vs e16_merged（修复后）")
print("=" * 100)

a = {r["name"].replace("_e16.wav", ""): r for r in analyze(D / "e16")}
b = {r["name"].replace("_e16_merged.wav", ""): r for r in analyze(D / "e16_merged")}

if not a:
    print("  e16/ 为空")
if not b:
    print("  e16_merged/ 为空")

print(f"\n  {'#':4s} {'字数':>4s} │ {'e16 时长':>9s} {'语速':>6s} "
      f"│ {'merged 时长':>11s} {'语速':>6s} {'提速':>6s}")
print("  " + "-" * 92)

rows = []
for k in sorted(set(a) | set(b)):
    ra, rb = a.get(k), b.get(k)
    if not (ra and rb):
        continue
    print(f"  {k:4s} {ra['n']:4d} │ {ra['dur']:8.2f}s {ra['speed']:6.2f} "
          f"│ {rb['dur']:10.2f}s {rb['speed']:6.2f} "
          f"{ra['dur']/rb['dur']:5.2f}x")
    rows.append((ra, rb))

if rows:
    da = sum(r[0]["dur"] for r in rows)
    db = sum(r[1]["dur"] for r in rows)
    na = sum(r[0]["n"] for r in rows)
    print()
    print("=" * 100)
    print("汇总")
    print("=" * 100)
    print(f"  总字数    : {na}")
    print(f"  总时长    : e16 {da:.2f}s  ->  merged {db:.2f}s   "
          f"（缩短 {(1-db/da)*100:.0f}%）")
    print(f"  平均语速  : e16 {na/da:.2f} 字/秒  ->  merged {na/db:.2f} 字/秒")
    print(f"  （正常中文 3~5 字/秒）")
    print()
    first = next((r for r in rows if r[0]["name"].startswith("01")), None)
    if first:
        print(f"  **「让我想想。」（01）**")
        print(f"     e16       : {first[0]['dur']:.2f}s  语速 {first[0]['speed']:.2f}")
        print(f"     merged    : {first[1]['dur']:.2f}s  语速 {first[1]['speed']:.2f}")
        if first[1]["dur"] < 3.0:
            print("     -> ✓ 时长已恢复正常，ssl_proj 修复**完全成功**")
        else:
            print("     -> ✗ 仍异常")

    print()
    print("  静音检查（峰值 < 0.01）:")
    for tag, d in (("e16", a), ("merged", b)):
        sil = [v["name"] for v in d.values() if v["peak"] < 0.01]
        print(f"     {tag:8s}: {sil if sil else '无'}")

print()
print("=" * 100)
print("文件清单")
print("=" * 100)
for sub in sorted(D.iterdir()):
    if sub.is_dir():
        n = len(list(sub.glob("*.wav")))
        print(f"  {sub.name:16s} {n:3d} 个 wav")
