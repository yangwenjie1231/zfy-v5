# -*- coding: utf-8 -*-
"""调试：为什么 剧情语音 与训练集的「同内容」判定是 0？

训练集有 745 条 `剧情_*` 命名的音频，它们**必然**来自
`终末地解包/剧情语音/`。所以时长匹配不可能是 0。

## 要查的三件事
  1. 训练集里 `剧情_*` 的时长，与 `剧情语音` 里的时长**有没有交集**
  2. 若有时长交集，指纹相关系数是多少（我的阈值 0.6 是否太高）
  3. 我的 probe() 是否对某些文件返回了 None

## 为什么这件事很关键
如果"留出集"里混着训练音频，那测出来的相似度是**虚高的背诵分**
（技能里明确写过：训练集内素材 0.91 vs 训练集外 0.67）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
TRAIN_WAV = GSV / "logs" / "zfy_v5" / "5-wav32k"
PLOT = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频\终末地解包\剧情语音")


def probe(p: Path):
    try:
        info = sf.info(str(p))
        dur = info.frames / float(info.samplerate)
        n = int(min(info.frames, info.samplerate * 8))
        x, sr = sf.read(str(p), frames=n, always_2d=True, dtype="float32")
        x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
        if len(x) < 100:
            return dur, None
        hop = max(1, sr // 2000)
        m = (len(x) // hop) * hop
        env = np.abs(x[:m]).reshape(-1, hop).max(axis=1)
        return dur, env.astype(np.float32)
    except Exception as e:  # noqa: BLE001
        print(f"    probe 失败 {p.name}: {type(e).__name__}: {e}")
        return None, None


print("=" * 88)
print("1. 训练集里 剧情_* 的时长")
print("=" * 88)
tw = sorted(TRAIN_WAV.glob("*.wav"))
print(f"  训练集总文件: {len(tw)}")

train_plot = [(w, *probe(w)) for w in tw if w.name.startswith("剧情_")]
train_plot = [(w, d, e) for w, d, e in train_plot if d is not None]
print(f"  其中 剧情_* : {len(train_plot)}")
if train_plot:
    ds = np.array([d for _, d, _ in train_plot])
    print(f"    时长: 最小 {ds.min():.2f}  中位 {np.median(ds):.2f}  "
          f"最大 {ds.max():.2f}")
    print(f"    样例:")
    for w, d, _ in train_plot[:4]:
        print(f"      {d:7.3f}s  {w.name[:66]}")
    print(f"    probe 返回 env=None 的: "
          f"{sum(1 for _, _, e in train_plot if e is None)}")

print()
print("=" * 88)
print("2. 剧情语音 目录的时长")
print("=" * 88)
pw = sorted(PLOT.rglob("*.wav"))
print(f"  文件数: {len(pw)}")
plot_list = []
for w in pw[:400]:            # 抽样 400 个够了
    d, e = probe(w)
    if d is not None:
        plot_list.append((w, d, e))
ds2 = np.array([d for _, d, _ in plot_list])
print(f"  抽样 {len(plot_list)} 个，时长: 最小 {ds2.min():.2f}  "
      f"中位 {np.median(ds2):.2f} 最大 {ds2.max():.2f}")

print()
print("=" * 88)
print("3. 时长交集检验（容差 0.15s）")
print("=" * 88)
td = np.array([d for _, d, _ in train_plot])
hits = 0
examples = []
for w, d, e in plot_list:
    near = np.where(np.abs(td - d) <= 0.15)[0]
    if len(near):
        hits += 1
        if len(examples) < 6:
            i = near[0]
            tw_name, tw_dur, tw_env = train_plot[i]
            c = None
            if tw_env is not None and e is not None:
                m = min(len(tw_env), len(e))
                if m >= 20:
                    c = float(np.corrcoef(tw_env[:m], e[:m])[0, 1])
            examples.append((w.name, d, tw_name.name, tw_dur, c))
print(f"  时长匹配上的: {hits} / {len(plot_list)}  "
      f"（{hits/max(1,len(plot_list))*100:.1f}%）")
print()
print(f"  {'剧情语音文件':<40s} {'时长':>7s}  {'训练集文件':<40s} {'时长':>7s} {'相关':>7s}")
for a, ad, b, bd, c in examples:
    cs = f"{c:7.3f}" if c is not None else "   None"
    print(f"  {a[:40]:<40s} {ad:7.3f}  {b[:40]:<40s} {bd:7.3f} {cs}")

print()
print("=" * 88)
print("4. 结论")
print("=" * 88)
if hits == 0:
    print("  !! 时长完全对不上 -> 说明两个目录**不是同一批音频**")
    print("     可能：剧情语音 是**重新切分**过的（切片边界不同 -> 时长不同）")
    print("     那么它们确实是独立素材，但需要确认内容是否重复")
else:
    print(f"  时长匹配率 {hits/len(plot_list)*100:.1f}%")
    cs = [c for *_, c in examples if c is not None]
    if cs:
        print(f"  指纹相关系数样例: {[round(x,3) for x in cs]}")
        print("  -> 相关系数高 = 同一段音频（我的阈值 0.6 是否合适看这里）")
print("=" * 88)
