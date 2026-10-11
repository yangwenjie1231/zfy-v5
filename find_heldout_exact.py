# -*- coding: utf-8 -*-
"""精确判定留出集 —— 用**音频内容**而不是文件名。

## 为什么不能用文件名
`find_heldout.py` 发现两个目录文件名**完全不重叠**但数量接近：
    D:\\A\\voiceclone\\data\\endfield\\wav24k        922 个  209.5 MB  平均 232.7 KB
    新素材\\音频\\终末地解包\\剧情语音                905 个  436.1 MB  平均 493.4 KB
同名交集 **0** —— 但平均大小差一倍，很可能是**同一批音频的两种导出**
（24k vs 48k，或不同编码）。若按文件名判，会把同一段音频当成"留出集"，
于是又变成"拿训练集素材当目标"的老错误。

## 正确做法：用**时长 + 波形指纹**
  1. 读每条 wav 的时长（秒）与降采样后的粗指纹
  2. 时长在 ±0.15s 内且指纹相关的，判为同一段音频
  3. 训练集的 825 条音频（logs/zfy_v5/5-wav32k）是"已用"的基准

## 为什么用时长
同一段语音无论 24k/32k/48k，**时长必然相同**（差 < 0.01s）。
时长是最便宜、最可靠的"同内容"判据。

## 输出
    D:\\A\\zfy-v5\\dist\\heldout\\heldout_exact.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
NEW = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频")
OUT = V5 / "dist" / "heldout"

TRAIN_WAV = GSV / "logs" / "zfy_v5" / "5-wav32k"
CANDIDATES = {
    "剧情语音": NEW / "终末地解包" / "剧情语音",
    "干员语音": NEW / "终末地解包" / "干员语音",
    "训练集audio": NEW / "庄方宜_TTS训练集" / "audio",
    "语音合集_分条": NEW / "语音合集_分条",
    "wav24k": Path(r"D:\A\voiceclone\data\endfield\wav24k"),
}
TOL = 0.15          # 时长容差（秒）
FINGER_SR = 2000    # 指纹降采样率


def probe(p: Path):
    """返回 (时长秒, 指纹数组) 或 None。指纹 = 降采样到 2kHz 的幅度包络。"""
    try:
        info = sf.info(str(p))
        dur = info.frames / float(info.samplerate)
        # 读前 8 秒用于指纹（足够区分不同句子）
        n = int(min(info.frames, info.samplerate * 8))
        x, sr = sf.read(str(p), frames=n, always_2d=True, dtype="float32")
        x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
        if len(x) < 100:
            return dur, None
        # 幅度包络（对相位不敏感，适合跨采样率比对）
        hop = max(1, sr // FINGER_SR)
        m = (len(x) // hop) * hop
        env = np.abs(x[:m]).reshape(-1, hop).max(axis=1)
        return dur, env.astype(np.float32)
    except Exception:  # noqa: BLE001
        return None, None


print("=" * 96)
print("1. 建立训练集的「时长指纹库」")
print("=" * 96)
train_durs = []
train_envs = []
if TRAIN_WAV.is_dir():
    tw = sorted(TRAIN_WAV.glob("*.wav"))
    for w in tw:
        d, e = probe(w)
        if d is None:
            continue
        train_durs.append(d)
        train_envs.append(e)
print(f"  训练集音频: {len(train_durs)} 条")
if train_durs:
    a = np.array(train_durs)
    print(f"  时长: 最小 {a.min():.2f}s  中位 {np.median(a):.2f}s  "
          f"最大 {a.max():.2f}s  合计 {a.sum()/60:.1f} 分钟")
train_durs = np.array(train_durs)

print()
print("=" * 96)
print("2. 逐目录判定：同内容（时长匹配）还是新素材")
print("=" * 96)

result: dict[str, dict] = {}
for label, d in CANDIDATES.items():
    if not d.is_dir():
        print(f"  {label}: 目录不存在 {d}")
        continue
    wavs = sorted(d.rglob("*.wav"))
    same, fresh = [], []
    for w in wavs:
        dur, env = probe(w)
        if dur is None:
            continue
        # 时长匹配（训练集内）
        if len(train_durs):
            near = np.abs(train_durs - dur) <= TOL
            if near.any():
                # 再做指纹相关确认（避免纯巧合）
                idxs = np.where(near)[0]
                ok = False
                for i in idxs[:3]:
                    te = train_envs[i]
                    if te is None or env is None:
                        ok = True
                        break
                    m = min(len(te), len(env))
                    if m < 20:
                        ok = True
                        break
                    c = np.corrcoef(te[:m], env[:m])[0, 1]
                    if c > 0.6:
                        ok = True
                        break
                if ok:
                    same.append((str(w), dur))
                    continue
        fresh.append((str(w), dur))
    result[label] = {
        "dir": str(d), "total": len(wavs),
        "same_as_train": len(same), "fresh": len(fresh),
        "fresh_files": [{"path": p, "dur": round(t, 3)} for p, t in fresh],
    }
    ft = sum(t for _, t in fresh)
    print(f"  {label:14s} 共 {len(wavs):5d}  "
          f"同内容 {len(same):5d}  **留出 {len(fresh):5d}**  "
          f"（{ft/60:.1f} 分钟）")

print()
print("=" * 96)
print("3. 留出集汇总")
print("=" * 96)
all_fresh = []
for k, v in result.items():
    all_fresh.extend(v["fresh_files"])
print(f"  留出文件总数: {len(all_fresh)}")
if all_fresh:
    tot = sum(f["dur"] for f in all_fresh)
    print(f"  总时长: {tot/60:.1f} 分钟")
    print()
    print("  按目录:")
    for k, v in sorted(result.items(), key=lambda t: -t[1]["fresh"]):
        if v["fresh"]:
            ft = sum(f["dur"] for f in v["fresh_files"])
            print(f"    {v['fresh']:5d} 个  {ft/60:6.1f} 分钟   {k}")
    print()
    print("  时长分布（留出集）:")
    ds = np.array([f["dur"] for f in all_fresh])
    for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 10), (10, 1e9)):
        n = int(((ds >= lo) & (ds < hi)).sum())
        print(f"    {lo:4.0f}~{hi if hi < 1e9 else '∞':>4} 秒 : {n:5d}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "heldout_exact.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print(f"  -> {OUT / 'heldout_exact.json'}")
print("=" * 96)
