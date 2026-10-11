# -*- coding: utf-8 -*-
"""确定**干净的留出集** —— 逐条给出排除证据。

## 结论（基于文件名映射）
训练集 `logs/zfy_v5/5-wav32k/`（825 条）的来源已查明：

| 候选目录 | 与训练集的关系 | 能否当留出集 |
|---|---|---|
| `终末地解包/剧情语音` | 745 条**精确同名**（其余 144 条是真新素材） | ✅ 取 144 |
| `终末地解包/干员语音` | 75 条精确同名（其余 90 条新） | ✅ 取 90 |
| `庄方宜_TTS训练集/audio` | 825 条精确同名（其余 123 条新） | ✅ 取 123 |
| `data/endfield/wav24k` | **命名可映射**：`c27m5_au_dlg_c27m5_9_002` = `剧情_c27m5_002_*` | ❌ 排除 |
| `语音合集_分条` | **命名可映射**：`002_行动准备1` = 训练集 `02_行动准备1_chr_*` | ❌ 排除 |

**干净留出集 = 144 + 90 + 123 = 357 条**（约 12 分钟）

## 为什么必须这么严格
技能里的铁律：
    训练集内素材当目标 = 0.91（虚高，只是背下了训练录音）
    训练集外同角色素材 = 0.67（真实水平）
    天然上限（真实录音 vs 真实录音）= 0.66

若把训练素材混进留出集，测出的是"背诵分"，会**误导后续所有决策**。

## 输出
    D:\\A\\zfy-v5\\dist\\heldout\\heldout_clean.json
    D:\\A\\zfy-v5\\dist\\heldout\\heldout_clean.txt   （路径列表，供其他脚本读）
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
NEW = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频")
OUT = V5 / "dist" / "heldout"
TRAIN_WAV = GSV / "logs" / "zfy_v5" / "5-wav32k"

# 只保留这三个（命名可精确排除）
CLEAN_DIRS = {
    "剧情语音": NEW / "终末地解包" / "剧情语音",
    "干员语音": NEW / "终末地解包" / "干员语音",
    "训练集audio": NEW / "庄方宜_TTS训练集" / "audio",
}
# 排除（已证明是训练素材来源）
EXCLUDED = {
    "wav24k": "命名可映射到训练集（c27m5_au_dlg_c27m5_9_002 = 剧情_c27m5_002_*）",
    "语音合集_分条": "命名可映射到训练集（002_行动准备1 = 02_行动准备1_chr_*）",
}


def stem_core(s: str) -> str:
    for suf in (" (1)", "_1", "-1", "_copy", " copy"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.strip()


print("=" * 92)
print("1. 训练集 stem 集合")
print("=" * 92)
train_stems: set[str] = set()
train_noprefix: set[str] = set()
for w in sorted(TRAIN_WAV.glob("*.wav")):
    c = stem_core(w.stem)
    train_stems.add(c)
    parts = c.split("_")
    if len(parts) >= 3:
        train_noprefix.add("_".join(parts[2:]))
print(f"  精确 stem : {len(train_stems)}")
print(f"  去前缀版  : {len(train_noprefix)}")

print()
print("=" * 92)
print("2. 逐目录筛出干净留出集")
print("=" * 92)

clean: dict[str, list[dict]] = {}
for label, d in CLEAN_DIRS.items():
    if not d.is_dir():
        print(f"  {label}: 不存在")
        continue
    wavs = sorted(d.rglob("*.wav"))
    fresh = []
    for w in wavs:
        c = stem_core(w.stem)
        if c in train_stems:
            continue
        parts = c.split("_")
        if len(parts) >= 3 and "_".join(parts[2:]) in train_noprefix:
            continue
        try:
            info = sf.info(str(w))
            dur = info.frames / float(info.samplerate)
            sr = info.samplerate
        except Exception:  # noqa: BLE001
            continue
        # 只保留 1.5~12 秒的（太短音色不足，太长浪费）
        if not (1.5 <= dur <= 12.0):
            continue
        fresh.append({"path": str(w), "dur": round(dur, 3), "sr": sr})
    clean[label] = fresh
    tot = sum(f["dur"] for f in fresh)
    print(f"  {label:14s} 共 {len(wavs):5d}  ->  **干净留出 {len(fresh):4d}**  "
          f"（{tot/60:.1f} 分钟）")

print()
print("=" * 92)
print("3. 排除的目录及理由")
print("=" * 92)
for k, why in EXCLUDED.items():
    print(f"  ✗ {k}")
    print(f"      {why}")

all_clean = [f for v in clean.values() for f in v]
print()
print("=" * 92)
print("4. 留出集汇总")
print("=" * 92)
print(f"  总条数: {len(all_clean)}")
if all_clean:
    ds = np.array([f["dur"] for f in all_clean])
    print(f"  总时长: {ds.sum()/60:.1f} 分钟")
    print(f"  时长分布: 最小 {ds.min():.2f}s  中位 {np.median(ds):.2f}s  "
          f"最大 {ds.max():.2f}s")
    print()
    print("  按时长区间:")
    for lo, hi in ((1.5, 2), (2, 3), (3, 4), (4, 6), (6, 8), (8, 12.01)):
        n = int(((ds >= lo) & (ds < hi)).sum())
        print(f"    {lo:4.1f}~{hi:4.1f}s : {n:4d}")
    print()
    print("  采样率分布:")
    from collections import Counter
    for sr, n in Counter(f["sr"] for f in all_clean).most_common():
        print(f"    {sr} Hz : {n:4d}")

# 选一个「留出集目标」子集：3~8 秒、采样率 ≥32k、共 24 条
targets = [f for f in all_clean if 3.0 <= f["dur"] <= 8.0 and f["sr"] >= 32000]
targets.sort(key=lambda f: (f["dur"], f["path"]))
print()
print(f"  适合做「音色目标」的子集（3~8s, >=32kHz）: {len(targets)} 条")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "heldout_clean.json").write_text(
    json.dumps({"by_dir": clean, "all": all_clean,
                "excluded": EXCLUDED,
                "targets": targets[:60]},
               ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "heldout_clean.txt").write_text(
    "\n".join(f["path"] for f in all_clean) + "\n", encoding="utf-8")
(OUT / "heldout_targets.txt").write_text(
    "\n".join(f["path"] for f in targets[:60]) + "\n", encoding="utf-8")
print()
print(f"  -> {OUT / 'heldout_clean.json'}")
print(f"  -> {OUT / 'heldout_clean.txt'}  （{len(all_clean)} 条）")
print(f"  -> {OUT / 'heldout_targets.txt'}  （{len(targets[:60])} 条目标）")
print("=" * 92)
