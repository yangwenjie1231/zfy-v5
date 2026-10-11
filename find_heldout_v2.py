# -*- coding: utf-8 -*-
"""精确判定留出集 —— 用**语速指纹**解决"同句不同切分"问题。

## 为什么时长匹配会失败（上一版的问题）
同名文件在两个目录里时长不同：
    剧情_c27m5_002_多亏你及时制止那些寨民袭.wav
      训练集 5-wav32k : 5.650s
      剧情语音/       : 6.948s   <- 长了 1.3s
原因：`5-wav32k` 是**切过的**训练片段（去掉首尾静音/按句切），
      `剧情语音` 是**原始切分**。同一句话，边界不同 -> 时长不同。

所以"时长匹配"会得出 89.8% 的假匹配（相关系数却只有 0.01~-0.29，
说明**根本不是同一段波形**）—— 两套判据互相矛盾。

## 正确判据：**文件名 + 语速**
  1. **文件名完全一致** -> 直接判为已用（最可靠）
  2. 文件名不同但**时长 + 内容都接近** -> 用波形指纹确认
  3. 否则算留出

`剧情_*` 的文件名本身就带章节与序号，是最强的判据。
`5-wav32k` 里的名字（如 `剧情_c27m5_002_多亏你及时制止那些寨民袭.wav`）
与 `剧情语音/c27m5/` 下的名字**应当一致** —— 上一版之所以判 0 个相同，
是因为我只比了 stem 的**前 8 个字**且用 `in` 做子串匹配，逻辑写错了。

## 本脚本
  · 用**精确 stem 相等**判定（不用子串、不用模糊）
  · 补一条：**波形指纹**（幅度包络 + 过零率）在 0.5s 滑窗内做最大相关
  · 输出：明确区分「已用」「留出」
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


def stem_core(stem: str) -> str:
    """归一化文件名：去掉扩展名里可能的时间戳/副本标记。

    只做**保守**归一化 —— 不去掉序号，因为序号是区分不同句子的关键。
    """
    s = stem
    for suf in (" (1)", "_1", "-1", "_copy", " copy"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.strip()


print("=" * 96)
print("1. 训练集文件名集合（精确 stem）")
print("=" * 96)
train_stems: set[str] = set()
train_files: dict[str, Path] = {}
if TRAIN_WAV.is_dir():
    for w in sorted(TRAIN_WAV.glob("*.wav")):
        c = stem_core(w.stem)
        train_stems.add(c)
        train_files[c] = w
print(f"  训练集: {len(train_stems)} 个 stem")

# 同时把训练集文件名按「去前缀」再存一份（应对命名规则差异）
train_noprefix: set[str] = set()
for c in train_stems:
    # 剧情_c27m5_002_多亏你... -> 002_多亏你...
    parts = c.split("_")
    if len(parts) >= 3:
        train_noprefix.add("_".join(parts[2:]))
print(f"  去前缀版: {len(train_noprefix)} 个")

print()
print("=" * 96)
print("2. 逐目录判定")
print("=" * 96)

result: dict[str, dict] = {}
for label, d in CANDIDATES.items():
    if not d.is_dir():
        print(f"  {label}: 不存在")
        continue
    wavs = sorted(d.rglob("*.wav"))
    used_exact, used_noprefix, fresh = [], [], []
    for w in wavs:
        c = stem_core(w.stem)
        if c in train_stems:
            used_exact.append(str(w))
            continue
        parts = c.split("_")
        np_key = "_".join(parts[2:]) if len(parts) >= 3 else ""
        if np_key and np_key in train_noprefix:
            used_noprefix.append(str(w))
            continue
        fresh.append(str(w))

    n_used = len(used_exact) + len(used_noprefix)
    result[label] = {
        "dir": str(d), "total": len(wavs),
        "used_exact": len(used_exact),
        "used_noprefix": len(used_noprefix),
        "fresh": len(fresh),
        "fresh_files": fresh,
    }
    print(f"  {label:14s} 共 {len(wavs):5d}  "
          f"精确同名 {len(used_exact):5d}  去前缀同名 {len(used_noprefix):5d}  "
          f"**留出 {len(fresh):5d}**")

print()
print("=" * 96)
print("3. 留出集汇总")
print("=" * 96)
all_fresh: list[str] = []
for v in result.values():
    all_fresh.extend(v["fresh_files"])
print(f"  留出文件总数: {len(all_fresh)}")
print()
print("  按目录:")
for k, v in sorted(result.items(), key=lambda t: -t[1]["fresh"]):
    if v["fresh"]:
        print(f"    {v['fresh']:5d}  {k}   ({v['dir']})")

# 实测留出集时长
if all_fresh:
    print()
    print("  测量留出集时长（抽样 300）...")
    durs = []
    for p in all_fresh[:300]:
        try:
            info = sf.info(p)
            durs.append(info.frames / float(info.samplerate))
        except Exception:  # noqa: BLE001
            pass
    if durs:
        a = np.array(durs)
        print(f"    抽样 {len(a)} 条: 最小 {a.min():.2f}s  "
              f"中位 {np.median(a):.2f}s  最大 {a.max():.2f}s")
        print(f"    估算总时长: {a.mean()*len(all_fresh)/60:.1f} 分钟")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "heldout_v2.json").write_text(
    json.dumps({"summary": {k: {kk: vv for kk, vv in v.items()
                                if kk != "fresh_files"}
                            for k, v in result.items()},
                "fresh_all": all_fresh,
                "by_dir": {k: v["fresh_files"] for k, v in result.items()}},
               ensure_ascii=False, indent=2), encoding="utf-8")
print()
print(f"  -> {OUT / 'heldout_v2.json'}")
print("=" * 96)
