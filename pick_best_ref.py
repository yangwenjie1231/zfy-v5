# -*- coding: utf-8 -*-
"""全面筛选参考音频 —— 尾音干净 **且** 相似度高。

## 之前的问题
`pick_clean_ref.py` 只扫了 3 个目录（1895 个），**漏了**：
    · 庄方宜素材/12-音频与语音/语音/中文   <- **原始官方素材**
      （历史上相似度最优的 `006_更换武器` 就在这里，0.6459）
    · 庄方宜_TTS训练集/audio（951）
    · 语音合集_分条/音频_WAV48k（81）
    · 语音（43）

而且**只筛了尾音，没筛相似度** —— 可能挑出"尾音干净但不像庄方宜"的。

## 本脚本：两阶段筛选
    阶段 1  声学筛选（快，不用 GPU）
        时长 3~10s / 尾平坦 <0.20 / 尾/体 < +2dB / 静音尾 <0.6s
    阶段 2  **相似度筛选**（GPU）
        对阶段 1 的候选算说话人嵌入，与留出集目标比对
        取相似度 top-N

## 判据（两个都要好）
    · 尾平坦低    = 尾音干净（用户的问题）
    · 相似度高    = 音色像庄方宜（不能牺牲）

**综合评分** = 相似度 - 尾平坦 × 0.3
（相似度是主指标，尾音是次要修正）

## 输出
    dist/ref_best/
        report.txt
        top01~top08.wav   （复制出来便于试听）
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
FAKE = Path(r"D:\Users\A\Downloads\仿通行证")
OUT = V5 / "dist" / "ref_best"
HELD = V5 / "dist" / "heldout" / "heldout_clean.json"

# ---------- 全部音源（这次不漏）----------
SRC: list[tuple[str, Path]] = [
    ("干员语音", FAKE / "新素材" / "音频" / "终末地解包" / "干员语音"),
    ("剧情语音", FAKE / "新素材" / "音频" / "终末地解包" / "剧情语音"),
    ("训练集audio", FAKE / "新素材" / "音频" / "庄方宜_TTS训练集" / "audio"),
    ("语音合集48k", FAKE / "新素材" / "音频" / "语音合集_分条" / "音频_WAV48k"),
    ("语音", FAKE / "新素材" / "音频" / "语音"),
    ("素材_中文", FAKE / "庄方宜素材" / "12-音频与语音" / "语音" / "中文"),
    ("wav32k", GSV / "logs" / "zfy_v5" / "5-wav32k"),
]

MIN_DUR, MAX_DUR = 3.0, 10.0
MAX_FLAT = 0.20
MAX_TB = 2.0
MAX_SIL = 0.60
MIN_RMS = 0.01

print("=" * 104)
print("全面筛选参考音频（尾音干净 + 相似度高）")
print("=" * 104)
print(f"  声学门槛: 时长 {MIN_DUR}~{MAX_DUR}s  尾平坦 <{MAX_FLAT}  "
      f"尾/体 <{MAX_TB:+.1f}dB  静音尾 <{MAX_SIL}s")


def load(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return x, int(sr)


def acoustic(p: Path) -> dict | None:
    try:
        x, sr = load(p)
    except Exception:  # noqa: BLE001
        return None
    dur = len(x) / sr
    if not (MIN_DUR <= dur <= MAX_DUR):
        return None
    rms = float(np.sqrt((x ** 2).mean()))
    if rms < MIN_RMS:
        return None
    nf = 256
    m = (len(x) // nf) * nf
    if m < nf * 8:
        return None
    rmsf = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
    voiced = np.where(rmsf > rms * (10 ** (-45 / 20)))[0]
    if len(voiced) == 0:
        return None
    end = int((voiced[-1] + 1) * nf)
    sil_tail = (len(x) - end) / sr
    if sil_tail > MAX_SIL:
        return None
    n = int(0.4 * sr)
    tail = x[max(0, end - n):end]
    body = x[max(0, end // 2 - sr // 2):end // 2 + sr // 2]
    if len(tail) < n // 2 or len(body) < n // 2:
        return None
    tr = float(np.sqrt((tail ** 2).mean()))
    br = float(np.sqrt((body ** 2).mean())) + 1e-12
    tb = 20 * np.log10(tr / br + 1e-12)
    nn = min(len(tail), 2048)
    S = np.abs(np.fft.rfft(tail[:nn] * np.hanning(nn))) + 1e-12
    flat = float(np.exp(np.log(S).mean()) / S.mean())
    pk = float(np.abs(x).max())
    return {"dur": dur, "rms": rms, "tb": tb, "flat": flat,
            "sil": sil_tail, "peak": pk, "sr": sr}


# ---------- 阶段 1 ----------
stage1: list[tuple[str, Path, dict]] = []
seen: set[str] = set()
for label, d in SRC:
    if not d.is_dir():
        print(f"  (跳过) {label}: 目录不存在")
        continue
    ws = sorted(d.rglob("*.wav"))
    hit = 0
    for w in ws:
        key = w.stem[:60]
        if key in seen:
            continue
        seen.add(key)
        mt = acoustic(w)
        if mt is None:
            continue
        if (mt["flat"] < MAX_FLAT and mt["tb"] < MAX_TB
                and mt["peak"] < 0.99):
            stage1.append((label, w, mt))
            hit += 1
    print(f"  {label:<14s} 扫 {len(ws):5d} 个  ->  合格 {hit:3d}")

print()
print(f"  阶段 1 合格: **{len(stage1)} 个**")
if not stage1:
    print("  !! 无合格候选")
    sys.exit(1)

# ---------- 阶段 2：相似度 ----------
print()
print("=" * 104)
print("阶段 2：相似度筛选（GPU）")
print("=" * 104)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
sys.path[:0] = [str(GSV), str(GSV / "GPT_SoVITS"), str(GSV / "tools")]
os.chdir(str(GSV))
from GPT_SoVITS.sv import SV  # noqa: E402

sv = SV(DEV, is_half=False)
print(f"  device: {DEV}   ✓ SV 加载成功")

hd = json.loads(HELD.read_text(encoding="utf-8"))
targets = [Path(t["path"]) for t in hd.get("targets", [])
           if Path(t["path"]).exists()]
print(f"  留出集目标: {len(targets)} 条")


def emb(p: Path):
    """逐字复刻 eval_from_wav.py:195-206。"""
    try:
        import torchaudio
        w, sr = torchaudio.load(str(p))
        if w.shape[0] > 1:
            w = w.mean(0, keepdim=True)
        if sr != 16000:
            w = torchaudio.functional.resample(w, sr, 16000)
        w = w.squeeze(0)
        if w.shape[0] > 160000:
            w = w[:160000]
        e = sv.compute_embedding3(w.to(DEV).unsqueeze(0).float())
        return F.normalize(e.float(), dim=-1)
    except Exception:  # noqa: BLE001
        return None


T = [e for e in (emb(p) for p in targets) if e is not None]
T = torch.stack([t.reshape(-1) for t in T])
print(f"  有效目标嵌入: {len(T)}")
print(f"  待评候选: {len(stage1)}")

scored = []
for i, (label, w, mt) in enumerate(stage1):
    e = emb(w)
    if e is None:
        continue
    sim = float((T @ e.reshape(-1)).max().item())
    scored.append({"label": label, "path": w, "sim": sim, **mt})
    if (i + 1) % 50 == 0:
        print(f"    ...{i+1}/{len(stage1)}")

print(f"  完成: {len(scored)} 个有效")

# 综合评分
for s in scored:
    s["score"] = s["sim"] - s["flat"] * 0.3
scored.sort(key=lambda s: -s["score"])

# ---------- 报告 ----------
print()
print("=" * 104)
print(f"TOP 30（按 相似度 - 尾平坦×0.3 排序）")
print("=" * 104)
print(f"  {'#':>3s} {'来源':<12s} {'文件':<40s} {'相似度':>8s} "
      f"{'尾平坦':>8s} {'尾/体':>8s} {'时长':>6s} {'评分':>8s}")
print("  " + "-" * 100)
for i, s in enumerate(scored[:30], 1):
    print(f"  {i:3d} {s['label']:<12s} {s['path'].stem[:40]:<40s} "
          f"{s['sim']:8.4f} {s['flat']:8.4f} {s['tb']:+8.2f} "
          f"{s['dur']:5.2f}s {s['score']:8.4f}")

print()
print("=" * 104)
print("对照")
print("=" * 104)
REF = GSV / "ref" / "default.wav"
x, sr = load(REF)
print(f"  当前 ref/default.wav : 时长 {len(x)/sr:.2f}s")
mt = acoustic(REF)
if mt:
    print(f"    尾平坦 {mt['flat']:.4f}  尾/体 {mt['tb']:+.2f}dB  "
          f"静音尾 {mt['sil']:.2f}s")
else:
    # 超时长范围，手工算
    nf = 256
    m = (len(x) // nf) * nf
    rmsf = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
    rms = float(np.sqrt((x ** 2).mean()))
    voiced = np.where(rmsf > rms * (10 ** (-45 / 20)))[0]
    end = int((voiced[-1] + 1) * nf)
    n = int(0.4 * sr)
    tail = x[max(0, end - n):end]
    body = x[max(0, end // 2 - sr // 2):end // 2 + sr // 2]
    nn = min(len(tail), 2048)
    S = np.abs(np.fft.rfft(tail[:nn] * np.hanning(nn))) + 1e-12
    fl = float(np.exp(np.log(S).mean()) / S.mean())
    tr = float(np.sqrt((tail ** 2).mean()))
    br = float(np.sqrt((body ** 2).mean())) + 1e-12
    print(f"    尾平坦 {fl:.4f}  尾/体 {20*np.log10(tr/br+1e-12):+.2f}dB  "
          f"（**超 3~10s 范围**）")
    e = emb(REF)
    if e is not None:
        print(f"    相似度 {float((T @ e.reshape(-1)).max().item()):.4f}")

print(f"  真人录音典型: 尾平坦 0.1352  尾/体 -4.94dB")

# ---------- 导出 ----------
OUT.mkdir(parents=True, exist_ok=True)
for i, s in enumerate(scored[:8], 1):
    dst = OUT / f"top{i:02d}_{s['path'].stem[:44]}.wav"
    shutil.copy2(s["path"], dst)
print()
print(f"  前 8 名已复制到 {OUT}")

print()
print("=" * 104)
print("判读")
print("=" * 104)
if scored:
    b = scored[0]
    print(f"  最优: {b['path'].name}")
    print(f"    来源 {b['label']}   相似度 {b['sim']:.4f}   "
          f"尾平坦 {b['flat']:.4f}   尾/体 {b['tb']:+.2f}dB   "
          f"时长 {b['dur']:.2f}s")
    print()
    print("  ## 下一步")
    print("    1. 用 top01~top03 各合成 10 句（`compare_ref_swap.py` 改路径）")
    print("    2. 测相似度（`eval_dirs.py`）")
    print("    3. 做盲听包（`make_ref_listen.py` 改路径）")
    print()
    print("  ## ⚠️ 提醒")
    print("    · 本脚本的相似度是**参考音频自身**与留出集的相似度")
    print("    · 而**合成音频**的相似度可能不同（还受模型影响）")
    print("    · 所以仍需实际合成后复测")

rep = ["=" * 80, "参考音频全面筛选（尾音 + 相似度）", "=" * 80, ""]
rep.append(f"门槛: {MIN_DUR}~{MAX_DUR}s  平坦<{MAX_FLAT}  尾/体<{MAX_TB}  "
           f"静音尾<{MAX_SIL}")
rep.append(f"阶段1 合格: {len(stage1)}   阶段2 有效: {len(scored)}")
rep.append("")
rep.append(f"{'#':>3s} {'来源':<12s} {'文件':<42s} {'相似度':>8s} "
           f"{'尾平坦':>8s} {'尾/体':>8s}")
for i, s in enumerate(scored[:40], 1):
    rep.append(f"{i:3d} {s['label']:<12s} {s['path'].stem[:42]:<42s} "
               f"{s['sim']:8.4f} {s['flat']:8.4f} {s['tb']:+8.2f}")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
(OUT / "candidates.json").write_text(json.dumps(
    [{"label": s["label"], "path": str(s["path"]), "sim": s["sim"],
      "flat": s["flat"], "tb": s["tb"], "dur": s["dur"],
      "score": s["score"]} for s in scored[:100]],
    ensure_ascii=False, indent=2), encoding="utf-8")
print()
print(f"  -> {OUT / 'report.txt'}")
print("=" * 104)
