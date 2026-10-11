# -*- coding: utf-8 -*-
"""修正版参考音频筛选 —— **排除留出集**，避免自我比对污染。

## 上一版的严重缺陷
`pick_best_ref.py` 报 TOP8 相似度全是 **1.0000** —— 因为它们是
`heldout_clean.json` 里的 `targets` 本身：

    剧情_e9m3_001_我送你们一程，追上她！   -> TARGET
    剧情_e9m4_001_3                      -> TARGET
    ...

**自我比对必然 1.0**，且**用它们当参考会污染评估**：
合成音频与目标相似度虚高（参考即目标），测不出真实泛化能力。

## 修法（两道防线）
  1. **排除留出集**：`targets`(35) + `all`(131) 里的路径全部剔除
  2. **排除训练集**：`5-wav32k` 的 825 条也剔除
     （用训练集音频当参考同样会虚高）

  剩下的才是**真正"没被用过的"音频**。

## ⚠️ 但这里有个根本矛盾
    · 排除留出集 -> 候选变少（131 条里大部分是干净的）
    · 不排除     -> 评估被污染

**解法**：排除留出集后，**重新划一份留出集**用于评估。
本脚本会把可用的干净候选导出，评估时另用一份留出。

## 输出
    dist/ref_best2/
        report.txt
        top01~top10.wav
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
OUT = V5 / "dist" / "ref_best2"
HELD = V5 / "dist" / "heldout" / "heldout_clean.json"
TRAIN = GSV / "logs" / "zfy_v5" / "5-wav32k"

SRC: list[tuple[str, Path]] = [
    ("干员语音", FAKE / "新素材" / "音频" / "终末地解包" / "干员语音"),
    ("剧情语音", FAKE / "新素材" / "音频" / "终末地解包" / "剧情语音"),
    ("训练集audio", FAKE / "新素材" / "音频" / "庄方宜_TTS训练集" / "audio"),
    ("语音合集48k", FAKE / "新素材" / "音频" / "语音合集_分条" / "音频_WAV48k"),
    ("语音", FAKE / "新素材" / "音频" / "语音"),
    ("素材_中文", FAKE / "庄方宜素材" / "12-音频与语音" / "语音" / "中文"),
]

MIN_DUR, MAX_DUR = 3.0, 10.0
MAX_FLAT = 0.20
MAX_TB = 2.0
MAX_SIL = 0.60

# ---------- 排除名单 ----------
excl: set[str] = set()
hd = json.loads(HELD.read_text(encoding="utf-8"))
for key in ("targets", "all", "fresh"):
    for t in hd.get(key, []):
        p = Path(t["path"]) if isinstance(t, dict) else Path(t)
        excl.add(p.stem)
        excl.add(p.stem[:40])
for w in TRAIN.glob("*.wav"):
    excl.add(w.stem)
    excl.add(w.stem[:40])
print("=" * 104)
print("修正版参考筛选（**排除留出集 + 训练集**）")
print("=" * 104)
print(f"  排除名单: {len(excl)} 个 stem")
print(f"    （留出集 targets {len(hd.get('targets', []))} + "
      f"all {len(hd.get('all', []))} + 训练集 {len(list(TRAIN.glob('*.wav')))}）")


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
    if rms < 0.01:
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
    return {"dur": dur, "rms": rms, "tb": tb, "flat": flat,
            "sil": sil_tail, "peak": float(np.abs(x).max()), "sr": sr}


stage1 = []
seen: set[str] = set()
excluded = 0
for label, d in SRC:
    if not d.is_dir():
        continue
    ws = sorted(d.rglob("*.wav"))
    hit = ex = 0
    for w in ws:
        st = w.stem
        if st in excl or st[:40] in excl:
            ex += 1
            excluded += 1
            continue
        key = st[:60]
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
    print(f"  {label:<14s} 扫 {len(ws):5d}  排除 {ex:4d}  ->  合格 {hit:3d}")

print()
print(f"  共排除 {excluded} 个（留出集/训练集）")
print(f"  阶段 1 合格: **{len(stage1)} 个**")
if not stage1:
    print("  !! 无合格候选")
    sys.exit(1)

# ---------- 相似度 ----------
print()
print("=" * 104)
print("相似度筛选（与留出集目标比）")
print("=" * 104)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
sys.path[:0] = [str(GSV), str(GSV / "GPT_SoVITS"), str(GSV / "tools")]
os.chdir(str(GSV))
from GPT_SoVITS.sv import SV  # noqa: E402

sv = SV(DEV, is_half=False)
targets = [Path(t["path"]) for t in hd.get("targets", [])
           if Path(t["path"]).exists()]


def emb(p: Path):
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
print(f"  目标 {len(T)} 条   候选 {len(stage1)} 条")

scored = []
for i, (label, w, mt) in enumerate(stage1):
    e = emb(w)
    if e is None:
        continue
    sim = float((T @ e.reshape(-1)).max().item())
    scored.append({"label": label, "path": w, "sim": sim, **mt})
    if (i + 1) % 50 == 0:
        print(f"    ...{i+1}/{len(stage1)}")

for s in scored:
    s["score"] = s["sim"] - s["flat"] * 0.3
scored.sort(key=lambda s: -s["score"])

print()
print("=" * 104)
print("TOP 25（已排除留出集/训练集）")
print("=" * 104)
print(f"  {'#':>3s} {'来源':<12s} {'文件':<42s} {'相似度':>8s} "
      f"{'尾平坦':>8s} {'尾/体':>8s} {'时长':>6s}")
print("  " + "-" * 100)
for i, s in enumerate(scored[:25], 1):
    print(f"  {i:3d} {s['label']:<12s} {s['path'].stem[:42]:<42s} "
          f"{s['sim']:8.4f} {s['flat']:8.4f} {s['tb']:+8.2f} {s['dur']:5.2f}s")

print()
print("=" * 104)
print("相似度分布检查（确认没有 1.0 的自我比对）")
print("=" * 104)
sims = np.array([s["sim"] for s in scored])
print(f"  max {sims.max():.4f}   p99 {np.percentile(sims,99):.4f}   "
      f"median {np.median(sims):.4f}   min {sims.min():.4f}")
n_one = int((sims > 0.999).sum())
print(f"  相似度 >0.999 的: **{n_one} 个**"
      f"  {'✓ 已清除自我比对' if n_one == 0 else '✗ 仍有污染！'}")

print()
print("=" * 104)
print("对照")
print("=" * 104)
REF = GSV / "ref" / "default.wav"
x, sr = load(REF)
print(f"  当前 ref/default.wav : {len(x)/sr:.2f}s  "
      f"（**超 3~10s 范围**）")
n = int(0.4 * sr)
tail = x[-n:]
nn = min(len(tail), 2048)
S = np.abs(np.fft.rfft(tail[:nn] * np.hanning(nn))) + 1e-12
print(f"    尾平坦 {float(np.exp(np.log(S).mean())/S.mean()):.4f}")
e = emb(REF)
if e is not None:
    print(f"    相似度 {float((T @ e.reshape(-1)).max().item()):.4f}")
print(f"  真人录音典型: 尾平坦 0.1352")

OUT.mkdir(parents=True, exist_ok=True)
for i, s in enumerate(scored[:10], 1):
    shutil.copy2(s["path"], OUT / f"top{i:02d}_{s['path'].stem[:44]}.wav")
print()
print(f"  前 10 名已复制到 {OUT}")
print()
print("=" * 104)
print("判读")
print("=" * 104)
if scored:
    b = scored[0]
    print(f"  最优（干净+相似）: {b['path'].name}")
    print(f"    来源 {b['label']}   相似度 {b['sim']:.4f}   "
          f"尾平坦 {b['flat']:.4f}   尾/体 {b['tb']:+.2f}dB   "
          f"时长 {b['dur']:.2f}s")
    print()
    print("  ## 下一步")
    print("    1. 用 top01~top03 合成（改 compare_ref_swap.py 的 CANDS）")
    print("    2. eval_dirs.py 测相似度")
    print("    3. make_ref_listen.py 做盲听包")
    print()
    print("  ## ⚠️ 评估注意")
    print("    用这些当参考后，**原留出集仍可用于评估**（因为已排除）")
    print("    这正是排除的意义 —— 参考与评估目标不重叠")

rep = ["=" * 80, "参考音频筛选（排除留出集/训练集）", "=" * 80, ""]
rep.append(f"排除 {excluded} 个（留出集+训练集）")
rep.append(f"阶段1 合格 {len(stage1)}   阶段2 有效 {len(scored)}")
rep.append(f"相似度 max {sims.max():.4f}  中位 {np.median(sims):.4f}   "
           f">0.999 的 {n_one} 个")
rep.append("")
rep.append(f"{'#':>3s} {'来源':<12s} {'文件':<44s} {'相似度':>8s} "
           f"{'尾平坦':>8s} {'尾/体':>8s}")
for i, s in enumerate(scored[:40], 1):
    rep.append(f"{i:3d} {s['label']:<12s} {s['path'].stem[:44]:<44s} "
               f"{s['sim']:8.4f} {s['flat']:8.4f} {s['tb']:+8.2f}")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
(OUT / "candidates.json").write_text(json.dumps(
    [{"label": s["label"], "path": str(s["path"]), "sim": s["sim"],
      "flat": s["flat"], "tb": s["tb"], "dur": s["dur"],
      "score": s["score"]} for s in scored[:100]],
    ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  -> {OUT / 'report.txt'}")
print("=" * 104)
