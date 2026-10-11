# -*- coding: utf-8 -*-
"""从训练集 `5-wav32k` 里挑参考音频 —— **文本零风险**。

## 为什么这条路最稳
前面几版都在跟"文本对应"搏斗：
    · `剧情语音` / `语音合集_分条` 的文件名与训练集不同 -> 需推断映射
    · 推断出的映射有错位风险（技能警告：`ref_text` 写错 = 音色提示作废）
    · 实测 12 个候选里只有 5 个通过包络互相关验证

**而 `5-wav32k` 的文本直接躺在 `2-name2text.txt` 里**：
    key   = 文件名（如 `剧情_e9m4_010_……潘工，真不用帮忙？摆摆桌椅的时间还是有的`）
    value = 第 4 列 = 规范化文本
-> **同一条记录，不可能错位**

## 为什么用训练集音频当参考是合理的
    1. 参考音频**不需要"未见过的"** —— 它只是音色示范
    2. 评估用的是**留出集**（训练集外），两者互不影响
    3. 训练集片段**已被裁剪**（去首尾静音），天然比原始素材干净
    4. 这正是所有人做 GPT-SoVITS 的标准做法

## 筛选
    时长 3~10s / 尾平坦 <0.20 / 尾/体 <+2dB / 静音尾 <0.6s
    再按「相似度 - 尾平坦×0.3」排序

## 输出
    dist/ref_train/
        report.txt
        top01~top08.wav
        CANDS.py    （可直接粘进 compare_ref_swap.py）
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
OUT = V5 / "dist" / "ref_train"
TRAIN = GSV / "logs" / "zfy_v5" / "5-wav32k"
TXT = GSV / "logs" / "zfy_v5" / "2-name2text.txt"
HELD = V5 / "dist" / "heldout" / "heldout_clean.json"

MIN_DUR, MAX_DUR = 3.0, 10.0
MAX_FLAT = 0.20
MAX_TB = 2.0
MAX_SIL = 0.60

# 文本表（key = stem）
tm: dict[str, str] = {}
for line in TXT.read_text(encoding="utf-8").splitlines():
    p = line.split("\t")
    if len(p) >= 4:
        tm[p[0].replace(".wav", "")] = p[3].strip()
print("=" * 104)
print("从训练集挑参考音频（文本零风险）")
print("=" * 104)
print(f"  训练集: {len(list(TRAIN.glob('*.wav')))} 个 wav")
print(f"  文本表: {len(tm)} 条（key = 文件名，**同记录，不会错位**）")


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
    sil = (len(x) - end) / sr
    if sil > MAX_SIL:
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
    return {"dur": dur, "rms": rms, "tb": tb, "flat": flat, "sil": sil,
            "peak": float(np.abs(x).max()), "sr": sr}


stage1 = []
for w in sorted(TRAIN.glob("*.wav")):
    if w.stem not in tm:
        continue                      # 无文本 -> 不能用
    mt = acoustic(w)
    if mt is None:
        continue
    if mt["flat"] < MAX_FLAT and mt["tb"] < MAX_TB and mt["peak"] < 0.99:
        stage1.append((w, mt))

print(f"  声学合格: **{len(stage1)} 个**")
if not stage1:
    print("  !! 无合格候选")
    sys.exit(1)

# ---------- 相似度 ----------
print()
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
sys.path[:0] = [str(GSV), str(GSV / "GPT_SoVITS"), str(GSV / "tools")]
os.chdir(str(GSV))
from GPT_SoVITS.sv import SV  # noqa: E402

sv = SV(DEV, is_half=False)
hd = json.loads(HELD.read_text(encoding="utf-8"))
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
for i, (w, mt) in enumerate(stage1):
    e = emb(w)
    if e is None:
        continue
    scored.append({"path": w, "stem": w.stem,
                   "text": tm[w.stem],
                   "sim": float((T @ e.reshape(-1)).max().item()), **mt})
    if (i + 1) % 50 == 0:
        print(f"    ...{i+1}/{len(stage1)}")

for s in scored:
    s["score"] = s["sim"] - s["flat"] * 0.3
scored.sort(key=lambda s: -s["score"])

print()
print("=" * 104)
print(f"TOP 25（共 {len(scored)} 个合格）")
print("=" * 104)
print(f"  {'#':>3s} {'文件':<40s} {'相似度':>8s} {'尾平坦':>8s} "
      f"{'尾/体':>8s} {'时长':>6s} {'静音尾':>7s}")
print("  " + "-" * 100)
for i, s in enumerate(scored[:25], 1):
    print(f"  {i:3d} {s['stem'][:40]:<40s} {s['sim']:8.4f} {s['flat']:8.4f} "
          f"{s['tb']:+8.2f} {s['dur']:5.2f}s {s['sil']:6.2f}s")

print()
print("=" * 104)
print("对照")
print("=" * 104)
REF = GSV / "ref" / "default.wav"
x, sr = load(REF)
n = int(0.4 * sr)
tail = x[-n:]
nn = min(len(tail), 2048)
S = np.abs(np.fft.rfft(tail[:nn] * np.hanning(nn))) + 1e-12
fl_ref = float(np.exp(np.log(S).mean()) / S.mean())
e_ref = emb(REF)
print(f"  当前 ref/default.wav : {len(x)/sr:.2f}s  "
      f"（**超 3~10s**）  尾平坦(末0.4s) {fl_ref:.4f}")
if e_ref is not None:
    print(f"    相似度 {float((T @ e_ref.reshape(-1)).max().item()):.4f}")
print(f"  真人录音典型: 尾平坦 0.1352  尾/体 -4.94dB")

# 导出
OUT.mkdir(parents=True, exist_ok=True)
for i, s in enumerate(scored[:8], 1):
    shutil.copy2(s["path"], OUT / f"top{i:02d}_{s['stem'][:40]}.wav")

# 生成 CANDS：orig + 尾最干净的 2 个 + 相似度最高的 1 个
by_flat = sorted(scored, key=lambda s: s["flat"])
sel, seen = [], set()
for s in by_flat[:2]:
    sel.append(s)
    seen.add(s["stem"])
for s in sorted(scored, key=lambda s: -s["sim"]):
    if s["stem"] not in seen and len(sel) < 3:
        sel.append(s)
        seen.add(s["stem"])
for s in sorted(scored, key=lambda s: -s["sim"]):
    if s["stem"] not in seen and len(sel) < 4:
        sel.append(s)
        seen.add(s["stem"])

lines = ["CANDS = [",
         '    ("orig", GSV / "ref" / "default.wav",',
         '     (GSV / "ref" / "ref.txt").read_text(encoding="utf-8").strip()),']
for s in sel:
    lines.append(f'    ("t{s["sim"]*1000:.0f}", Path(r"{s["path"]}"),')
    lines.append(f'     {s["text"]!r}),   # flat={s["flat"]:.4f} '
                 f'sim={s["sim"]:.4f} tb={s["tb"]:+.2f}')
lines.append("]")
(OUT / "CANDS.py").write_text("\n".join(lines) + "\n", encoding="utf-8")

print()
print("=" * 104)
print("推荐 CANDS（文本零风险）")
print("=" * 104)
print("\n".join(lines))
print()
print("  三个选取维度：")
print("    · 尾音最干净 2 个（解决用户的'尾音气声'）")
print("    · 相似度最高 2 个（保证音色）")

rep = ["=" * 80, "训练集参考候选（文本零风险）", "=" * 80, ""]
rep.append(f"合格 {len(scored)} 个   门槛 平坦<{MAX_FLAT} 尾/体<{MAX_TB} "
           f"静音尾<{MAX_SIL}")
rep.append("")
rep.append(f"{'#':>3s} {'文件':<44s} {'相似度':>8s} {'尾平坦':>8s} {'尾/体':>8s}")
for i, s in enumerate(scored[:40], 1):
    rep.append(f"{i:3d} {s['stem'][:44]:<44s} {s['sim']:8.4f} "
               f"{s['flat']:8.4f} {s['tb']:+8.2f}")
rep.append("")
rep.append("文本（前 10）:")
for i, s in enumerate(scored[:10], 1):
    rep.append(f"{i:2d}. {s['text'][:60]}")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
(OUT / "candidates.json").write_text(json.dumps(
    [{k: v for k, v in s.items() if k != "path"} | {"path": str(s["path"])}
     for s in scored[:50]], ensure_ascii=False, indent=2), encoding="utf-8")
print()
print(f"  -> {OUT / 'CANDS.py'}")
print(f"  -> {OUT / 'report.txt'}")
print(f"  -> 前 8 名 wav 已复制")
print("=" * 104)
