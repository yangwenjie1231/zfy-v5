# -*- coding: utf-8 -*-
"""决定性对比：换参考音频 -> 尾音是否变干净？

## 问题（实测）
    v2Pro 输出 : 尾/体 **+0.83 dB**, 尾平坦 **0.3338**
    真人录音   : 尾/体 **-4.94 dB**, 尾平坦 **0.1352**
    -> 模型句末**不衰减**且**噪声性高 2.5 倍** = 用户听到的"尾音气声强"

    参考音频 ref/default : 尾/体 **+29.83 dB**, 尾平坦 **0.5619**
    -> **比模型输出还脏 1.7 倍**（平坦度）

## 假设
模型在**模仿参考音频的尾部特征**。换成"尾部干净"的参考 -> 输出尾部也变干净。

## 设计（只换参考，其余全同）
    SoVITS : v2Pro（固定）
    GPT    : zfh-e4.ckpt（固定）
    参数   : top_k 15 / temp 1.0 / rep 1.35 / seed 42（固定）
    文本   : 10 句（含 3 句长句，最能看尾部）
    ────────────────────────────────────────────
    变量   : 参考音频
        orig        ref/default.wav（当前，评分 5.37）
        cand1       剧情_e9m4_010      （评分 0.589，最优）
        cand2       剧情_e9m4_011      （评分 0.748）
        cand3       剧情_e7m1_002      （评分 0.664）

## 判据
    · **尾部平坦度**（越低越干净）—— 核心
    · **尾/体dB**（越负越像真人衰减）
    · 时长/语速（换参考不该大幅改变）

## 输出
    dist/ref_swap_ab/<tag>/NN.wav
    dist/ref_swap_ab/report.txt
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
OUT = V5 / "dist" / "ref_swap_v2"
SOVITS = GSV / "ref" / "v2pro_sovits.pth"
GPT = GSV / "ref" / "zfh-e4.ckpt"
TXT = GSV / "logs" / "zfy_v5" / "2-name2text.txt"
VERSION = "v2Pro"
SEED = 42

# 10 句（3 长 + 中 + 短）
TEXTS = [
    "让我想想。",
    "好。",
    "迎敌或是迎客，都得整装后再出发。",
    "出门前，要不要先喝杯茶提提神？",
    "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？",
    "武陵城外有一片梨树林，每年春天都会开满白色的花。",
    "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。",
    "管理员，今天的实验数据我已经整理好了。",
    "若是要谈，就该拿出诚意来。",
    "或许吧。毕竟这座城，承载了太多了。",
]

# 参考候选： (标签, 音频路径, 文本来源)
# 从 dist/ref_train/CANDS.py 读取（由 pick_ref_from_train.py 生成）
_CANDS_FILE = V5 / "dist" / "ref_train" / "CANDS.py"
if _CANDS_FILE.exists():
    _ns: dict = {"GSV": GSV, "Path": Path}
    exec(_CANDS_FILE.read_text(encoding="utf-8"), _ns)
    CANDS = [(t, p, txt) for t, p, txt in _ns["CANDS"]]
else:
    CANDS = [("orig", GSV / "ref" / "default.wav",
              (GSV / "ref" / "ref.txt").read_text(encoding="utf-8").strip())]

# 从 2-name2text.txt 取文本（列 4 = 规范化文本）
text_map: dict[str, str] = {}
if TXT.exists():
    for line in TXT.read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if len(p) >= 4:
            text_map[p[0].replace(".wav", "")] = p[3].strip()

resolved: list[tuple[str, Path, str]] = []
for tag, p, t in CANDS:
    if not t:
        print(f"  !! {tag} 无参考文本（{p.name}），跳过")
        continue
    resolved.append((tag, p, t))

print("=" * 100)
print("换参考音频 A/B（只换参考，SoVITS/GPT/参数全同）")
print("=" * 100)
print(f"  SoVITS : {SOVITS.name}")
print(f"  GPT    : {GPT.name}")
print()
for tag, p, t in resolved:
    print(f"  {tag:6s} {p.name[:44]:<44s}")
    print(f"         文本: {t[:60]}")

sys.path[:0] = [str(GSV), str(GSV / "GPT_SoVITS"), str(V5)]
os.chdir(str(GSV))
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

PRE = GSV / "GPT_SoVITS" / "pretrained_models"


def metrics(p: Path) -> dict:
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if len(x) < sr * 0.2:
        return {}
    nf = 256
    m = (len(x) // nf) * nf
    rmsf = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
    ref = float(np.sqrt((x ** 2).mean())) + 1e-12
    voiced = np.where(rmsf > ref * (10 ** (-45 / 20)))[0]
    if len(voiced) == 0:
        return {}
    end = int((voiced[-1] + 1) * nf)
    n = int(0.4 * sr)
    tail = x[max(0, end - n):end]
    body = x[max(0, end // 2 - sr // 2):end // 2 + sr // 2]
    if len(tail) < n // 2:
        return {}
    tr = float(np.sqrt((tail ** 2).mean()))
    br = float(np.sqrt((body ** 2).mean())) + 1e-12
    nn = min(len(tail), 2048)
    S = np.abs(np.fft.rfft(tail[:nn] * np.hanning(nn))) + 1e-12
    flat = float(np.exp(np.log(S).mean()) / S.mean())
    return {"dur": len(x) / sr, "tb": 20 * np.log10(tr / br + 1e-12),
            "flat": flat, "sr": sr, "sil": (len(x) - end) / sr}


results: dict[str, list] = {}
for tag, ref_path, ref_txt in resolved:
    print()
    print("=" * 100)
    print(f"生成: {tag}   参考={ref_path.name}")
    print("=" * 100)
    cfg = {"custom": {
        "device": "cuda", "is_half": False, "version": VERSION,
        "t2s_weights_path": str(GPT),
        "vits_weights_path": str(SOVITS),
        "bert_base_path": str(PRE / "chinese-roberta-wwm-ext-large"),
        "cnhuhbert_base_path": str(PRE / "chinese-hubert-base"),
    }}
    try:
        tts = TTS(TTS_Config(cfg))
    except Exception as e:  # noqa: BLE001
        print(f"  !! TTS 构造失败: {type(e).__name__}: {str(e)[:100]}")
        continue
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, text in enumerate(TEXTS):
        try:
            chunks, sr = [], None
            for item in tts.run({
                "text": text, "text_lang": "zh",
                "ref_audio_path": str(ref_path), "prompt_text": ref_txt,
                "prompt_lang": "zh", "top_k": 15, "temperature": 1.0,
                "repetition_penalty": 1.35, "text_split_method": "cut1",
                "batch_size": 1, "speed_factor": 1.0, "seed": SEED,
                "parallel_infer": True,
            }):
                if item is None:
                    continue
                sr, a = item
                chunks.append(a)
            if not chunks:
                continue
            audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            fp = d / f"{i+1:02d}.wav"
            sf.write(str(fp), audio, sr)
            mt = metrics(fp)
            if mt:
                n_ch = len(re.sub(r"[，。？！、…,.?!\s]", "", text))
                mt["speed"] = n_ch / mt["dur"]
                rows.append(mt)
            if i < 2 or i == len(TEXTS) - 1:
                s = rows[-1] if rows else {}
                print(f"  {i+1:2d} {s.get('dur',0):5.2f}s  "
                      f"尾/体 {s.get('tb',0):+6.2f}dB  "
                      f"平坦 {s.get('flat',0):.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"  {i+1:2d} 失败 {type(e).__name__}: {str(e)[:70]}")
    if rows:
        results[tag] = rows
    del tts
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

print()
print("=" * 100)
print("汇总")
print("=" * 100)
if not results:
    print("  !! 无数据")
    sys.exit(1)
print(f"  {'参考':<8s} {'句数':>4s} {'平均时长':>9s} {'语速':>6s} "
      f"{'**尾平坦**':>10s} {'**尾/体dB**':>11s} {'静音尾':>8s}")
print("  " + "-" * 74)
summ = {}
for tag, rows in results.items():
    s = {
        "n": len(rows),
        "dur": float(np.mean([r["dur"] for r in rows])),
        "speed": float(np.mean([r["speed"] for r in rows])),
        "flat": float(np.mean([r["flat"] for r in rows])),
        "tb": float(np.mean([r["tb"] for r in rows])),
        "sil": float(np.mean([r["sil"] for r in rows])),
    }
    summ[tag] = s
    print(f"  {tag:<8s} {s['n']:4d} {s['dur']:8.2f}s {s['speed']:6.2f} "
          f"{s['flat']:10.4f} {s['tb']:+11.2f} {s['sil']:7.2f}s")

print()
print("  对照基准:")
print(f"    真人录音   尾平坦 0.1352   尾/体 -4.94 dB")
print(f"    v2Pro(orig) 见上表 orig 行")

if "orig" in summ:
    a = summ["orig"]
    print()
    print("=" * 100)
    print("相对 orig（当前参考）的变化")
    print("=" * 100)
    print(f"  {'参考':<8s} {'Δ尾平坦':>10s} {'Δ尾/体':>10s} {'Δ时长':>9s} "
          f"{'Δ语速':>8s}   判定")
    print("  " + "-" * 76)
    for tag, s in summ.items():
        if tag == "orig":
            continue
        df = s["flat"] - a["flat"]
        dt = s["tb"] - a["tb"]
        dd = s["dur"] - a["dur"]
        dsp = s["speed"] - a["speed"]
        verdict = ("✓ **尾部更干净**" if df < -0.02
                   else ("✗ 更脏" if df > 0.02 else "~ 无变化"))
        print(f"  {tag:<8s} {df:+10.4f} {dt:+10.2f} {dd:+8.2f}s "
              f"{dsp:+8.2f}   {verdict}")
    print()
    print("  判读：")
    best = min(summ.items(), key=lambda kv: kv[1]["flat"])
    print(f"    尾平坦最低: **{best[0]}** ({best[1]['flat']:.4f})")
    print(f"    orig 尾平坦: {a['flat']:.4f}")
    if best[0] != "orig":
        gain = (1 - best[1]["flat"] / a["flat"]) * 100
        print(f"    -> **换参考后尾音噪声降低 {gain:.0f}%**")
        print("       = 假设成立，参考音频的尾部脏度被模型模仿了")
    else:
        print("    -> orig 仍最优，**换参考没用**")
        print("       说明尾部噪声是模型自身特性，不是参考导致的")
    print()
    print("  ⚠️ 但必须同时看**音色相似度** —— 换参考会改变音色。")
    print("     用 eval_from_wav.py 测相似度再决定。")

rep = ["=" * 78, "换参考音频 A/B", "=" * 78, ""]
rep.append(f"SoVITS: {SOVITS.name}")
rep.append(f"GPT   : {GPT.name}")
rep.append("")
for tag, p, t in resolved:
    rep.append(f"{tag}: {p.name}")
    rep.append(f"  文本: {t}")
rep.append("")
rep.append(f"{'参考':<8s}{'句数':>5s}{'平均时长':>10s}{'语速':>8s}"
           f"{'尾平坦':>9s}{'尾/体dB':>10s}")
for tag, s in summ.items():
    rep.append(f"{tag:<8s}{s['n']:5d}{s['dur']:9.2f}s{s['speed']:8.2f}"
               f"{s['flat']:9.4f}{s['tb']:+10.2f}")
rep.append("")
rep.append("对照: 真人 尾平坦 0.1352 / 尾体 -4.94dB")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
(OUT / "report.json").write_text(json.dumps(
    {"summary": summ, "refs": {t: str(p) for t, p, _ in resolved}},
    ensure_ascii=False, indent=2), encoding="utf-8")
print()
print(f"  -> {OUT / 'report.txt'}")
print(f"  -> wav 在 {OUT}/<tag>/")
print("=" * 100)
