# -*- coding: utf-8 -*-
"""对比参考音频修复前后 —— 验证"尾音气声"是否消失。

## 假设
`ref/default.wav` 在 6.92s 后多说了 0.34s（文本里没有）
-> 模型学成"句末附加发声" -> 所有合成音频尾部多一截

**若假设成立**：用裁剪版 `default_trimmed.wav` 重新合成
-> 尾部那段应该**消失**

## 设计（只换参考音频，其余全同）
    SoVITS : v2Pro 生产权重（固定）
    GPT    : zfh-e4.ckpt（固定）
    参数   : top_k 15 / temp 1.0 / rep 1.35 / seed 42（固定）
    ─────────────────────────────────────
    变量   : 参考音频（原始 vs 裁剪）

## 判据
    尾部 0.4s 的
      · **低频占比**（0~2k）—— 多余段是浊音，低频高
      · **高频占比**（4~16k）—— 正常语音收尾以高频为主
      · **尾/体dB**、**静音尾时长**
    以及**总时长**是否变短（少了一截附加发声）

## 输出
    dist/ref_fix_ab/{orig,trimmed}/NN.wav
    dist/ref_fix_ab/report.txt
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
OUT = V5 / "dist" / "ref_fix_ab"
SOVITS = GSV / "ref" / "v2pro_sovits.pth"
GPT = GSV / "ref" / "zfh-e4.ckpt"
REF_ORIG = GSV / "ref" / "default.wav"
REF_TRIM = V5 / "dist" / "ref_fix" / "default_trimmed.wav"
REF_TXT_F = GSV / "ref" / "ref.txt"
VERSION = "v2Pro"
SEED = 42

TEXTS = [
    "让我想想。", "好。",
    "迎敌或是迎客，都得整装后再出发。",
    "出门前，要不要先喝杯茶提提神？", "嗯……我理一理。",
    "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？",
    "武陵城外有一片梨树林，每年春天都会开满白色的花。",
    "她就在那阵花香里，只是他自己不知道。",
    "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。",
    "这件事我三点五十分再去处理。",
    "管理员，今天的实验数据我已经整理好了。",
    "清波城那边的情况，我一直在关注。",
    "若是要谈，就该拿出诚意来。",
    "我答应你，会好好照顾自己。",
    "这些话，我本想留到以后再说的。",
    "你来了。坐吧，茶还热着。",
    "有些事，不是努力就能改变的。",
    "但不去试，就永远不知道结果。",
    "汤汤说，我是个念旧的人。",
    "或许吧。毕竟这座城，承载了太多。",
]
BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000),
         (4000, 8000), (8000, 16000)]


def load(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return x, int(sr)


def band_dist(s: np.ndarray, sr: int) -> dict:
    n = min(len(s), 4096)
    if n < 64:
        return {}
    S = np.abs(np.fft.rfft(s[:n] * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1 / sr)
    S = S / (S.sum() + 1e-12)
    return {f"{lo}-{hi}": float(S[(f >= lo) & (f < min(hi, sr / 2))].sum())
            for lo, hi in BANDS}


def analyze(p: Path) -> dict:
    x, sr = load(p)
    if len(x) < sr * 0.2:
        return {}
    dur = len(x) / sr
    # 裁尾部静音
    nf = 256
    m = (len(x) // nf) * nf
    rmsf = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
    ref = float(np.sqrt((x ** 2).mean())) + 1e-12
    voiced = np.where(rmsf > ref * (10 ** (-45 / 20)))[0]
    end = int((voiced[-1] + 1) * nf) if len(voiced) else len(x)
    tail = x[max(0, end - int(0.4 * sr)):end]
    bd = band_dist(tail, sr)
    lo = sum(v for k, v in bd.items() if int(k.split("-")[1]) <= 2000)
    hi = sum(v for k, v in bd.items() if int(k.split("-")[0]) >= 4000)
    mid = end // 2
    body = x[max(0, mid - sr // 2):mid + sr // 2]
    tr = float(np.sqrt((tail ** 2).mean())) if len(tail) else 0.0
    br = float(np.sqrt((body ** 2).mean())) + 1e-12
    return {
        "dur": dur, "voiced_end": end / sr,
        "sil_tail": (len(x) - end) / sr,
        "tail_lo": lo, "tail_hi": hi,
        "tail_rms": tr, "tail_body_db": 20 * np.log10(tr / br + 1e-12),
        "n_ch": 0,
    }


print("=" * 100)
print("参考音频修复对比（只换参考，SoVITS/GPT/参数全同）")
print("=" * 100)
print(f"  原始   : {REF_ORIG.name}  "
      f"({REF_ORIG.stat().st_size/1e6:.2f} MB)")
print(f"  裁剪   : {REF_TRIM.name}  "
      f"({REF_TRIM.stat().st_size/1e6:.2f} MB)")
print(f"  SoVITS : {SOVITS.name}")
print(f"  GPT    : {GPT.name}")

sys.path[:0] = [str(GSV), str(GSV / "GPT_SoVITS"), str(V5)]
os.chdir(str(GSV))
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

PRE = GSV / "GPT_SoVITS" / "pretrained_models"
ref_txt = REF_TXT_F.read_text(encoding="utf-8").strip()

results = {}
for tag, ref_path in (("orig", REF_ORIG), ("trimmed", REF_TRIM)):
    if not ref_path.exists():
        print(f"\n  !! 缺 {ref_path}")
        continue
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
            st = analyze(fp)
            if st:
                n_ch = len(re.sub(r"[，。？！、…,.?!\s]", "", text))
                st["n_ch"] = n_ch
                st["speed"] = n_ch / st["dur"]
                rows.append(st)
            if i < 2 or i == len(TEXTS) - 1:
                s = rows[-1] if rows else {}
                print(f"  {i+1:2d} {s.get('dur',0):5.2f}s  尾部低频 "
                      f"{s.get('tail_lo',0)*100:5.1f}%  高频 "
                      f"{s.get('tail_hi',0)*100:5.1f}%")
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

# ---------- 汇总 ----------
print()
print("=" * 100)
print("汇总对比")
print("=" * 100)
if len(results) < 2:
    print("  !! 数据不足")
    sys.exit(1)
print(f"  {'参考':<10s} {'句数':>4s} {'平均时长':>9s} {'语速':>6s} "
      f"{'尾低频':>8s} {'尾高频':>8s} {'尾/体dB':>8s} {'静音尾':>8s}")
print("  " + "-" * 76)
summ = {}
for tag, rows in results.items():
    dur = np.mean([r["dur"] for r in rows])
    sp = np.mean([r["speed"] for r in rows])
    lo = np.mean([r["tail_lo"] for r in rows])
    hi = np.mean([r["tail_hi"] for r in rows])
    tb = np.mean([r["tail_body_db"] for r in rows])
    st = np.mean([r["sil_tail"] for r in rows])
    summ[tag] = {"n": len(rows), "dur": dur, "speed": sp,
                 "lo": lo, "hi": hi, "tb": tb, "sil": st}
    print(f"  {tag:<10s} {len(rows):4d} {dur:8.2f}s {sp:6.2f} "
          f"{lo*100:7.1f}% {hi*100:7.1f}% {tb:+8.2f} {st:7.2f}s")

if "orig" in summ and "trimmed" in summ:
    a, b = summ["orig"], summ["trimmed"]
    print()
    print("=" * 100)
    print("变化（trimmed - orig）")
    print("=" * 100)
    print(f"  平均时长   {a['dur']:.2f}s -> {b['dur']:.2f}s  "
          f"({b['dur']-a['dur']:+.2f}s)")
    print(f"  语速       {a['speed']:.2f} -> {b['speed']:.2f}  "
          f"({b['speed']-a['speed']:+.2f})")
    print(f"  **尾部低频** {a['lo']*100:.1f}% -> {b['lo']*100:.1f}%  "
          f"({(b['lo']-a['lo'])*100:+.1f}pp)")
    print(f"  **尾部高频** {a['hi']*100:.1f}% -> {b['hi']*100:.1f}%  "
          f"({(b['hi']-a['hi'])*100:+.1f}pp)")
    print(f"  尾/体dB    {a['tb']:+.2f} -> {b['tb']:+.2f}")
    print()
    print("  判读：")
    if b["lo"] < a["lo"] - 0.05:
        print(f"    ✓ **裁剪后尾部低频下降 {(a['lo']-b['lo'])*100:.1f}pp**")
        print("      -> 那段『附加发声』确实消失了，**假设成立**")
    elif b["lo"] > a["lo"] + 0.05:
        print(f"    ✗ 尾部低频反而**上升** {(b['lo']-a['lo'])*100:.1f}pp")
    else:
        print(f"    ~ 尾部低频变化很小（{(b['lo']-a['lo'])*100:+.1f}pp）")
        print("      -> 可能不是参考音频导致的，需另找原因")
    if b["dur"] < a["dur"] - 0.1:
        print(f"    ✓ 平均时长缩短 {a['dur']-b['dur']:.2f}s"
              f"（少了一截附加发声）")

rep = ["=" * 78, "参考音频修复对比", "=" * 78, ""]
rep.append(f"原始  : {REF_ORIG.name}")
rep.append(f"裁剪  : {REF_TRIM.name}")
rep.append(f"SoVITS: {SOVITS.name}")
rep.append(f"GPT   : {GPT.name}")
rep.append("")
rep.append(f"{'参考':<10s}{'句数':>5s}{'平均时长':>10s}{'语速':>8s}"
           f"{'尾低频':>9s}{'尾高频':>9s}{'尾/体dB':>10s}")
for tag, s in summ.items():
    rep.append(f"{tag:<10s}{s['n']:5d}{s['dur']:9.2f}s{s['speed']:8.2f}"
               f"{s['lo']*100:8.1f}%{s['hi']*100:8.1f}%{s['tb']:+10.2f}")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
(OUT / "report.json").write_text(
    json.dumps({"summary": summ,
                "detail": {k: v for k, v in results.items()}},
               ensure_ascii=False, indent=2), encoding="utf-8")
print()
print(f"  -> {OUT / 'report.txt'}")
print(f"  -> wav 在 {OUT}/orig/ 与 {OUT}/trimmed/")
print("=" * 100)
