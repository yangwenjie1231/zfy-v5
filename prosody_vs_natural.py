# -*- coding: utf-8 -*-
"""韵律自然度对比：v5 / v2Pro 的 F0 轮廓与**真人录音**的距离。

## 为什么这个测试才是对的

前面用的韵律指标（F0 范围、能量动态、停顿）都是**统计量**，
它们测的是"起伏多大"，不是"起伏得对不对"。
用户听出的"语气好"是**轮廓形状**的自然度，统计量测不到。

正确做法：**拿真人录音当参照**，比对合成语音的 F0 轮廓。
  · 数据集中有官方逐字文本的句子，可以找到对应的真人录音
  · 用 DTW（动态时间规整）对齐后算 F0 距离 —— 容忍时长差异
  · 距离越小 = 韵律越接近真人

同时对比时长（节奏）：合成时长 / 真人时长。

## 关键前提

必须用**数据集中存在的句子**，否则没有真人参照。
所以本脚本从 manifest 里筛句子，而不是用固定的测试集。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
OLD = Path(r"D:\A\voiceclone\GPT-SoVITS")
OUT = V5 / "out" / "prosody_natural"
DATASET = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频\庄方宜_TTS训练集")

GOOD_REF = Path(r"D:\A\voiceclone\zfh-models-v2pro\ref\default.wav")
GOOD_REF_TEXT = "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？"
ZFH_GPT = r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh-e4.ckpt"
V2PRO_SOV = r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh_v2pro_e12_s384.pth"
SEED = 42


def f0_curve(p: Path, fmin=70.0, fmax=400.0, win_ms=40.0, hop_ms=10.0):
    """返回 (f0 半音序列, 有声掩码)。只在有声帧上有值。"""
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    win, hop = int(win_ms / 1000 * sr), int(hop_ms / 1000 * sr)
    if len(x) < win * 3:
        return None, None
    nf = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(nf)[:, None]
    fr = x[idx] * np.hanning(win)
    e = np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-12
    thr = max(np.percentile(e, 55), np.max(e) * 0.02)
    lo, hi = max(2, int(sr / fmax)), int(sr / fmin)
    f0 = np.full(nf, np.nan)
    for i in range(nf):
        if e[i] < thr:
            continue
        s = fr[i] - fr[i].mean()
        ac = np.correlate(s, s, mode="full")[len(s) - 1:]
        if ac[0] <= 0:
            continue
        seg = ac[lo:hi]
        if len(seg) < 2:
            continue
        k = int(np.argmax(seg)) + lo
        if seg.max() > 0.30 * ac[0]:
            f0[i] = sr / k
    voiced = ~np.isnan(f0)
    if voiced.sum() < 8:
        return None, None
    st = 12 * np.log2(f0[voiced] / 55.0 + 1e-9)
    # 去八度跳变
    med = np.median(st)
    st = np.where((st > med + 9) | (st < med - 9), med, st)
    # 去均值：只比轮廓形状，不比绝对音高
    return st - st.mean(), voiced


def dtw_dist(a: np.ndarray, b: np.ndarray, band: int = 60) -> float:
    """DTW 距离（带 Sakoe-Chiba 带约束），容忍时长差异。

    返回**每步平均代价**，所以长度不同的序列也可比。
    注意：`cur[j-1]` 在 jlo 之前是 INF，不能用 0 初始化整行——
    否则边界会被错误地"免费"通过。
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan")
    INF = 1e18
    prev = np.full(m + 1, INF)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, INF)
        jlo = max(1, i - band)
        jhi = min(m, i + band)
        if jlo > jhi:
            continue
        for j in range(jlo, jhi + 1):
            cost = abs(a[i - 1] - b[j - 1])
            best = min(prev[j], cur[j - 1], prev[j - 1])
            if best < INF:
                cur[j] = cost + best
        prev = cur
    if prev[m] >= INF:
        return float("nan")
    # 归一化到路径长度（n+m 是上界），使不同长度可比
    return float(prev[m] / (n + m))


def use_v5():
    need = [str(GSV), str(GSV / "GPT_SoVITS"),
            str(GSV / "GPT_SoVITS" / "BigVGAN"),
            str(GSV / "tools"), str(GSV / "tools" / "asr"),
            str(GSV / "tools" / "uvr5")]
    for p in reversed(need):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    os.chdir(str(GSV))
    for m in list(sys.modules):
        if m.split(".")[0] in ("GPT_SoVITS", "module", "text", "tools",
                               "sv", "f5_tts", "TTS_infer_pack"):
            del sys.modules[m]


def use_old():
    need = [str(OLD), str(OLD / "GPT_SoVITS"),
            str(OLD / "GPT_SoVITS" / "BigVGAN"),
            str(OLD / "tools"), str(OLD / "tools" / "asr"),
            str(OLD / "tools" / "uvr5")]
    for p in reversed(need):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    os.chdir(str(OLD))
    for m in list(sys.modules):
        if m.split(".")[0] in ("GPT_SoVITS", "module", "text", "tools",
                               "sv", "f5_tts", "TTS_infer_pack"):
            del sys.modules[m]


def synth(version, t2s, vits, texts, tag, is_half=True, steps=None, cfg=None):
    import torch
    if version.startswith("v5"):
        use_v5()
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        base = GSV
    else:
        use_old()
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        base = OLD
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    cf = {"custom": {
        "device": "cuda", "is_half": is_half, "version": version,
        "t2s_weights_path": t2s, "vits_weights_path": vits,
        "bert_base_path": str(base / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"),
        "cnhuhbert_base_path": str(base / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"),
    }}
    tts = TTS(TTS_Config(cf))
    for i, t in enumerate(texts):
        p = d / f"{i:02d}.wav"
        if p.exists():
            continue
        inp = {"text": t, "text_lang": "zh", "ref_audio_path": str(GOOD_REF),
               "prompt_text": GOOD_REF_TEXT, "prompt_lang": "zh",
               "top_k": 15, "temperature": 1.0, "repetition_penalty": 1.35,
               "text_split_method": "cut1", "batch_size": 1,
               "speed_factor": 1.0, "seed": SEED, "parallel_infer": True}
        if steps:
            inp["sample_steps"] = steps
            inp["cfg_rate"] = cfg
        ch, sr = [], None
        for it in tts.run(inp):
            if it is None:
                continue
            sr, a = it
            ch.append(np.asarray(a))
        sf.write(str(p), np.concatenate(ch), sr)
    del tts
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--v5", default="r32_e02")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # 从数据集筛句子（要有真人录音）
    cands = []
    for r in csv.DictReader(open(DATASET / "manifest.csv", encoding="utf-8-sig")):
        if r.get("处理") != "保留":
            continue
        t = (r.get("文本清洗") or "").strip()
        dur = float(r.get("新时长") or 0)
        if not t or "|" in t or not (3.0 <= dur <= 10.0):
            continue
        wav = DATASET / "audio" / Path(r["原文件"]).name
        if wav.exists():
            cands.append((t, wav, dur))
    # 去重（同一句可能多条）
    seen, uniq = set(), []
    for t, w, d in cands:
        if t in seen:
            continue
        seen.add(t)
        uniq.append((t, w, d))
    texts = [t for t, _, _ in uniq[: args.n]]
    nat = [w for _, w, _ in uniq[: args.n]]
    natdur = [d for _, _, d in uniq[: args.n]]
    print(f"选用 {len(texts)} 句（均有真人录音）")

    print("合成 v5 …")
    w5 = next((GSV / "SoVITS_weights_v5").glob("*.pth"), None)
    w5 = (GSV / "SoVITS_weights_v5" / f"zfy_v5_e8_s3256_l32.pth")
    synth("v5dev", ZFH_GPT, str(w5), texts, "v5", True, 32, 1.30)
    print("合成 v2Pro …")
    synth("v2Pro", ZFH_GPT, V2PRO_SOV, texts, "v2pro", False)

    # 对比 F0 轮廓与真人
    print("\n计算 F0 轮廓 DTW 距离（越小越接近真人）…")
    rows = []
    for i, (t, natw, nd) in enumerate(zip(texts, nat, natdur)):
        nf, _ = f0_curve(natw)
        if nf is None:
            continue
        rec = {"i": i, "text": t[:30], "nat_dur": round(nd, 2)}
        for tag in ("v5", "v2pro"):
            p = OUT / tag / f"{i:02d}.wav"
            if not p.exists():
                continue
            gf, _ = f0_curve(p)
            if gf is None:
                continue
            x, sr = sf.read(str(p), dtype="float32", always_2d=True)
            rec[f"{tag}_dtw"] = round(dtw_dist(gf, nf), 4)
            rec[f"{tag}_dur"] = round(len(x.mean(axis=1)) / sr, 2)
        if "v5_dtw" in rec and "v2pro_dtw" in rec:
            rows.append(rec)

    if not rows:
        print("没有可比对的数据")
        return 1

    print(f"\n{'#':>3s} {'v5 DTW':>9s} {'v2Pro DTW':>10s} {'胜者':>6s} "
          f"{'v5时长':>7s} {'v2Pro时长':>9s} {'真人':>6s}  句子")
    print("-" * 92)
    w5n = w2n = 0
    for r in rows:
        win = "v5" if r["v5_dtw"] < r["v2pro_dtw"] else "v2Pro"
        w5n += win == "v5"
        w2n += win == "v2Pro"
        print(f"{r['i']:3d} {r['v5_dtw']:9.4f} {r['v2pro_dtw']:10.4f} "
              f"{win:>6s} {r.get('v5_dur', 0):7.2f} "
              f"{r.get('v2pro_dur', 0):9.2f} {r['nat_dur']:6.2f}  {r['text']}")

    a = np.array([r["v5_dtw"] for r in rows])
    b = np.array([r["v2pro_dtw"] for r in rows])
    dv = a - b
    n = len(dv)
    se = dv.std(ddof=1) / np.sqrt(n)
    t = float(dv.mean() / se) if se > 0 else 0.0
    crit = {19: 2.093, 23: 2.069, 24: 2.064}.get(n, 2.06)
    print("-" * 92)
    print(f"平均 DTW: v5 {a.mean():.4f}  vs  v2Pro {b.mean():.4f}")
    print(f"配对差 (v5-v2Pro): {dv.mean():+.4f}  t={t:.2f}  临界={crit}  "
          f"{'显著' if abs(t) > crit else '不显著'}")
    print(f"逐句胜场: v5 {w5n} / v2Pro {w2n}   (共 {n} 句)")
    print("（DTW 越小 = F0 轮廓越接近真人 = 韵律越自然）")

    (OUT / "natural_prosody.json").write_text(json.dumps(
        {"rows": rows, "n": n, "v5_mean": round(float(a.mean()), 4),
         "v2pro_mean": round(float(b.mean()), 4),
         "paired_diff": round(float(dv.mean()), 4), "t": round(t, 2),
         "crit": crit, "sig": bool(abs(t) > crit),
         "win_v5": w5n, "win_v2pro": w2n},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果: {OUT/'natural_prosody.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
