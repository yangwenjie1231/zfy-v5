# -*- coding: utf-8 -*-
"""诊断：为什么换上 zfh-e4 GPT 后 CER 飙到 43-47%、时长变成自然语速的 1.9 倍。

必须查清的三件事
----------------
1. CER 43-47% 是**真实乱读**，还是**测量假象**（例如模型多输出了一段，
   ASR 把重复内容也转出来，拉高了 CER）？
   → 逐句打印 参考 / 识别 / 长度比。

2. 时长 1.9 倍是**真拖长**，还是**首尾/中间多了静音**？
   → 测静音占比与"有效语音时长"。

3. 我上一轮 `compare_v5_v2pro.py` 给 v2Pro 用的是 `s1v3.ckpt`（底模），
   而**生产用的是 zfh-e4.ckpt** —— 这意味着我上一轮的"v2Pro 更好"
   结论测的不是用户实际在用的配置。本脚本同时跑两种，量化差异。

输出：逐句对照表 + 静音分析。
"""
from __future__ import annotations

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
OUT = V5 / "out"
DATASET = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频\庄方宜_TTS训练集")

TEXTS = [
    "迎敌或是迎客，都得整装后再出发。",
    "出门前，要不要先喝杯茶提提神？",
    "让我想想。",
    "武陵城外有一片梨树林，每年春天都会开满白色的花。",
    "她就在那阵花香里，只是他自己不知道。",
    "学得越多，就越是清楚，自己学到的不过沧海一粟。",
    "上次外出执行任务……都是好几年前了，但愿别生疏。",
    "新的是更厉害些，可那旧的用惯了，确实……也不舍得。",
    "再好的武器，也得朝夕相处磨合上几天。",
    "你找我帮忙，想必是遇上棘手的问题了，直说就好。",
    "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。",
    "嗯……我理一理。",
]


def norm(s: str) -> str:
    s = re.sub(r"<\|[^|]*\|>", "", s or "")
    return re.sub(r"[^\u4e00-\u9fff0-9a-z]", "", s.lower())


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(h) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (0 if r[i - 1] == h[j - 1] else 1))
            prev = cur
    return dp[len(h)] / len(r)


def silence_ratio(x: np.ndarray, sr: int, thr_db=-45.0):
    """静音帧占比 + 有效语音时长。"""
    win, hop = int(0.025 * sr), int(0.010 * sr)
    nf = 1 + (len(x) - win) // hop
    if nf < 2:
        return 1.0, 0.0
    idx = np.arange(win)[None, :] + hop * np.arange(nf)[:, None]
    e = np.sqrt(np.mean(x[idx] ** 2, axis=1)) + 1e-12
    db = 20 * np.log10(e)
    voiced = db > thr_db
    return (1.0 - float(voiced.mean()), float(voiced.sum()) * 0.010)


def main() -> int:
    sys.path.insert(0, str(GSV))
    sys.path.insert(0, str(GSV / "GPT_SoVITS"))
    os.chdir(str(GSV))
    from funasr import AutoModel
    import librosa

    asr = AutoModel(model="iic/SenseVoiceSmall", device="cuda:0",
                    disable_update=True, cache_dir=r"D:\A\voiceclone\models")

    groups = [
        ("A_v5_basegpt", OUT / "gpt_fix" / "A_v5_basegpt"),
        ("B_v5_zfhgpt", OUT / "gpt_fix" / "B_v5_zfhgpt"),
        ("C_v2pro_zfh(生产)", OUT / "gpt_fix" / "v2pro_prod"),
        ("D_v2pro_s1v3", OUT / "fair_compare" / "v2pro"),
    ]

    # 自然时长参照
    import csv
    nat = {}
    for r in csv.DictReader(open(DATASET / "manifest.csv", encoding="utf-8-sig")):
        if r.get("处理") == "保留":
            nat[(r.get("文本清洗") or "").strip()] = float(r["新时长"])

    rows = []
    for tag, d in groups:
        if not d.exists():
            print(f"跳过 {tag}（不存在）")
            continue
        print(f"\n{'='*100}\n{tag}\n{'='*100}")
        ps = sorted(d.glob("*.wav"))
        for i, p in enumerate(ps):
            if i >= len(TEXTS):
                break
            ref = TEXTS[i]
            x, sr = sf.read(str(p), dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            w16 = librosa.load(str(p), sr=16000, mono=True)[0]
            r = asr.generate(input=w16, cache={}, language="zh", use_itn=False,
                             batch_size_s=60, merge_vad=True, merge_length_s=15)
            hyp = r[0]["text"] if r else ""
            e = cer(ref, hyp)
            sil, voiced = silence_ratio(x, sr)
            dur = len(x) / sr
            nref = len(norm(ref))
            nhyp = len(norm(hyp))
            nats = nat.get(ref)
            rows.append({"tag": tag, "i": i, "ref": ref, "hyp": hyp,
                         "cer": round(e, 4), "dur": round(dur, 3),
                         "nat_dur": nats, "sil": round(sil, 3),
                         "voiced": round(voiced, 3),
                         "len_ratio": round(nhyp / max(nref, 1), 2)})
            flag = "  <== 异常" if e > 0.35 or nhyp > nref * 1.4 else ""
            nat_s = f"{nats:5.2f}" if nats else "  -- "
            print(f"  [{i:2d}] CER={e*100:5.1f}%  时长={dur:5.2f}s (自然{nat_s}s) "
                  f"静音={sil*100:4.0f}%  字数 {nref}->{nhyp} ({nhyp/max(nref,1):.2f}x){flag}")
            if e > 0.35:
                print(f"        参考: {ref}")
                print(f"        识别: {hyp[:90]}")

    (OUT / "diag_transcribe.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总
    print("\n" + "=" * 100)
    print("汇总")
    print("=" * 100)
    print(f"{'组':22s} {'CER':>8s} {'时长':>8s} {'自然':>8s} {'静音%':>7s} "
          f"{'有效语音':>9s} {'字数比':>7s}")
    for tag, _ in groups:
        rs = [r for r in rows if r["tag"] == tag]
        if not rs:
            continue
        c = np.mean([r["cer"] for r in rs])
        d = np.mean([r["dur"] for r in rs])
        n = [r["nat_dur"] for r in rs if r["nat_dur"]]
        s = np.mean([r["sil"] for r in rs])
        v = np.mean([r["voiced"] for r in rs])
        lr = np.mean([r["len_ratio"] for r in rs])
        print(f"{tag:22s} {c*100:7.1f}% {d:7.2f}s "
              f"{(np.mean(n) if n else 0):7.2f}s {s*100:6.0f}% {v:8.2f}s {lr:7.2f}")
    print(f"\n结果: {OUT/'diag_transcribe.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
