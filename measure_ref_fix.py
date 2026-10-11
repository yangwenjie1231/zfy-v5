# -*- coding: utf-8 -*-
"""量化「修正参考音频」的影响。

四组（同 12 句、同 seed、同 GPT=zfh-e4）：
  P1 = v5    + 生产参考(对齐)      P2 = v5    + 数据集参考(错配)
  P3 = v2Pro + 生产参考(对齐)      P4 = v2Pro + 数据集参考(错配)

关注指标：
  · CER：串词/泄漏会直接抬高 CER
  · 时长：错配参考会导致时长异常（模型补完参考文本）
  · SIM：音色相似度
  · 字数比：>1.2 说明多念了内容（泄漏）
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
OUT = V5 / "out" / "ref_fix"
GOOD_REF = Path(r"D:\A\voiceclone\zfh-models-v2pro\ref\default.wav")

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


def main() -> int:
    sys.path.insert(0, str(GSV))
    sys.path.insert(0, str(GSV / "GPT_SoVITS"))
    os.chdir(str(GSV))
    import torch
    from funasr import AutoModel
    import librosa
    from sv import SV

    asr = AutoModel(model="iic/SenseVoiceSmall", device="cuda:0",
                    disable_update=True, cache_dir=r"D:\A\voiceclone\models")
    sv = SV("cuda", False)

    def emb(w):
        t = torch.from_numpy(np.ascontiguousarray(w)).unsqueeze(0).to("cuda")
        with torch.no_grad():
            e = sv.compute_embedding3(t)
        return e.detach().cpu().flatten().numpy().astype(np.float64)

    ref_emb = emb(librosa.load(str(GOOD_REF), sr=32000, mono=True)[0])

    def sim(p):
        b = emb(librosa.load(str(p), sr=32000, mono=True)[0])
        a = ref_emb
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    groups = [("P1_v5_goodref", "v5 + 对齐参考"),
              ("P2_v5_badref", "v5 + 错配参考"),
              ("P3_v2pro_goodref", "v2Pro + 对齐参考"),
              ("P4_v2pro_badref", "v2Pro + 错配参考")]

    rows = []
    print(f"{'组':24s} {'CER':>7s} {'时长':>7s} {'字数比':>7s} {'SIM':>8s}")
    print("-" * 60)
    for tag, label in groups:
        d = OUT / tag
        if not d.exists():
            continue
        ps = sorted(d.glob("*.wav"))
        cs, ds, lr, ss = [], [], [], []
        for i, p in enumerate(ps):
            if i >= len(TEXTS):
                break
            x, sr = sf.read(str(p), dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            ds.append(len(x) / sr)
            w16 = librosa.load(str(p), sr=16000, mono=True)[0]
            r = asr.generate(input=w16, cache={}, language="zh", use_itn=False,
                             batch_size_s=60, merge_vad=True, merge_length_s=15)
            hyp = r[0]["text"] if r else ""
            cs.append(cer(TEXTS[i], hyp))
            lr.append(len(norm(hyp)) / max(len(norm(TEXTS[i])), 1))
            ss.append(sim(p))
        row = {"tag": tag, "label": label, "n": len(cs),
               "cer": round(float(np.mean(cs)), 4),
               "dur": round(float(np.mean(ds)), 3),
               "len_ratio": round(float(np.mean(lr)), 3),
               "sim": round(float(np.mean(ss)), 4),
               "_cer": cs}
        rows.append(row)
        print(f"{label:24s} {row['cer']*100:6.1f}% {row['dur']:6.2f}s "
              f"{row['len_ratio']:7.2f} {row['sim']:8.4f}")

    # 配对检验：对齐 vs 错配（同模型内）
    def paired(a, b):
        dv = np.array(a) - np.array(b)
        n = len(dv)
        se = dv.std(ddof=1) / np.sqrt(n)
        t = float(dv.mean() / se) if se > 0 else 0.0
        crit = {11: 2.201, 12: 2.179}.get(n, 2.2)
        return {"mean": round(float(dv.mean()), 4), "t": round(t, 2),
                "crit": crit, "sig": bool(abs(t) > crit)}

    print("\n" + "=" * 70)
    print("配对检验：修正参考音频的收益（CER 下降为正）")
    print("=" * 70)
    tests = {}
    for good, bad, name in [("P1_v5_goodref", "P2_v5_badref", "v5"),
                            ("P3_v2pro_goodref", "P4_v2pro_badref", "v2Pro")]:
        g = next((r for r in rows if r["tag"] == good), None)
        b = next((r for r in rows if r["tag"] == bad), None)
        if not g or not b:
            continue
        p = paired(b["_cer"], g["_cer"])   # 错配 - 对齐 > 0 表示修正后 CER 下降
        tests[name] = p
        print(f"  {name:6s} CER 差(错配-对齐) = {p['mean']:+.4f}  "
              f"t={p['t']}  临界={p['crit']}  "
              f"{'显著' if p['sig'] else '不显著'}")

    (OUT / "ref_fix_metrics.json").write_text(json.dumps(
        {"rows": [{k: v for k, v in r.items() if not k.startswith("_")}
                  for r in rows], "paired": tests},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果: {OUT/'ref_fix_metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
