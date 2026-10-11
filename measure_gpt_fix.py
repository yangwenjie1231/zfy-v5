# -*- coding: utf-8 -*-
"""量化「换用微调 GPT」对 v5 的影响，并与生产 v2Pro 配对比较。

三组（同 12 句、同参考、同 seed）：
  A = v5 + s1v3 底模 GPT     （我上一轮的配置，用户反馈语气差）
  B = v5 + zfh-e4 微调 GPT   （本次修复）
  C = v2Pro 生产（zfh-e4 GPT）

指标：
  1. 说话人相似度（ERes2NetV2，同口径）
  2. ASR CER（可懂度）
  3. **韵律/语气代理指标**：
     · 时长（GPT 决定，换 GPT 应显著改变）
     · 语速曲线（前 1/3 vs 后 1/3 的帧能量重心）
     · F0（基频）动态范围 —— 语气起伏的直接体现
     · 能量包络动态范围 —— 重音强弱
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
OUT = V5 / "out" / "gpt_fix"
DATASET = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频\庄方宜_TTS训练集")
REF = DATASET / "ref_audio" / "ref.wav"

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


def f0_stats(x: np.ndarray, sr: int, fmin=70, fmax=400):
    """自相关法逐帧估 F0，返回（有效帧的）中位与半音动态范围。"""
    win, hop = int(0.040 * sr), int(0.010 * sr)
    if len(x) < win + hop:
        return None
    nf = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(nf)[:, None]
    fr = x[idx] * np.hanning(win)
    # 只取能量足够高的帧
    e = np.sqrt(np.mean(fr ** 2, axis=1))
    thr = np.percentile(e, 60)
    f0s = []
    lo, hi = int(sr / fmax), int(sr / fmin)
    for i in range(nf):
        if e[i] < thr:
            continue
        s = fr[i] - fr[i].mean()
        ac = np.correlate(s, s, mode="full")[len(s) - 1:]
        seg = ac[lo:hi]
        if len(seg) == 0 or ac[0] <= 0:
            continue
        k = int(np.argmax(seg)) + lo
        if seg.max() > 0.3 * ac[0]:
            f0s.append(sr / k)
    if len(f0s) < 10:
        return None
    f0s = np.array(f0s)
    med = float(np.median(f0s))
    p10, p90 = np.percentile(f0s, [10, 90])
    # 半音动态范围
    st = 12 * np.log2((p90 + 1e-9) / (p10 + 1e-9))
    return {"f0_median": round(med, 1), "f0_range_semitone": round(float(st), 2)}


def env_dynamics(x: np.ndarray, sr: int):
    """能量包络动态范围（dB）：衡量重音强弱起伏。"""
    win, hop = int(0.025 * sr), int(0.010 * sr)
    nf = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(nf)[:, None]
    e = np.sqrt(np.mean(x[idx] ** 2, axis=1)) + 1e-12
    db = 20 * np.log10(e)
    hi = np.percentile(db, 95)
    lo = np.percentile(db, 20)
    return round(float(hi - lo), 2)


def main() -> int:
    sys.path.insert(0, str(GSV))
    sys.path.insert(0, str(GSV / "GPT_SoVITS"))
    os.chdir(str(GSV))
    import torch
    from funasr import AutoModel
    import librosa
    from sv import SV

    DEV = "cuda"
    print("加载 ASR / 说话人编码器 …", flush=True)
    asr = AutoModel(model="iic/SenseVoiceSmall", device=DEV,
                    disable_update=True, cache_dir=r"D:\A\voiceclone\models")
    sv = SV(DEV, False)

    def emb(w):
        t = torch.from_numpy(np.ascontiguousarray(w)).unsqueeze(0).to(DEV)
        with torch.no_grad():
            e = sv.compute_embedding3(t)
        return e.detach().cpu().flatten().numpy().astype(np.float64)

    ref_emb = emb(librosa.load(str(REF), sr=32000, mono=True)[0])

    def sim(p):
        b = emb(librosa.load(str(p), sr=32000, mono=True)[0])
        a = ref_emb
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def cer(ref, hyp):
        import re
        n = lambda s: re.sub(r"[^\u4e00-\u9fff0-9a-z]", "",  # noqa: E731
                             re.sub(r"<\|[^|]*\|>", "", s or "").lower())
        r, h = n(ref), n(hyp)
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

    tags = [t for t in ("A_v5_basegpt", "B_v5_zfhgpt", "v2pro_prod")
            if (OUT / t).exists()]
    res = {}
    for tag in tags:
        ps = sorted((OUT / tag).glob("*.wav"))
        sims, cers, durs, f0s, envs, srs = [], [], [], [], [], set()
        for i, p in enumerate(ps):
            x, sr = sf.read(str(p), dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            srs.add(sr)
            durs.append(len(x) / sr)
            sims.append(sim(p))
            w16 = librosa.load(str(p), sr=16000, mono=True)[0]
            r = asr.generate(input=w16, cache={}, language="zh", use_itn=False,
                             batch_size_s=60, merge_vad=True, merge_length_s=15)
            cers.append(cer(TEXTS[i] if i < len(TEXTS) else "",
                            r[0]["text"] if r else ""))
            f = f0_stats(x, sr)
            if f:
                f0s.append(f)
            envs.append(env_dynamics(x, sr))
        res[tag] = {
            "n": len(ps), "sr": sorted(srs),
            "sim": round(float(np.mean(sims)), 4),
            "cer": round(float(np.mean(cers)), 4),
            "dur_mean": round(float(np.mean(durs)), 3),
            "dur_cv": round(float(np.std(durs) / np.mean(durs)), 4),
            "f0_median": round(float(np.median([f["f0_median"] for f in f0s])), 1) if f0s else None,
            "f0_range_st": round(float(np.mean([f["f0_range_semitone"] for f in f0s])), 2) if f0s else None,
            "env_dyn_db": round(float(np.mean(envs)), 2),
            "_sim": sims, "_cer": cers, "_dur": durs,
        }
        print(f"  {tag:16s} sr={sorted(srs)} SIM={res[tag]['sim']:.4f} "
              f"CER={res[tag]['cer']*100:5.1f}% dur={res[tag]['dur_mean']:.2f}s "
              f"F0={res[tag]['f0_median']}Hz 起伏={res[tag]['f0_range_st']}半音 "
              f"包络动态={res[tag]['env_dyn_db']}dB", flush=True)

    # 配对检验
    def paired(dv):
        dv = np.asarray(dv, dtype=float)
        n = len(dv)
        if n < 3:
            return {"mean": None, "t": None, "sig": False}
        se = dv.std(ddof=1) / np.sqrt(n)
        t = float(dv.mean() / se) if se > 0 else 0.0
        crit = {11: 2.201, 12: 2.179}.get(n, 2.2)
        return {"mean": round(float(dv.mean()), 4), "t": round(t, 2),
                "crit": crit, "sig": bool(abs(t) > crit),
                "win": int((dv > 0).sum()), "lose": int((dv < 0).sum())}

    print("\n" + "=" * 78)
    print("配对检验（同 12 句）")
    print("=" * 78)
    tests = {}
    pairs = [("B_v5_zfhgpt", "A_v5_basegpt", "换微调GPT vs 底模GPT（都在v5上）"),
             ("B_v5_zfhgpt", "v2pro_prod", "v5+微调GPT vs v2Pro生产"),
             ("A_v5_basegpt", "v2pro_prod", "v5+底模GPT vs v2Pro生产（上一轮配置）")]
    for x, y, label in pairs:
        if x not in res or y not in res:
            continue
        print(f"\n{label}")
        row = {}
        for key, name in (("_sim", "SIM"), ("_cer", "CER"), ("_dur", "时长")):
            p = paired(np.array(res[x][key]) - np.array(res[y][key]))
            row[name] = p
            print(f"  {name:5s} 差={p['mean']:+}  t={p['t']}  临界={p.get('crit')}  "
                  f"{'显著' if p['sig'] else '不显著'}  "
                  f"(胜{p.get('win')}/负{p.get('lose')})")
        tests[f"{x}_vs_{y}"] = row

    (OUT / "gpt_fix_metrics.json").write_text(json.dumps(
        {"results": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in res.items()},
         "paired": tests}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果: {OUT/'gpt_fix_metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
