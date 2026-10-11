# -*- coding: utf-8 -*-
"""在**同一批 20 句**上扫 r32 的全部 8 个 epoch，找出真正的"最佳 v5"。

## 为什么做这个

用户听感反馈：**v5 最佳的效果不错**。而我的客观指标（ERes2NetV2 SIM）
说 v2Pro 明显更好（0.8888 vs 0.8236）——**指标与听感不一致**。

可能原因：
  · SIM 衡量的是"与参考音频的说话人嵌入余弦相似度"，
    不等于人耳感知的"像不像 / 好不好听"
  · v5 有 48kHz 带宽优势（22.8kHz vs v2Pro 16kHz 硬上限），
    高频细节更丰富 → 听感更清晰自然，即使音色匹配度略低
  · 上一轮 r32 的 8 个 epoch 是在**旧口径（12 句、错配参考、底模 GPT）**
    下测的，与现在的 20 句口径不可比

所以要在统一口径下重扫，确认：
  1. r32 的哪个 epoch 最好（可能不是 e8）
  2. 是否存在"指标低但听感好"的稳定现象

用法：
    python sweep_r32_all.py --n 20
"""
from __future__ import annotations

import argparse
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
OUT = V5 / "out" / "r32_sweep"

GOOD_REF = Path(r"D:\A\voiceclone\zfh-models-v2pro\ref\default.wav")
GOOD_REF_TEXT = "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？"
ZFH_GPT = r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh-e4.ckpt"
V2PRO_SOV = r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh_v2pro_e12_s384.pth"

TEXTS = [
    "迎敌或是迎客，都得整装后再出发。",
    "出门前，要不要先喝杯茶提提神？",
    "让我想想。",
    "好。",
    "武陵城外有一片梨树林，每年春天都会开满白色的花。",
    "她就在那阵花香里，只是他自己不知道。",
    "学得越多，就越是清楚，自己学到的不过沧海一粟。",
    "上次外出执行任务……都是好几年前了，但愿别生疏。",
    "新的是更厉害些，可那旧的用惯了，确实……也不舍得。",
    "再好的武器，也得朝夕相处磨合上几天。",
    "你找我帮忙，想必是遇上棘手的问题了，直说就好。",
    "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。",
    "嗯……我理一理。",
    "三成火力，足够了。",
    "这件事我三点五十分再去处理。",
    "武陵城的梨花开了，要不要一起去看看？",
    "别紧张，多难多苦，我们都会一直陪在你身边。",
    "冬天总会过去，春天总会再来。",
    "我确实是第一次来帝江号。",
    "下个春天的好风景，我们一起去看。",
]
SEED = 42


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


def synth_v5(weights: Path, tag: str, texts, steps=32, cfg=1.30):
    use_v5()
    import torch
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    cf = {"custom": {
        "device": "cuda", "is_half": True, "version": "v5dev",
        "t2s_weights_path": ZFH_GPT, "vits_weights_path": str(weights),
        "bert_base_path": str(GSV / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"),
        "cnhuhbert_base_path": str(GSV / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"),
    }}
    tts = TTS(TTS_Config(cf))
    for i, t in enumerate(texts):
        p = d / f"{i:02d}.wav"
        if p.exists():
            continue
        ch, sr = [], None
        for it in tts.run({
            "text": t, "text_lang": "zh", "ref_audio_path": str(GOOD_REF),
            "prompt_text": GOOD_REF_TEXT, "prompt_lang": "zh",
            "top_k": 15, "temperature": 1.0, "repetition_penalty": 1.35,
            "text_split_method": "cut1", "batch_size": 1, "speed_factor": 1.0,
            "seed": SEED, "sample_steps": steps, "cfg_rate": cfg,
            "parallel_infer": True,
        }):
            if it is None:
                continue
            sr, a = it
            ch.append(np.asarray(a))
        sf.write(str(p), np.concatenate(ch), sr)
    del tts
    torch.cuda.empty_cache()


def band_ratio(path: Path, nfft=2048):
    """12-16k 相对 8-12k 的 dB（带宽延伸指标）。"""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if 16000 > sr / 2:
        return None
    hop = nfft // 4
    nf = 1 + (len(x) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(nf)[:, None]
    P = (np.abs(np.fft.rfft(x[idx] * np.hanning(nfft), axis=1)) ** 2).mean(axis=0)
    f = np.fft.rfftfreq(nfft, 1 / sr)
    a = float(np.mean(P[(f >= 8000) & (f < 12000)])) + 1e-20
    b = float(np.mean(P[(f >= 12000) & (f < 16000)])) + 1e-20
    return 10 * np.log10(b / a)


def norm(s):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=1.30)
    args = ap.parse_args()
    texts = TEXTS[: args.n]
    OUT.mkdir(parents=True, exist_ok=True)

    wdir = GSV / "SoVITS_weights_v5"
    cks = sorted(wdir.glob("*.pth"),
                 key=lambda p: int(re.search(r"_e(\d+)_", p.name).group(1)))
    print(f"r32 检查点 {len(cks)} 个：{[p.name for p in cks]}")
    print(f"句子数 {len(texts)}\n")

    for ck in cks:
        e = int(re.search(r"_e(\d+)_", ck.name).group(1))
        print(f"合成 r32_e{e:02d} …", flush=True)
        synth_v5(ck, f"r32_e{e:02d}", texts, args.steps, args.cfg)

    # 评测
    use_v5()
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

    rows = []
    for d in sorted(OUT.iterdir()):
        if not d.is_dir():
            continue
        ps = sorted(d.glob("*.wav"))
        if len(ps) < len(texts):
            print(f"跳过 {d.name}（{len(ps)}/{len(texts)}）")
            continue
        cs, ss, brs = [], [], []
        for i, p in enumerate(ps[:len(texts)]):
            w16 = librosa.load(str(p), sr=16000, mono=True)[0]
            r = asr.generate(input=w16, cache={}, language="zh", use_itn=False,
                             batch_size_s=60, merge_vad=True, merge_length_s=15)
            cs.append(cer(texts[i], r[0]["text"] if r else ""))
            ss.append(sim(p))
            br = band_ratio(p)
            if br is not None:
                brs.append(br)
        rows.append({"tag": d.name, "cer": round(float(np.mean(cs)), 4),
                     "sim": round(float(np.mean(ss)), 4),
                     "band": round(float(np.mean(brs)), 2) if brs else None,
                     "_sim": ss, "_cer": cs})

    print("\n" + "=" * 70)
    print(f"r32 全部 epoch（{len(texts)} 句，同参考/同 GPT=zfh-e4）")
    print("=" * 70)
    print(f"{'epoch':10s} {'CER':>8s} {'SIM':>9s} {'12-16k/8-12k':>13s}")
    print("-" * 70)
    for r in sorted(rows, key=lambda x: x["tag"]):
        b = f"{r['band']:13.2f}" if r["band"] is not None else f"{'--':>13s}"
        print(f"{r['tag']:10s} {r['cer']*100:7.1f}% {r['sim']:9.4f} {b}")

    if rows:
        best = max(rows, key=lambda r: r["sim"])
        print(f"\nSIM 最高: {best['tag']}  ({best['sim']:.4f})")
        sims = [r["sim"] for r in rows]
        print(f"SIM 极差: {max(sims)-min(sims):.4f}（{min(sims):.4f} ~ {max(sims):.4f}）")

    (OUT / "sweep_r32.json").write_text(json.dumps(
        {"rows": [{k: v for k, v in r.items() if not k.startswith("_")}
                  for r in rows], "texts": texts},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果: {OUT/'sweep_r32.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
