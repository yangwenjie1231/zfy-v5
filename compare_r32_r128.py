# -*- coding: utf-8 -*-
"""r32 vs r128 vs v2Pro 同口径对比（同样 20 句、同参考、同 GPT）。

## 为什么要做这个

`eval_retrain.py` 的 r128 结果（20 句）与上一轮 r32 的结果（12 句）
**文本集不同**，CER 不可直接比（CER 是双峰分布，强依赖句子）。
SIM 也可能受内容影响。所以必须把 r32 拉到**同一批 20 句**上重测。

## 要回答

  1. 提高 LoRA 秩（32 -> 128）到底有没有收益？
  2. v5 的 LoRA 上限离 v2Pro 全量微调差多远？

用法：
    python compare_r32_r128.py --n 20
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
OUT = V5 / "out" / "r32_vs_r128"

GOOD_REF = Path(r"D:\A\voiceclone\zfh-models-v2pro\ref\default.wav")
GOOD_REF_TEXT = "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？"
ZFH_GPT = r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh-e4.ckpt"
V2PRO_SOV = r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh_v2pro_e12_s384.pth"
R32_E8 = GSV / "SoVITS_weights_v5" / "zfy_v5_e8_s3256_l32.pth"

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


def synth_v2pro(tag: str, texts):
    use_old()
    import torch
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    cf = {"custom": {
        "device": "cuda", "is_half": False, "version": "v2Pro",
        "t2s_weights_path": ZFH_GPT, "vits_weights_path": V2PRO_SOV,
        "bert_base_path": str(OLD / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"),
        "cnhuhbert_base_path": str(OLD / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"),
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
            "seed": SEED, "parallel_infer": True,
        }):
            if it is None:
                continue
            sr, a = it
            ch.append(np.asarray(a))
        sf.write(str(p), np.concatenate(ch), sr)
    del tts
    torch.cuda.empty_cache()


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
    ap.add_argument("--r128-epochs", default="12,20,24")
    args = ap.parse_args()
    texts = TEXTS[: args.n]
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = [("r32_e8", R32_E8, None)]
    for e in args.r128_epochs.split(","):
        w = next((GSV / "SoVITS_weights_v5_r128").glob(f"*_e{e}_*.pth"), None)
        if w:
            jobs.append((f"r128_e{int(e):02d}", w, None))

    for tag, w, _ in jobs:
        if not w.exists():
            print(f"跳过 {tag}: 缺权重 {w}")
            continue
        print(f"合成 {tag} …", flush=True)
        synth_v5(w, tag, texts)

    print("合成 v2pro …", flush=True)
    synth_v2pro("v2pro", texts)

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
        cs, ss, lr = [], [], []
        for i, p in enumerate(ps[:len(texts)]):
            w16 = librosa.load(str(p), sr=16000, mono=True)[0]
            r = asr.generate(input=w16, cache={}, language="zh", use_itn=False,
                             batch_size_s=60, merge_vad=True, merge_length_s=15)
            hyp = r[0]["text"] if r else ""
            cs.append(cer(texts[i], hyp))
            lr.append(len(norm(hyp)) / max(len(norm(texts[i])), 1))
            ss.append(sim(p))
        rows.append({"tag": d.name, "cer": round(float(np.mean(cs)), 4),
                     "sim": round(float(np.mean(ss)), 4),
                     "len_ratio": round(float(np.mean(lr)), 3),
                     "_cer": cs, "_sim": ss})

    print("\n" + "=" * 74)
    print(f"同口径对比（{len(texts)} 句，同参考/同 GPT=zfh-e4）")
    print("=" * 74)
    print(f"{'配置':12s} {'CER':>8s} {'SIM':>9s} {'字数比':>8s}")
    print("-" * 74)
    for r in sorted(rows, key=lambda x: x["tag"]):
        print(f"{r['tag']:12s} {r['cer']*100:7.1f}% {r['sim']:9.4f} "
              f"{r['len_ratio']:8.2f}")

    ref = next((r for r in rows if r["tag"] == "v2pro"), None)
    r32 = next((r for r in rows if r["tag"] == "r32_e8"), None)
    if ref:
        print("\n与 v2Pro（全量微调）的配对检验：")
        for r in rows:
            if r["tag"] == "v2pro":
                continue
            ds = np.array(r["_sim"]) - np.array(ref["_sim"])
            n = len(ds)
            t = ds.mean() / (ds.std(ddof=1) / np.sqrt(n)) if ds.std(ddof=1) > 0 else 0
            crit = 2.093
            print(f"  {r['tag']:12s} SIM差={ds.mean():+.4f}  t={t:6.2f}  "
                  f"{'显著' if abs(t) > crit else '不显著'}")
    if r32:
        print("\n与 r32（上一轮 LoRA）的配对检验（检验提秩是否有收益）：")
        for r in rows:
            if r["tag"] in ("v2pro", "r32_e8"):
                continue
            ds = np.array(r["_sim"]) - np.array(r32["_sim"])
            n = len(ds)
            t = ds.mean() / (ds.std(ddof=1) / np.sqrt(n)) if ds.std(ddof=1) > 0 else 0
            print(f"  {r['tag']:12s} SIM差={ds.mean():+.4f}  t={t:6.2f}  "
                  f"{'显著' if abs(t) > 2.093 else '不显著'}")

    (OUT / "compare.json").write_text(json.dumps(
        {"rows": [{k: v for k, v in r.items() if not k.startswith("_")}
                  for r in rows], "texts": texts},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果: {OUT/'compare.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
