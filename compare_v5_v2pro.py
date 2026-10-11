# -*- coding: utf-8 -*-
"""v5 检查点的**配对显著性检验** + 与 v2Pro 的同口径对比。

为什么要做这一步：
    上面 sweep 的表里，CER 11.7%~19.4%、SIM 0.7979~0.8171 看起来有差异，
    但**这些差异是否真实**必须检验，不能直接取"最高分"。

    我在 v2Pro 那轮就犯过这个错：e10 的 SIM 比 e12 高 0.0157，
    看起来更好，但配对置信区间 [-0.0045, +0.0359] 跨 0 —— 不显著。
    而且 CER 是**双峰分布**（要么≈0 要么≈25%），12 句的样本量
    不足以分辨 10 个百分点以内的差异。

本脚本：
  1. 逐句算 SIM / CER / band，对候选 epoch 做**配对差分 + t 检验**
  2. 用**同一个 ERes2NetV2 验证器**测 v2Pro（消除验证器口径差异）
  3. 输出"哪些差异可信、哪些是噪声"的结论

用法：
    python compare_v5_v2pro.py [--n 12] [--epochs 4,7,8]
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

V5_ROOT = Path(r"D:\A\zfy-v5")
GSV = V5_ROOT / "GPT-SoVITS"
OUT = V5_ROOT / "out" / "fair_compare"
DATASET = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频\庄方宜_TTS训练集")
REF = DATASET / "ref_audio" / "ref.wav"
REF_TEXT = "多亏你及时制止那些寨民袭击武陵城，为了感谢你们，我想回赠一些物资。"

V2PRO_WEIGHTS = Path(r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh_v2pro_e12_s384.pth")
OLD_GSV = Path(r"D:\A\voiceclone\GPT-SoVITS")

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
SEED = 42


def setup_path():
    need = [str(GSV), str(GSV / "GPT_SoVITS"),
            str(GSV / "GPT_SoVITS" / "BigVGAN"),
            str(GSV / "tools"), str(GSV / "tools" / "asr"),
            str(GSV / "tools" / "uvr5")]
    for p in reversed(need):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    os.chdir(str(GSV))


def band_ratio(path: Path, nfft=2048):
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


def paired(dv):
    """配对差分的均值 / SD / t / 判定。"""
    dv = np.asarray(dv, dtype=float)
    n = len(dv)
    if n < 3:
        return {"n": n, "mean": float(dv.mean()) if n else None, "t": None,
                "sig": False, "note": "样本不足"}
    se = dv.std(ddof=1) / np.sqrt(n)
    t = float(dv.mean() / se) if se > 0 else 0.0
    crit = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
            8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
            13: 2.160, 14: 2.145, 15: 2.131}.get(n - 1, 2.0)
    return {"n": n, "mean": round(float(dv.mean()), 4),
            "sd": round(float(dv.std(ddof=1)), 4),
            "t": round(t, 2), "crit": crit,
            "sig": bool(abs(t) > crit),
            "win": int(np.sum(dv > 0)), "lose": int(np.sum(dv < 0))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--epochs", default="4,7,8")
    args = ap.parse_args()
    texts = TEXTS[: args.n]
    epochs = [int(x) for x in args.epochs.split(",")]

    setup_path()
    import torch
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    from funasr import AutoModel
    import librosa
    from sv import SV

    OUT.mkdir(parents=True, exist_ok=True)
    DEV = "cuda"

    print("加载 ASR / 说话人编码器 …", flush=True)
    asr = AutoModel(model="iic/SenseVoiceSmall", device=DEV,
                    disable_update=True, cache_dir=r"D:\A\voiceclone\models")
    sv = SV(DEV, False)

    def emb(wav32):
        t = torch.from_numpy(np.ascontiguousarray(wav32)).unsqueeze(0).to(DEV)
        with torch.no_grad():
            e = sv.compute_embedding3(t)
        return e.detach().cpu().flatten().numpy().astype(np.float64)

    ref_emb = emb(librosa.load(str(REF), sr=32000, mono=True)[0])

    def sim(p: Path):
        b = emb(librosa.load(str(p), sr=32000, mono=True)[0])
        a = ref_emb
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def asr_cer(p: Path, ref_text: str):
        w = librosa.load(str(p), sr=16000, mono=True)[0]
        res = asr.generate(input=w, cache={}, language="zh", use_itn=False,
                           batch_size_s=60, merge_vad=True, merge_length_s=15)
        hyp = res[0]["text"] if res else ""
        return cer(ref_text, hyp), hyp

    def synth_v5(weights: Path, tag: str):
        d = OUT / tag
        d.mkdir(parents=True, exist_ok=True)
        cfg = {"custom": {
            "device": DEV, "is_half": True, "version": "v5dev",
            "t2s_weights_path": str(GSV / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"),
            "vits_weights_path": str(weights),
            "bert_base_path": str(GSV / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": str(GSV / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"),
        }}
        tts = TTS(TTS_Config(cfg))
        paths = []
        for i, t in enumerate(texts):
            p = d / f"{i:02d}.wav"
            if not p.exists():
                gen = tts.run({
                    "text": t, "text_lang": "zh",
                    "ref_audio_path": str(REF), "prompt_text": REF_TEXT,
                    "prompt_lang": "zh", "top_k": 15, "temperature": 1.0,
                    "repetition_penalty": 1.35, "text_split_method": "cut1",
                    "batch_size": 1, "speed_factor": 1.0, "seed": SEED,
                    "sample_steps": 32, "cfg_rate": 1.30, "parallel_infer": True,
                })
                ch, sr = [], None
                for it in gen:
                    if it is None:
                        continue
                    sr, a = it
                    ch.append(np.asarray(a))
                sf.write(str(p), np.concatenate(ch), sr)
            paths.append(p)
        del tts
        torch.cuda.empty_cache()
        return paths

    def synth_v2pro():
        """v2Pro 走旧检出（生产路径），同参考/同 seed。"""
        d = OUT / "v2pro"
        d.mkdir(parents=True, exist_ok=True)
        # 切到旧检出的 path 布局
        need = [str(OLD_GSV), str(OLD_GSV / "GPT_SoVITS"),
                str(OLD_GSV / "GPT_SoVITS" / "BigVGAN"),
                str(OLD_GSV / "tools"), str(OLD_GSV / "tools" / "asr"),
                str(OLD_GSV / "tools" / "uvr5")]
        for p in reversed(need):
            while p in sys.path:
                sys.path.remove(p)
            sys.path.insert(0, p)
        os.chdir(str(OLD_GSV))
        for m in list(sys.modules):
            if m.split(".")[0] in ("GPT_SoVITS", "module", "text", "tools",
                                   "sv", "f5_tts", "TTS_infer_pack"):
                del sys.modules[m]
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS as TTS2, TTS_Config as Cfg2
        cfg = {"custom": {
            "device": DEV, "is_half": False, "version": "v2Pro",
            "t2s_weights_path": str(OLD_GSV / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"),
            "vits_weights_path": str(V2PRO_WEIGHTS),
            "bert_base_path": str(OLD_GSV / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": str(OLD_GSV / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"),
        }}
        tts = TTS2(Cfg2(cfg))
        paths = []
        for i, t in enumerate(texts):
            p = d / f"{i:02d}.wav"
            if not p.exists():
                gen = tts.run({
                    "text": t, "text_lang": "zh",
                    "ref_audio_path": str(REF), "prompt_text": REF_TEXT,
                    "prompt_lang": "zh", "top_k": 15, "temperature": 1.0,
                    "repetition_penalty": 1.35, "text_split_method": "cut1",
                    "batch_size": 1, "speed_factor": 1.0, "seed": SEED,
                })
                ch, sr = [], None
                for it in gen:
                    if it is None:
                        continue
                    sr, a = it
                    ch.append(np.asarray(a))
                sf.write(str(p), np.concatenate(ch), sr)
            paths.append(p)
        del tts
        torch.cuda.empty_cache()
        return paths

    # ---- 合成 ----
    per = {}
    for ep in epochs:
        w = next((GSV / "SoVITS_weights_v5").glob(f"*_e{ep}_*.pth"), None)
        if w is None:
            print(f"跳过 e{ep}: 无权重")
            continue
        print(f"\n合成 v5 e{ep} …", flush=True)
        per[f"v5_e{ep}"] = synth_v5(w, f"v5_e{ep}")

    print("\n合成 v2Pro（旧检出，生产路径）…", flush=True)
    per["v2pro"] = synth_v2pro()

    # ---- 逐句评测 ----
    print("\n逐句评测 …", flush=True)
    metrics = {}
    for tag, ps in per.items():
        brs, cers, sims, durs = [], [], [], []
        for i, p in enumerate(ps):
            br = band_ratio(p)
            if br is not None:
                brs.append(br)
            e, _ = asr_cer(p, texts[i])
            cers.append(e)
            sims.append(sim(p))
            x, sr = sf.read(str(p), dtype="float32", always_2d=True)
            durs.append(len(x) / sr)
        metrics[tag] = {"band": brs, "cer": cers, "sim": sims, "dur": durs,
                        "sr": sr}
        print(f"  {tag:10s} sr={sr} band={np.mean(brs):7.2f} "
              f"CER={np.mean(cers)*100:5.1f}% SIM={np.mean(sims):.4f} "
              f"dur={np.mean(durs):5.2f}s")

    # ---- 配对检验 ----
    print("\n" + "=" * 78)
    print("配对显著性（同 12 句，同参考/同 seed）")
    print("=" * 78)
    tests = {}
    ref_tag = "v2pro"
    for tag in metrics:
        if tag == ref_tag:
            continue
        t_sim = paired(np.array(metrics[tag]["sim"]) - np.array(metrics[ref_tag]["sim"]))
        t_cer = paired(np.array(metrics[ref_tag]["cer"]) - np.array(metrics[tag]["cer"]))
        t_band = paired(np.array(metrics[tag]["band"]) - np.array(metrics[ref_tag]["band"]))
        tests[f"{tag}_vs_v2pro"] = {"sim": t_sim, "cer": t_cer, "band": t_band}
        print(f"\n{tag}  vs  v2Pro")
        print(f"  SIM  差 {t_sim['mean']:+.4f}  t={t_sim['t']} (临界 {t_sim['crit']}) "
              f"胜{t_sim['win']}/负{t_sim['lose']}  -> "
              f"{'显著' if t_sim['sig'] else '不显著'}")
        print(f"  CER  差 {t_cer['mean']:+.4f}  t={t_cer['t']} "
              f"胜{t_cer['win']}/负{t_cer['lose']}  -> "
              f"{'显著' if t_cer['sig'] else '不显著'}")
        print(f"  band 差 {t_band['mean']:+.4f}  t={t_band['t']} "
              f"胜{t_band['win']}/负{t_band['lose']}  -> "
              f"{'显著' if t_band['sig'] else '不显著'}")

    (OUT / "metrics.json").write_text(json.dumps(
        {"texts": texts, "metrics": metrics, "paired_tests": tests,
         "n": len(texts)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果: {OUT/'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
