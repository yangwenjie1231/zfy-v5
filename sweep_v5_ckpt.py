# -*- coding: utf-8 -*-
"""v5 LoRA 检查点扫描：逐 epoch 生成同一批句子，用客观指标选最佳。

为什么必须逐 epoch 扫（而不是直接取最后一个）：
    之前 v2Pro 那次实验里，"看起来更好"的 e10 在配对检验下并不显著。
    而且 LoRA 的**最佳 epoch 通常早于全量微调**（社区报告 3-8 之间），
    取 e8 可能已经过拟合。

指标（全部同文本、同参考、同 seed）：
  1. **12-16k / 8-12k 频带比** —— 高频延伸（v5 的卖点之一）
  2. **ASR CER**（SenseVoice）—— 可懂度
  3. **说话人相似度** —— 用 ERes2NetV2（仓库自带，与 v2Pro 同口径）
  4. **时长稳定性** —— 与参考时长比是否离谱

用法：
    python sweep_v5_ckpt.py [--n 12]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

V5_ROOT = Path(r"D:\A\zfy-v5")
GSV = V5_ROOT / "GPT-SoVITS"
OUT = V5_ROOT / "out" / "sweep_v5"
DATASET = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频\庄方宜_TTS训练集")
REF = DATASET / "ref_audio" / "ref.wav"
REF_TEXT = "多亏你及时制止那些寨民袭击武陵城，为了感谢你们，我想回赠一些物资。"

SEED = 42
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    args = ap.parse_args()
    texts = TEXTS[: args.n]

    setup_path()
    import torch
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    from GPT_SoVITS.module.v5_inference import resolve_sampling
    from funasr import AutoModel
    import librosa

    ckpts = sorted((GSV / "SoVITS_weights_v5").glob("*.pth"),
                   key=lambda p: int(re.search(r"_e(\d+)_", p.name).group(1)))
    print(f"检查点 {len(ckpts)} 个: {[p.name for p in ckpts]}\n")

    print("加载 ASR（SenseVoice）…", flush=True)
    asr = AutoModel(model="iic/SenseVoiceSmall", device="cuda:0",
                    disable_update=True, cache_dir=r"D:\A\voiceclone\models")

    # 说话人相似度：用仓库自带的 ERes2NetV2（与 v2Pro 口径一致）
    print("加载说话人编码器（ERes2NetV2）…", flush=True)
    from sv import SV
    sv = SV("cuda", False)
    # `sv.py:18-20` 只在 is_half==False 时 .to(device)，
    # 所以模型在 cuda 上；输入张量**必须也搬到 cuda**，否则
    # RuntimeError: Input type (torch.FloatTensor) and weight type
    #               (torch.cuda.FloatTensor) should be the same
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    def emb_of(wav32: np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(wav32)).unsqueeze(0).to(DEV)
        with torch.no_grad():
            e = sv.compute_embedding3(t)
        return e.detach().cpu().flatten().numpy().astype(np.float64)

    ref_emb = emb_of(librosa.load(str(REF), sr=32000, mono=True)[0])

    def sim(p: Path):
        w = librosa.load(str(p), sr=32000, mono=True)[0]
        b = emb_of(w)
        a = ref_emb
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    rows = []
    for ck in ckpts:
        epoch = int(re.search(r"_e(\d+)_", ck.name).group(1))
        d = OUT / f"e{epoch:02d}"
        d.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {ck.name} ===", flush=True)
        cfg = {"custom": {
            "device": "cuda", "is_half": True, "version": "v5dev",
            "t2s_weights_path": str(GSV / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"),
            "vits_weights_path": str(ck),
            "bert_base_path": str(GSV / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": str(GSV / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"),
        }}
        tts = TTS(TTS_Config(cfg))
        steps, cfr = resolve_sampling("v5dev", args.steps, args.cfg)

        brs, cers, sims, durs = [], [], [], []
        for i, t in enumerate(texts):
            p = d / f"{i:02d}.wav"
            if not p.exists():
                gen = tts.run({
                    "text": t, "text_lang": "zh",
                    "ref_audio_path": str(REF), "prompt_text": REF_TEXT,
                    "prompt_lang": "zh", "top_k": 15, "temperature": 1.0,
                    "repetition_penalty": 1.35, "text_split_method": "cut1",
                    "batch_size": 1, "speed_factor": 1.0, "seed": SEED,
                    "sample_steps": steps, "cfg_rate": cfr,
                    "parallel_infer": True,
                })
                chunks, sr = [], None
                for it in gen:
                    if it is None:
                        continue
                    sr, a = it
                    chunks.append(np.asarray(a))
                audio = np.concatenate(chunks)
                sf.write(str(p), audio, sr)

            br = band_ratio(p)
            if br is not None:
                brs.append(br)
            x, sr = sf.read(str(p), dtype="float32", always_2d=True)
            durs.append(len(x) / sr)
            w16 = librosa.load(str(p), sr=16000, mono=True)[0]
            res = asr.generate(input=w16, cache={}, language="zh",
                               use_itn=False, batch_size_s=60,
                               merge_vad=True, merge_length_s=15)
            hyp = res[0]["text"] if res else ""
            cers.append(cer(t, hyp))
            try:
                sims.append(sim(p))
            except Exception:  # noqa: BLE001
                pass

        row = {
            "epoch": epoch, "weights": ck.name,
            "band_ratio": round(float(np.mean(brs)), 3) if brs else None,
            "cer": round(float(np.mean(cers)), 4),
            "sim": round(float(np.mean(sims)), 4) if sims else None,
            "dur_mean": round(float(np.mean(durs)), 3),
            "dur_cv": round(float(np.std(durs) / (np.mean(durs) + 1e-9)), 4),
        }
        rows.append(row)
        print(f"  band={row['band_ratio']}  CER={row['cer']*100:.1f}%  "
              f"SIM={row['sim']}  dur={row['dur_mean']}s", flush=True)
        del tts
        torch.cuda.empty_cache()

    # 参考：底模（未微调）作为对照
    print("\n" + "=" * 78)
    print(f"{'epoch':>6s} {'12-16k/8-12k':>13s} {'CER':>8s} {'SIM':>8s} {'时长CV':>8s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['epoch']:6d} {r['band_ratio']:13.2f} {r['cer']*100:7.1f}% "
              f"{r['sim']:8.4f} {r['dur_cv']:8.4f}")

    # 综合评分：CER 越低越好、SIM 越高越好、band 越高越好
    def score(r):
        if r["band_ratio"] is None or r["sim"] is None:
            return -1e9
        return (-r["cer"]) * 2.0 + r["sim"] * 2.0 + r["band_ratio"] * 0.2

    best = max(rows, key=score)
    print(f"\n综合最佳: epoch {best['epoch']}  ({best['weights']})")
    (OUT / "sweep.json").write_text(
        json.dumps({"rows": rows, "best_epoch": best["epoch"],
                    "best_weights": best["weights"],
                    "n_texts": len(texts), "steps": args.steps,
                    "cfg": args.cfg}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"结果: {OUT/'sweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
