# -*- coding: utf-8 -*-
"""用**微调过的 GPT(zfh-e4)** 配 v5 SoVITS 合成，验证语气是否改善。

问题诊断
--------
用户反馈「v5 效果很差，特别是语气处理」。

根因怀疑：语气/韵律由 **GPT 生成的语义 token 序列**决定，
SoVITS 只把 token 还原成波形（决定音色）。

  · 我训练+评测 v5 时用的是**未微调底模 `s1v3.ckpt`** → 语气是底模的语气
  · 生产 v2Pro 用的是**微调过的 `zfh-e4.ckpt`** → 语气是庄方宜的语气

已核实两者**结构完全兼容**（295 键 / 1025 词表 / 732 音素 / 512 维 / 24 层
/config.model 逐项一致），所以可以把 zfh-e4 直接配到 v5 SoVITS 上。

本脚本合成三组做对照：
  A = v5 SoVITS + s1v3 底模 GPT      （我上一轮的配置）
  B = v5 SoVITS + zfh-e4 微调 GPT    （本次修复）
  C = v2Pro + zfh-e4（生产基准）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

V5_ROOT = Path(r"D:\A\zfy-v5")
GSV = V5_ROOT / "GPT-SoVITS"
OUT = V5_ROOT / "out" / "gpt_fix"
DATASET = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频\庄方宜_TTS训练集")
REF = DATASET / "ref_audio" / "ref.wav"
REF_TEXT = "多亏你及时制止那些寨民袭击武陵城，为了感谢你们，我想回赠一些物资。"
ZFH_GPT = r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh-e4.ckpt"
S1V3 = str(GSV / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt")

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


def setup_v5_path():
    need = [str(GSV), str(GSV / "GPT_SoVITS"),
            str(GSV / "GPT_SoVITS" / "BigVGAN"),
            str(GSV / "tools"), str(GSV / "tools" / "asr"),
            str(GSV / "tools" / "uvr5")]
    for p in reversed(need):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    os.chdir(str(GSV))


def run_tts(tts, text, steps=32, cfg=1.30, **extra):
    inputs = {
        "text": text, "text_lang": "zh",
        "ref_audio_path": str(REF), "prompt_text": REF_TEXT,
        "prompt_lang": "zh", "top_k": 15, "temperature": 1.0,
        "repetition_penalty": 1.35, "text_split_method": "cut1",
        "batch_size": 1, "speed_factor": 1.0, "seed": SEED,
        "parallel_infer": True,
    }
    inputs.update(extra)
    if steps is not None:
        inputs["sample_steps"] = steps
        inputs["cfg_rate"] = cfg
    ch, sr = [], None
    for it in tts.run(inputs):
        if it is None:
            continue
        sr, a = it
        ch.append(np.asarray(a))
    return sr, np.concatenate(ch)


def synth_v5(weights: Path, gpt: str, tag: str, steps=32, cfg=1.30):
    setup_v5_path()
    import torch
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    cfgd = {"custom": {
        "device": "cuda", "is_half": True, "version": "v5dev",
        "t2s_weights_path": gpt,
        "vits_weights_path": str(weights),
        "bert_base_path": str(GSV / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"),
        "cnhuhbert_base_path": str(GSV / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"),
    }}
    tts = TTS(TTS_Config(cfgd))
    print(f"  [{tag}] GPT={Path(gpt).name}  SoVITS={weights.name}")
    for i, t in enumerate(TEXTS):
        p = d / f"{i:02d}.wav"
        if p.exists():
            continue
        sr, a = run_tts(tts, t, steps, cfg)
        sf.write(str(p), a, sr)
    del tts
    torch.cuda.empty_cache()


def synth_v2pro(tag="v2pro_prod"):
    """生产 v2Pro（旧检出 + zfh-e4 GPT + v2Pro SoVITS）。"""
    OLD = Path(r"D:\A\voiceclone\GPT-SoVITS")
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
    import torch
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    cfgd = {"custom": {
        "device": "cuda", "is_half": False, "version": "v2Pro",
        "t2s_weights_path": ZFH_GPT,
        "vits_weights_path": r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh_v2pro_e12_s384.pth",
        "bert_base_path": str(OLD / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"),
        "cnhuhbert_base_path": str(OLD / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"),
    }}
    tts = TTS(TTS_Config(cfgd))
    print(f"  [{tag}] GPT=zfh-e4.ckpt  SoVITS=zfh_v2pro_e12_s384.pth")
    for i, t in enumerate(TEXTS):
        p = d / f"{i:02d}.wav"
        if p.exists():
            continue
        sr, a = run_tts(tts, t, steps=None)
        sf.write(str(p), a, sr)
    del tts
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=1.30)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    w = next((GSV / "SoVITS_weights_v5").glob(f"*_e{args.epoch}_*.pth"), None)
    if w is None:
        print(f"找不到 e{args.epoch} 权重")
        return 1

    print("=== A: v5 + 底模 GPT（上一轮配置）===")
    synth_v5(w, S1V3, "A_v5_basegpt", args.steps, args.cfg)
    print("=== B: v5 + zfh-e4 微调 GPT（本次修复）===")
    synth_v5(w, ZFH_GPT, "B_v5_zfhgpt", args.steps, args.cfg)
    print("=== C: v2Pro 生产 ===")
    synth_v2pro()

    print(f"\n输出: {OUT}")
    for t in ("A_v5_basegpt", "B_v5_zfhgpt", "v2pro_prod"):
        n = len(list((OUT / t).glob("*.wav")))
        print(f"  {t:16s} {n} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
