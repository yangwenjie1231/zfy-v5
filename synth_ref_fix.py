# -*- coding: utf-8 -*-
"""用**正确的参考音频**重测：验证「语气差」是否由 ref_audio 错配导致。

已确认的根因
------------
我上一轮用的参考音频 `庄方宜_TTS训练集/ref_audio/ref.wav`：
  · ref_text 声明「多亏你及时制止那些寨民袭击武陵城…回赠一些物资。」
  · 实际念的是「那里常年监测着周边环境的深度毒数是抵抗秦世朝的第一线」
  · 前缀一致率 **0%**，语速 6.94 字/秒（异常快）
  · 数据集 README 说该文件 6.95s，实测只有 4.32s（被换过/截断）

参考音频与 ref_text 不匹配时，模型会去"补完"参考文本，
表现为**串词**和**语气怪异** —— 这与用户反馈吻合。

生产用的 `default.wav`（006_更换武器）是**正确的**：前缀一致 100%、
语速 3.77 字/秒。

本脚本对比四种组合（同 12 句、同 seed）：
  P1 = v5  + 生产参考(default.wav)      <- 推荐
  P2 = v5  + 数据集参考(ref.wav)        <- 我上一轮的配置
  P3 = v2Pro + 生产参考                 <- 生产基准
  P4 = v2Pro + 数据集参考               <- 验证参考音频本身的影响
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

V5_ROOT = Path(r"D:\A\zfy-v5")
GSV = V5_ROOT / "GPT-SoVITS"
OLD = Path(r"D:\A\voiceclone\GPT-SoVITS")
OUT = V5_ROOT / "out" / "ref_fix"

DATASET = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频\庄方宜_TTS训练集")
BAD_REF = DATASET / "ref_audio" / "ref.wav"
BAD_REF_TEXT = "多亏你及时制止那些寨民袭击武陵城，为了感谢你们，我想回赠一些物资。"
GOOD_REF = Path(r"D:\A\voiceclone\zfh-models-v2pro\ref\default.wav")
GOOD_REF_TEXT = "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？"

ZFH_GPT = r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh-e4.ckpt"
S1V3 = str(GSV / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt")
V2PRO_SOV = r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh_v2pro_e12_s384.pth"

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


def use_v5_path():
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


def use_old_path():
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


def gen(version, t2s, vits, ref, ref_text, tag, is_half=True, steps=None, cfg=None):
    import torch
    if version.startswith("v5"):
        use_v5_path()
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        base = GSV
    else:
        use_old_path()
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
    print(f"  [{tag}] {version}  ref={Path(ref).name}", flush=True)
    for i, t in enumerate(TEXTS):
        p = d / f"{i:02d}.wav"
        if p.exists():
            continue
        inp = {"text": t, "text_lang": "zh", "ref_audio_path": str(ref),
               "prompt_text": ref_text, "prompt_lang": "zh",
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
    ap.add_argument("--epoch", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    w = next((GSV / "SoVITS_weights_v5").glob(f"*_e{args.epoch}_*.pth"), None)
    if w is None:
        print(f"找不到 e{args.epoch}")
        return 1

    print("P1: v5 + 生产参考（正确对齐）")
    gen("v5dev", ZFH_GPT, str(w), GOOD_REF, GOOD_REF_TEXT, "P1_v5_goodref", True, 32, 1.30)
    print("P2: v5 + 数据集参考（我上一轮用的，错配）")
    gen("v5dev", ZFH_GPT, str(w), BAD_REF, BAD_REF_TEXT, "P2_v5_badref", True, 32, 1.30)
    print("P3: v2Pro + 生产参考（生产基准）")
    gen("v2Pro", ZFH_GPT, V2PRO_SOV, GOOD_REF, GOOD_REF_TEXT, "P3_v2pro_goodref", False)
    print("P4: v2Pro + 数据集参考（错配）")
    gen("v2Pro", ZFH_GPT, V2PRO_SOV, BAD_REF, BAD_REF_TEXT, "P4_v2pro_badref", False)

    print(f"\n输出: {OUT}")
    for t in ("P1_v5_goodref", "P2_v5_badref", "P3_v2pro_goodref", "P4_v2pro_badref"):
        print(f"  {t:22s} {len(list((OUT/t).glob('*.wav')))} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
