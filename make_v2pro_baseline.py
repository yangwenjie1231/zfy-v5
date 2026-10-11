# -*- coding: utf-8 -*-
"""本地生成 v2Pro 生产版对照音频（与云端同一批 10 句）。

## 为什么
云端那 20 个 wav 只有：
  · e16（未合并，时长异常）
  · e16_merged（修复后）
**缺 v2Pro 基准** —— 而 v2Pro 是当前生产在用的模型，是判断"v5 全量微调有没有更好"
的唯一参照。

## 参数对齐
必须与云端 `cloud_synth_sweep.py` 完全一致，否则不可比：
    top_k 15 / temperature 1.0 / repetition_penalty 1.35
    text_split_method cut1 / batch_size 1 / speed_factor 1.0
    seed 42 / parallel_infer True / is_half False
参考音频：ref/default.wav（生产验证过的那条）
GPT：ref/zfh-e4.ckpt

## 输出
    D:\\A\\zfy-v5\\dist\\baseline_v2pro\\NN_类型_v2pro.wav
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"D:\A\zfy-v5\GPT-SoVITS")
REF = ROOT / "ref"
OUT = Path(r"D:\A\zfy-v5\dist\baseline_v2pro")

TEXTS = [
    ("短", "让我想想。"),
    ("短", "好。"),
    ("中", "迎敌或是迎客，都得整装后再出发。"),
    ("中", "出门前，要不要先喝杯茶提提神？"),
    ("中", "嗯……我理一理。"),
    ("长", "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？"),
    ("长", "武陵城外有一片梨树林，每年春天都会开满白色的花。"),
    ("长", "她就在那阵花香里，只是他自己不知道。"),
    ("长", "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。"),
    ("数字", "这件事我三点五十分再去处理。"),
]
SEED = 42

# ---- sys.path（顺序与 cloud_synth_sweep.setup_paths 一致）----
for p in reversed([str(ROOT), str(ROOT / "GPT_SoVITS"),
                   str(ROOT / "GPT_SoVITS" / "BigVGAN"),
                   str(ROOT / "tools"), str(ROOT / "tools" / "asr"),
                   str(ROOT / "tools" / "uvr5")]):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
os.chdir(str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

PRE = ROOT / "GPT_SoVITS" / "pretrained_models"

print("=" * 78)
print("本地生成 v2Pro 对照音频")
print("=" * 78)

ref_txt = (REF / "ref.txt").read_text(encoding="utf-8").strip()
print(f"  参考音频: {REF / 'default.wav'}")
print(f"  参考文本: {ref_txt[:50]}…")
print(f"  GPT 权重: {(REF / 'zfh-e4.ckpt').name}")
print(f"  v2Pro   : {(REF / 'v2pro_sovits.pth').name}")
print()

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

cfg = {
    "custom": {
        "device": "cuda",
        "is_half": False,
        "version": "v2Pro",
        "t2s_weights_path": str(REF / "zfh-e4.ckpt"),
        "vits_weights_path": str(REF / "v2pro_sovits.pth"),
        "bert_base_path": str(PRE / "chinese-roberta-wwm-ext-large"),
        "cnhuhbert_base_path": str(PRE / "chinese-hubert-base"),
    }
}
print("  构造 TTS …")
tts = TTS(TTS_Config(cfg))
print("  ✓ 完成")
print()

OUT.mkdir(parents=True, exist_ok=True)
n_ok = 0
for i, (kind, text) in enumerate(TEXTS, 1):
    try:
        gen = tts.run({
            "text": text,
            "text_lang": "zh",
            "ref_audio_path": str(REF / "default.wav"),
            "prompt_text": ref_txt,
            "prompt_lang": "zh",
            "top_k": 15,
            "temperature": 1.0,
            "repetition_penalty": 1.35,
            "text_split_method": "cut1",
            "batch_size": 1,
            "speed_factor": 1.0,
            "seed": SEED,
            "parallel_infer": True,
        })
        chunks, sr = [], None
        for item in gen:
            if item is None:
                continue
            s, a = item
            sr = s
            chunks.append(a)
        if not chunks:
            raise RuntimeError("TTS.run 无产出")
        audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        fn = OUT / f"{i:02d}_{kind}_v2pro.wav"
        sf.write(str(fn), audio, sr)
        n_ok += 1
        print(f"  {i:2d}/10 ok  {fn.name:26s} {len(audio)/sr:.2f}s @{sr}Hz")
    except Exception as e:  # noqa: BLE001
        print(f"  {i:2d}/10 失败 {type(e).__name__}: {str(e)[:130]}")

print()
print(f"  完成 {n_ok}/10，输出: {OUT}")
