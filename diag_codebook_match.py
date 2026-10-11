# -*- coding: utf-8 -*-
"""诊断时长异常（11.40s vs 正常 ~1.2s）：GPT 码本 vs SoVITS 量化器是否匹配。

## 症状
    T2S Decoding EOS [213 -> 494]     <- 281 个语义 token
    01_短_e16.wav  11.40s @48000Hz
「让我想想。」5 个字正常约 1.2s。**过度生成约 9 倍**。

## 核心假设
GPT-SoVITS 的语义 token 来自 CNHuBERT -> **量化器**（在 SoVITS 里）。
而 GPT 的 `ar_audio_embedding.weight` 是**同一套码本**（1024 x 512）。
推理时 GPT 预测 token -> SoVITS 用**自己的**量化器解码。

若两者码本不一致 -> token 语义错位 -> AR 找不到 EOS 位置 -> 过度生成。

- `zfh-e4.ckpt` 是从 **v2 底模**微调的（见 tmp_s1.yaml）
- v5 的量化器来自 **s2Gv5dev.pth**
- 官方 config.py：v3/v4/v5dev/v5turbo/v2Pro **都用 s1v3.ckpt**

本脚本对比四处码本，并核对 25Hz/50Hz 帧率。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
sys.path.insert(0, str(GSV))
sys.path.insert(0, str(GSV / "GPT_SoVITS"))

import torch  # noqa: E402

from process_ckpt import load_sovits_new  # noqa: E402

print("=" * 78)
print("1. GPT 侧的码本（ar_audio_embedding）")
print("=" * 78)
GPTS = [
    ("zfh-e4.ckpt (v2 底模微调)",
     Path(r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh-e4.ckpt")),
    ("s1v3.ckpt (v5 官方配对)",
     GSV / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"),
]
gpt_cb = {}
for label, p in GPTS:
    if not p.exists():
        print(f"  {label}: 不存在")
        continue
    d = torch.load(str(p), map_location="cpu", weights_only=False)
    w = d["weight"]
    cand = [k for k in w if "ar_audio_embedding" in k or "audio_embedding" in k]
    print(f"\n  【{label}】")
    print(f"    info: {d.get('info')}")
    for k in cand:
        t = w[k]
        gpt_cb[label] = t.float()
        print(f"    {k}  shape={tuple(t.shape)}  dtype={t.dtype}")
    # 也看 EOS / vocab
    cfg = d.get("config", {}).get("model", {})
    print(f"    EOS={cfg.get('EOS')}  vocab={cfg.get('phoneme_vocab_size')}")

print()
print("=" * 78)
print("2. SoVITS 侧的量化器码本")
print("=" * 78)
SOVITS = [
    ("s2Gv5dev.pth (v5 底模)",
     GSV / "GPT_SoVITS" / "pretrained_models" / "gsv-v5-pretrained"
     / "s2Gv5dev.pth"),
    ("s2G2333k.pth (v2 底模)",
     GSV / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained"
     / "s2G2333k.pth"),
    ("s2Gv2Pro.pth (v2Pro 底模)",
     GSV / "GPT_SoVITS" / "pretrained_models" / "v2Pro" / "s2Gv2Pro.pth"),
]
sov_cb = {}
for label, p in SOVITS:
    if not p.exists():
        print(f"  {label}: 不存在")
        continue
    d = load_sovits_new(str(p))
    w = d["weight"]
    cand = [k for k in w if "quantizer" in k and "codebook" in k]
    print(f"\n  【{label}】")
    print(f"    info: {d.get('info')}")
    for k in cand:
        t = w[k]
        sov_cb[label] = t.float()
        print(f"    {k}  shape={tuple(t.shape)}  dtype={t.dtype}")

print()
print("=" * 78)
print("3. 关键对比：GPT 码本 vs SoVITS 码本")
print("=" * 78)
if len(gpt_cb) >= 2 and sov_cb:
    import itertools
    gnames = list(gpt_cb)
    snames = list(sov_cb)
    print(f"  {'GPT 码本':30s} {'SoVITS 码本':28s} {'形状':>14s} {'余弦':>10s} {'最大差':>10s}")
    for g, s in itertools.product(gnames, snames):
        a, b = gpt_cb[g], sov_cb[s]
        if a.shape != b.shape:
            print(f"  {g[:28]:30s} {s[:26]:28s} {str(tuple(a.shape))+' vs '+str(tuple(b.shape)):>14s}  形状不同")
            continue
        cos = torch.nn.functional.cosine_similarity(
            a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()
        md = (a - b).abs().max().item()
        flag = "  <-- 差异大" if cos < 0.99 else ""
        print(f"  {g[:28]:30s} {s[:26]:28s} {str(tuple(a.shape)):>14s} "
              f"{cos:10.6f} {md:10.4f}{flag}")

print()
print("=" * 78)
print("4. 帧率：25Hz vs 50Hz")
print("=" * 78)
print("""
  TTS.py:687   self.configs.hz = 50            （硬编码，给 early_stop_num 用）
  TTS.py:593   hps["model"]["semantic_frame_rate"] = "25hz"   （强制 25hz！）
  models.py:1038  if self.semantic_frame_rate == "25hz":
                      quantized = F.interpolate(..., size=int(L*2))   # 2x 上采样

  所以 v5 的语义 token 是 **25Hz**，而 GPT 生成时按 **50Hz** 算 early_stop。
  日志：T2S Decoding EOS [213 -> 494] -> 281 token
      281 / 25Hz = 11.24s   <- 与实测 11.40s 吻合
      281 / 50Hz =  5.62s
""")

print("=" * 78)
print("5. 结论与决定性实验")
print("=" * 78)
print("""
  若上面「GPT 码本 vs v5 量化器」的余弦明显低于 1.0，说明**码本不匹配** ——
  那就必须换 GPT 权重（用 v5 官方配对的 s1v3.ckpt）。

  **决定性实验（同句同参数，只换 GPT）**：
     python _tools/cloud_synth_sweep.py \\
         --weights-dir SoVITS_weights_v5_full_e32 --epochs 16 \\
         --gpt GPT_SoVITS/pretrained_models/s1v3.ckpt --no-v2pro

  判据：
    · 若时长恢复到 ~1.2s -> 证实 zfh-e4.ckpt 与 v5 不兼容
    · 若仍是 11s     -> 问题在别处（采样参数 / 25Hz 处理）
""")
