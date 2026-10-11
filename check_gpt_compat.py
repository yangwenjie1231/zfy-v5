# -*- coding: utf-8 -*-
"""核对 GPT 权重的语义 token 空间，确认 zfh-e4 能否直接配 v5 SoVITS。

为什么这一步是当前最关键的：
    用户反馈「v5 效果很差，特别是语气处理」。
    语气/韵律由 **GPT 生成的语义 token 序列**决定，而 SoVITS 只负责
    把 token 还原成波形（音色）。我训练/评测 v5 时用的是**未微调的
    `s1v3.ckpt`**，而生产 v2Pro 用的是**微调过的 `zfh-e4.ckpt`**
    —— 这可以解释"语气差"。

    但换 GPT 的前提是**码本/词表必须兼容**，否则语义 token 对不上，
    输出会是噪声。本脚本逐项比对。
"""
from __future__ import annotations

import sys

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CKPTS = [
    ("zfh-e4 (微调)", r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh-e4.ckpt"),
    ("s1v3 (底模)", r"D:\A\zfy-v5\GPT-SoVITS\GPT_SoVITS\pretrained_models\s1v3.ckpt"),
]

info = {}
for name, p in CKPTS:
    d = torch.load(p, map_location="cpu", weights_only=False)
    w = d.get("weight", d)
    keys = list(w.keys())
    # 关键张量
    def find(*pats):
        for k in keys:
            if all(pt in k for pt in pats):
                return k
        return None

    cand = {
        "audio_embedding": find("ar_audio_embedding"),
        "text_embedding": find("ar_text_embedding", "word_embedding"),
        "text_position": find("ar_text_embedding", "position_embedding"),
        "codebook(quantizer)": find("vq_model", "codebook"),
    }
    shapes = {}
    for k, key in cand.items():
        shapes[k] = tuple(w[key].shape) if key else None

    # 量化器码本（决定语义 token 空间）
    vq = [k for k in keys if "vq_model" in k or "quantizer" in k]
    info[name] = {"n_keys": len(keys), "shapes": shapes, "vq_keys": vq[:6],
                  "config": d.get("config")}

    print(f"=== {name} ===")
    print(f"  路径: {p}")
    print(f"  键数: {len(keys)}")
    for k, v in shapes.items():
        print(f"  {k:24s} {v}")
    if vq:
        print(f"  量化器相关键: {vq[:6]}")
    cfg = d.get("config")
    if cfg is not None:
        m = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        for k in ("vocab_size", "phoneme_vocab_size", "hidden_dim", "n_layer",
                  "EOS", "semantic_frame_rate"):
            if k in m:
                print(f"  config.model.{k} = {m[k]}")
    print()

# 逐键比对
a, b = list(info.values())
print("=" * 70)
print("逐项比对")
print("=" * 70)
same = True
for k in a["shapes"]:
    va, vb = a["shapes"][k], b["shapes"][k]
    ok = va == vb
    same &= ok
    print(f"  {k:24s} {str(va):26s} vs {str(vb):26s} {'一致' if ok else '**不同**'}")

print()
ca, cb = a["config"], b["config"]
if isinstance(ca, dict) and isinstance(cb, dict):
    ma, mb = ca.get("model", {}), cb.get("model", {})
    diffs = {k: (ma.get(k), mb.get(k)) for k in set(ma) | set(mb)
             if ma.get(k) != mb.get(k)}
    if diffs:
        print("config.model 差异:")
        for k, (x, y) in sorted(diffs.items()):
            print(f"  {k}: {x}  vs  {y}")
        same = False
    else:
        print("config.model 完全一致")

print()
print("=" * 70)
print("结论:", "结构兼容，zfh-e4 可直接配 v5 SoVITS"
      if same else "存在差异，需进一步确认（尤其码本/词表）")
print("=" * 70)
