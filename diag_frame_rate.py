# -*- coding: utf-8 -*-
"""诊断语义 token 帧率：GPT 训练时的速率 vs 推理时的速率。

## 症状
    T2S Decoding EOS [213 -> 494]   -> 生成 281 个 token
    01_短_e16.wav  11.40s @48000Hz
「让我想想。」5 个字正常约 1.2s。

## 为什么查帧率
  281 token：
    @25Hz = 11.24s   <- 与实测 11.40s 吻合
    @50Hz =  5.62s
  若 GPT 是按 **50Hz** 训练的（v2 老管线），而 v5 SoVITS 按 **25Hz** 解释，
  就会产生 2 倍的时长误差。

## 关键事实（已知）
  · TTS.py:687   self.configs.hz = 50                 （硬编码，算 early_stop）
  · TTS.py:593   semantic_frame_rate = "25hz"          （强制 25Hz）
  · models.py:1038  25hz 时 quantized 做 2x 上采样
  · zfh-e4.ckpt 是从 **v2 底模**微调的（tmp_s1.yaml）
  · v5 语料的 6-name2semantic.tsv 用 hz=25hz 生成

本脚本核对：
  1. 两个 GPT 的 config 里与帧率相关的字段
  2. prepare_datasets 生成语义 token 时用的 hz
  3. 由 token 数反推"模型认为的时长"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
sys.path.insert(0, str(GSV))

import torch  # noqa: E402

print("=" * 78)
print("1. GPT config 的全部字段（找帧率线索）")
print("=" * 78)
GPTS = [
    ("zfh-e4.ckpt (v2 底模微调)",
     Path(r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh-e4.ckpt")),
    ("s1v3.ckpt (v5 官方配对)",
     GSV / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"),
]
for label, p in GPTS:
    if not p.exists():
        print(f"  {label}: 不存在")
        continue
    d = torch.load(str(p), map_location="cpu", weights_only=False)
    cfg = d.get("config", {})
    print(f"\n  【{label}】 info={d.get('info')}")
    for section in ("data", "model", "train"):
        sub = cfg.get(section, {})
        if not isinstance(sub, dict):
            continue
        hits = {k: v for k, v in sub.items()
                if any(t in k.lower() for t in
                       ("hz", "frame", "rate", "sec", "eos", "vocab", "token"))}
        print(f"    [{section}] {hits}")

print()
print("=" * 78)
print("2. 语义 token 生成脚本用的帧率")
print("=" * 78)
for f in ("1-get-text.py", "2-get-semantic.py", "3-get-semantic.py"):
    p = GSV / "GPT_SoVITS" / "prepare_datasets" / f
    if not p.exists():
        continue
    txt = p.read_text(encoding="utf-8", errors="replace")
    print(f"\n  【{f}】")
    for i, line in enumerate(txt.split("\n"), 1):
        if "hz" in line.lower() and ("environ" in line or "=" in line):
            s = line.strip()
            if "hz" in s.lower() and len(s) < 130:
                print(f"    L{i}: {s}")

print()
print("=" * 78)
print("3. 反推：281 token 在各帧率下的时长")
print("=" * 78)
N = 281
for hz in (25, 50, 100):
    print(f"  {N} token @ {hz}Hz = {N/hz:.2f}s")
print()
print("  实测 11.40s -> 与 @25Hz 的 11.24s 吻合")
print("  「让我想想。」5 字正常 ~1.2s -> 正常 token 数约：")
for hz in (25, 50):
    print(f"    @{hz}Hz: {1.2*hz:.0f} token")

print()
print("=" * 78)
print("4. 关键：v5 语料的语义 token 是 25Hz 还是 50Hz")
print("=" * 78)
tsv = GSV / "logs" / "zfy_v5" / "6-name2semantic.tsv"
if tsv.exists():
    lines = tsv.read_text(encoding="utf-8", errors="replace").strip().split("\n")
    print(f"  行数: {len(lines)}")
    print(f"  表头: {lines[0][:100]}")
    # 第 2 行起是 path\ttoken序列
    import statistics
    counts = []
    for line in lines[1:6]:
        parts = line.split("\t")
        if len(parts) >= 2:
            toks = parts[1].split()
            counts.append(len(toks))
            print(f"    {parts[0][:50]:52s} {len(toks)} token")
    if counts:
        print(f"\n  样例平均 {statistics.mean(counts):.0f} token")
        print("  （若音频平均 ~4.6s，则 25Hz -> ~115 token，50Hz -> ~230 token）")

print()
print("=" * 78)
print("5. 结论")
print("=" * 78)
print("""
  **决定性实验（最有价值，成本最低）**：
  看同一批合成里 v2Pro 基准的时长。脚本已把它排在最后一个任务：
      out_sweep/v2pro/01_短_v2pro.wav
  · 若 v2Pro 也是 ~11s  -> 是**脚本/参数**问题（与 v5 无关）
  · 若 v2Pro 是 ~1.2s   -> 是 **v5 专属**问题（帧率/权重配对）

  第二个实验（只换 GPT）：
      python _tools/cloud_synth_sweep.py \\
          --weights-dir SoVITS_weights_v5_full_e32 --epochs 16 \\
          --gpt GPT_SoVITS/pretrained_models/s1v3.ckpt --no-v2pro
""")
