# -*- coding: utf-8 -*-
"""诊断：为什么「让我想想。」生成了 281 个语义 token（11.4 秒）。

## 日志证据
    T2S Decoding EOS [213 -> 494]      <- 生成 494-213 = 281 个 token
    01_短_e16.wav  11.40s @48000Hz

281 token @ 25Hz = 11.24s。而「让我想想。」5 个字正常约 1.2s（~30 token）。
**9 倍超额生成** —— AR 模型没在正确位置预测出 EOS。

## 首要假设：GPT 与 SoVITS 的语义 token 约定不匹配
  · 生产用的 zfh-e4.ckpt 是**从 v2 底模**微调的（见 tmp_s1.yaml）
  · 而 v5 的语义 token 是**用 s1v3.ckpt** 生成的（config.py: v3/v4/v5* 都用 s1v3）
  · 推理时用 v2-based GPT 去喂 v5 SoVITS -> 分布不匹配 -> 过度生成

本脚本核对：
  1. 两个 GPT 的 config 差异（尤其 max_sec / 词表 / 训练参数）
  2. v5 语料的语义 token 是哪个 GPT 生成的
  3. 推理时 early_stop_num 的来源
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
V5 = Path(r"D:\A\zfy-v5")
sys.path.insert(0, str(GSV))

import torch  # noqa: E402

CAND = [
    ("生产 GPT（v2 底模微调）",
     Path(r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh-e4.ckpt")),
    ("s1v3 底模",
     GSV / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"),
    ("v2 底模",
     GSV / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained"
     / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
]

print("=" * 78)
print("1. 各 GPT 的 config 对比")
print("=" * 78)
rows = []
for label, p in CAND:
    if not p.exists():
        print(f"  {label}: 不存在 {p}")
        continue
    d = torch.load(str(p), map_location="cpu", weights_only=False)
    cfg = d.get("config", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    train = cfg.get("train", {})
    rows.append((label, p, d, data, model, train))
    print(f"\n  【{label}】")
    print(f"    文件      : {p.name}  ({p.stat().st_size/1e6:.1f} MB)")
    print(f"    info      : {d.get('info', '?')}")
    print(f"    max_sec   : {data.get('max_sec')}")
    print(f"    phoneme_vocab_size: {model.get('phoneme_vocab_size')}")
    print(f"    n_layer/n_head    : {model.get('n_layer')} / {model.get('n_head')}")
    print(f"    EOS       : {model.get('EOS')}")
    print(f"    batch/epochs      : {train.get('batch_size')} / {train.get('epochs')}")
    print(f"    weight 键数       : {len(d.get('weight', {}))}")

print()
print("=" * 78)
print("2. 关键：max_sec 决定 early_stop_num")
print("=" * 78)
print("  TTS.py:1400  early_stop_num = self.configs.hz * self.configs.max_sec")
print("  self.configs.hz = 50（硬编码 TTS.py:687）")
print("  self.configs.max_sec 来自 GPT 的 config['data']['max_sec']")
for label, p, d, data, model, train in rows:
    ms = data.get("max_sec")
    print(f"    {label:26s} max_sec={ms}  -> early_stop={50*ms if ms else '?'}")
print()
print("  日志里进度条是 /1500 -> max_sec=30")
print("  但 EOS 在 281 token 就触发了（远小于 1500）-> **不是被上限截断**")
print("  => 是 AR 模型自己没预测出 EOS，即**过度生成**")

print()
print("=" * 78)
print("3. v5 语料的语义 token 是谁生成的")
print("=" * 78)
# config.py 里 pretrained_gpt_name 的版本映射
cfgpy = GSV / "GPT_SoVITS" / "configs" / "s1.yaml"
if cfgpy.exists():
    txt = cfgpy.read_text(encoding="utf-8", errors="replace")
    for line in txt.split("\n"):
        if "pretrained" in line.lower() and "gpt" in line.lower():
            print(f"    s1.yaml: {line.strip()}")
# 根 config.py
rootcfg = GSV / "config.py"
if rootcfg.exists():
    txt = rootcfg.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(txt.split("\n"), 1):
        if "s1v3" in line or "pretrained_gpt" in line or "s1bert" in line:
            print(f"    config.py:{i}  {line.strip()[:100]}")

print()
print("=" * 78)
print("4. 结论")
print("=" * 78)
print("""
  若两个 GPT 的 max_sec / 词表一致，则不是配置问题，而是**权重分布**问题：
  v2-based GPT 生成的语义 token 分布，与 v5 SoVITS 训练时见过的
  （s1v3 生成的）不一致 -> AR 找不到 EOS 的合适位置 -> 过度生成。

  **决定性实验**：同句同参数，只换 GPT：
      --gpt ref/zfh-e4.ckpt      （v2-based，现在用的）
      --gpt GPT_SoVITS/pretrained_models/s1v3.ckpt   （v5 的原配）
  若后者时长正常 -> 证实不匹配。
""")
