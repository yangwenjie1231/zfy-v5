# -*- coding: utf-8 -*-
"""推翻 enc_p 假设，重新定位时长异常。

## 关键代码证据（models.py:1525-1544 decode_encp）
    y_lengths = torch.LongTensor([int(codes.size(2) * 2)])   # 由 codes 决定
    quantized = self.quantizer.decode(codes)
    if self.semantic_frame_rate == "25hz":
        quantized = F.interpolate(quantized, scale_factor=2, mode="nearest")
    x, m_p, logs_p, ... = self.enc_p(quantized, y_lengths, text, text_lengths, ge)
    fea = self.bridge(x)
    fea = F.interpolate(fea, scale_factor=1.875, mode="nearest")
    fea, y_mask_ = self.wns1(fea, y_lengths1, ge)

**`enc_p` 的输出长度被 y_lengths 约束**（length regulator 按 y_lengths 展开），
而 y_lengths 来自 codes。所以：
    fea 长度 ∝ codes 长度 = GPT 生成的 token 数
**`enc_p` 只影响内容（音质/音色），不影响时长。**

-> 我之前的「enc_p 缺失导致时长异常」结论**错了**。
   缺 enc_p 会导致音质差，但不会让「让我想想。」变成 11.4 秒。

## 那时长为什么是 281 token
时长完全由 **GPT 生成多少 token** 决定：
    T2S Decoding EOS [213 -> 494]  -> 281 token
    281 / 25Hz = 11.24s

GPT 在 281 步才预测出 EOS，说明它「以为」这句话需要这么长。

## 重新审视：v2Pro 基准也在同一批里
脚本把 v2Pro 排在最后（out_sweep/v2pro/）。
如果 v2Pro 时长正常 -> 问题在 **v5 侧**（SoVITS 权重或版本判定）
如果 v2Pro 也异常     -> 问题在 **GPT/参数/脚本**（与 v5 无关）

本脚本先看本地能否复现，并列出待查项。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 78)
print("1. 推翻 enc_p 假设的代码依据")
print("=" * 78)
print("""
  models.py:1532  y_lengths = LongTensor([int(codes.size(2) * 2)])
  models.py:1539  x, m_p, logs_p, ... = self.enc_p(quantized, y_lengths, ...)
  models.py:1540  fea = self.bridge(x)
  models.py:1541  fea = F.interpolate(fea, scale_factor=1.875, mode="nearest")

  `enc_p` 是 length-regulator 结构：它把 quantized 按 y_lengths 展开。
  输出长度 = y_lengths（= codes 长度 x2），**与 enc_p 权重无关**。

  所以：缺 enc_p -> 内容/音质错，但**时长不变**。
  -> 时长 11.4s 必须由 **GPT 的 token 数** 解释。
""")

print("=" * 78)
print("2. 已知事实汇总")
print("=" * 78)
print("""
  · 生成 281 token @25Hz = 11.24s（实测 11.40s 吻合）
  · 「让我想想。」5 字正常约 30 token / 1.2s
  · 进度条 /1500 -> max_sec=30（early_stop_num = 50*30 = 1500）
  · EOS 在 281 步触发，**远小于 1500** -> 不是被上限截断
  · GPT 权重 zfh-e4.ckpt（v2 底模微调），max_sec=54
  · v5 官方配对是 s1v3.ckpt

  => GPT 在 281 步就**主动**预测了 EOS，即它认为这句话该这么长。
""")

print("=" * 78)
print("3. 最可疑的三条（按可能性排序）")
print("=" * 78)
print("""
  A. **prompt_text 与 ref_audio 不匹配**
     prompt_cache 里的 prompt_semantic 由 ref_audio 编码而来。
     若 prompt_text 与 ref_audio 说的内容不一致，GPT 会收到矛盾的
     上下文 -> 可能无法正确判断"该在哪里停"。
     已知：ref/default.wav + ref.txt 是生产验证过的一对，应该匹配。
     **但云端 ref.txt 是 105 字节 —— 之前盘点脚本报过"疑似 LFS 指针"**！
     需要确认云端 ref.txt 内容是否真的是那句台词。

  B. **temperature / top_k / repetition_penalty**
     我们传了 temperature=1.0, top_k=15, repetition_penalty=1.35。
     生产 v2Pro 用同样的值，若 v2Pro 正常则排除。

  C. **v5 的 25Hz vs 50Hz 口径**
     GPT 训练时 hz=25（train_gpt.py 设了 hz=25hz）。
     推理时 early_stop_num = 50*max_sec，但实际 token 是 25Hz 的。
     这只影响**上限**，不影响 EOS 位置。

  D. **GPT 与 v5 语义 token 分布不匹配**（仍可能）
     zfh-e4 是 v2 系微调；v5 语料的 6-name2semantic.tsv 由 **s1v3** 生成。
     v2-based GPT 生成的 token 分布 ≠ v5 SoVITS 训练时见过的分布
     -> SoVITS 渲染出来可能不对，但**时长由 GPT 决定**，所以还是 GPT 的锅。
""")

print("=" * 78)
print("4. 决定性实验（成本低，信息量大）")
print("=" * 78)
print("""
  实验 1：**只换 GPT**（同权重、同参数、同句子）
    python _tools/cloud_synth_sweep.py \\
        --weights-dir SoVITS_weights_v5_full_e32 --epochs 16 --no-v2pro \\
        --gpt GPT_SoVITS/pretrained_models/s1v3.ckpt
    判据：若时长恢复正常 -> 证实 zfh-e4 与 v5 不匹配

  实验 2：**看 v2Pro 基准**（上一批已跑，但只出了 e16 —— 说明 v2Pro 那步
          可能还没跑到，或也失败了）。重跑并只跑 v2Pro：
    python _tools/cloud_synth_sweep.py --weights-dir SoVITS_weights_v5_full_e32 \\
        --epochs 999 --no-v2pro   # 空集合
    （更简单：直接看 out_sweep/v2pro/ 有没有文件）

  实验 3：**检查云端 ref.txt 内容**
    cat ref/ref.txt   # 应为「再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？」
    若内容不对 -> prompt_text 与 ref_audio 矛盾 -> 直接解释过度生成
""")
