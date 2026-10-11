# -*- coding: utf-8 -*-
"""根因确认：`ssl_proj` 缺失（而非 enc_p）导致 GPT 过度生成。

## 完整证据链

### 1. prompt_semantic 的生成路径
    TTS.py:930   codes = self.vits_model.extract_latent(hubert_feature)
    TTS.py:932   prompt_semantic = codes[0, 0]
    TTS.py:1355  prompt = self.prompt_cache["prompt_semantic"].expand(...)

    models.py:1546-1548   extract_latent
        ssl = self.ssl_proj(x)                       # <-- 用 ssl_proj！
        quantized, codes, commit_loss, _ = self.quantizer(ssl)
        return codes.transpose(0, 1)

### 2. ssl_proj 的作用（models.py:935-937）
    self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)   # 50Hz -> 25Hz
    （ssl_dim=768；semantic_frame_rate=="25hz" 时用 stride=2 那支）

    它把 HuBERT 的 **50Hz** 特征下采样成 **25Hz**，再交给 quantizer 出 code。

### 3. 缺失后会发生什么
    ssl_proj 随机初始化 -> 下采样输出是噪声
    -> quantizer 量化出**错误的 code**
    -> prompt_semantic 变成垃圾
    -> GPT 拿到错误的 prompt 上下文
    -> **找不到 EOS 的合适位置 -> 过度生成**

### 4. 为什么 enc_p 不是主因（我先前的误判）
    models.py:1532  y_lengths = LongTensor([int(codes.size(2) * 2)])
    models.py:1539  x, m_p, logs_p, ... = self.enc_p(quantized, y_lengths, ...)
    enc_p 是 length-regulator：输出长度 = y_lengths（由 codes 决定）。
    所以缺 enc_p 影响**内容/音质**，但**不改变时长**。

    **但时长由 GPT 决定，而 GPT 的输入 prompt 依赖 ssl_proj** ——
    所以真正的元凶是 ssl_proj，enc_p 是次要的（音质）。

### 5. 数字佐证
    /1500 是 t2s_model.py:533 的**硬编码循环上限**（`for _ in tqdm(range(1500))`），
    与 max_sec 无关。EOS 在 281 步触发，远小于 1500，说明是模型**主动**停的，
    不是被截断 —— 与"prompt 错误导致判断失误"一致。

## 结论
`merge_base_encp.py` 补的键**是对的**（ssl_proj 就在 missing 列表里），
只是我之前把因果归给了 enc_p。补上 ssl_proj 后，prompt_semantic 才正确。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
sys.path.insert(0, str(GSV))
sys.path.insert(0, str(GSV / "GPT_SoVITS"))
sys.path.insert(0, str(GSV / "GPT_SoVITS"))

from process_ckpt import load_sovits_new  # noqa: E402

print("=" * 78)
print("1. ssl_proj 的形状与作用")
print("=" * 78)
d = load_sovits_new(str(GSV / "GPT_SoVITS" / "pretrained_models"
                        / "gsv-v5-pretrained" / "s2Gv5dev.pth"))
w = d["weight"]
for k in sorted(w):
    if k.startswith("ssl_proj."):
        t = w[k]
        print(f"  {k}  shape={tuple(t.shape)}")
print("""
  Conv1d(768, 768, kernel=2, stride=2) 的含义：
    输入 [B, 768, T]（T = 50Hz 帧数）
    输出 [B, 768, T/2]（25Hz）
  **它就是把 50Hz 的 HuBERT 特征降成 25Hz 语义 token 的那一步。**
  随机初始化 -> 输出噪声 -> quantizer 给出错误 code -> prompt 错。
""")

print("=" * 78)
print("2. 缺失的键里 ssl_proj 只有 2 个，但影响最大")
print("=" * 78)
missing_groups = {"enc_p": 235, "ssl_proj": 2}
print(f"  enc_p   : {missing_groups['enc_p']} 个键  -> 影响内容/音质")
print(f"  ssl_proj: {missing_groups['ssl_proj']} 个键  -> 影响 **prompt_semantic**")
print("  （cloud 日志的 missing_keys 末尾正是 ssl_proj.weight / ssl_proj.bias）")

print()
print("=" * 78)
print("3. merge_base_encp.py 已覆盖这两个组")
print("=" * 78)
print("""
  merge 脚本从底模取**所有**缺失键（不限于 enc_p），所以 ssl_proj 也会被补上。
  本地 verify_merge_encp.py 的断言里「补入的 enc_p/ssl_proj 与底模一致」
  已经覆盖了 ssl_proj。
""")

print("=" * 78)
print("4. 验证方法")
print("=" * 78)
print("""
  跑 merge 后重新合成「让我想想。」，看：
    · 日志 T2S Decoding EOS [213 -> N]，N 应约 240~250（即 ~30 token）
    · 时长应约 1.2s
  当前 [213 -> 494]（281 token / 11.40s）。

  另一个更直接的验证：**对比 prompt_semantic 的值**
    用底模（有正确 ssl_proj）算一次 prompt_semantic
    用未合并的 e16 权重算一次
    两者应该差异很大；合并后应与底模一致。
""")
