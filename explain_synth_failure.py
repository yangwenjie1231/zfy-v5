# -*- coding: utf-8 -*-
"""完整解释云端合成失败的三层原因，并给出验证。

## 日志里的三个错误，其实只有一个根因

  1. `HFValidationError: Repo id must be ... models--nvidia--bigvgan...`
  2. `TorchCodec is required for load_with_torchcodec`
  3. `resolve_sampling 不可用（NameError）`   <- 已单独修

### 错误 1 的根因链（这是主因）

我们的权重是**全量微调**产物，`process_ckpt.savee()` 的逻辑是：

    if lora_rank:                        my_save2(...)   # 写版本头
    elif "Pro" in model_version:         my_save2(...)   # 写版本头
    else:                                my_save(...)    # 不写头 -> 纯 zip "PK"

我们 lora_rank=0 且 model_version="v5dev"（不含 "Pro"）-> **无版本头**。

于是 `get_sovits_version_from_path_fast` 走到**按文件大小猜**：

    if   size < 82978*1024:  v1
    elif size < 700*1024*1024: v2
    else:  version="v2", model_version="v3"      # <- 736MB 落这里！

-> model_version="v3" -> init_vocoder("v3") -> BigVGAN(24kHz)
-> 那个 HF 目录本地不存在 -> HFValidationError

### 但补头 b"07"（v5dev）**不够**，还有第二层坑

    b"07" -> ["v2", "v5dev", if_lora_v3=**True**]
                                              ^^^^
    if_lora_v3 == True 会走 LoRA 路径：
        lora_rank = dict_s2["lora_rank"]     # 全量权重没这个键 -> KeyError
        即使补个假值：get_peft_model 包装后键名变成
        cfm.base_model.model.*，我们的 cfm.estimator.* **匹配不上**，
        strict=False 会**静默忽略** -> 最终加载的是**底模**而非微调结果。
        **这是无声的错误，比崩溃更危险。**

### 所以完整修法 = 两步

  A. 权重补头 b"07"（让 model_version 正确 = v5dev -> 48kHz Generator）
  B. 给 TTS.py 打补丁：**没有 lora_rank 键就不是 LoRA 权重**，跳过 LoRA 路径

本脚本只做**静态分析验证**，实际改动由 patch_v5_fullft_infer.py 完成。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TTS = Path(r"D:\A\zfy-v5\GPT-SoVITS\GPT_SoVITS\TTS_infer_pack\TTS.py")
src = TTS.read_text(encoding="utf-8", errors="replace")

print("=" * 78)
print("1. savee 的三条分支（决定写不写版本头）")
print("=" * 78)
print("""
    if lora_rank:                             -> my_save2 (写头)
    elif model_version != None and "Pro" in model_version:
                                              -> my_save2 (写头)
    else:                                     -> my_save  (不写头, 纯 PK)
    我们的情况: lora_rank=0, model_version="v5dev" -> **第三条** -> 无头
""")

print("=" * 78)
print("2. 按大小猜的分支（736MB 落哪）")
print("=" * 78)
size_mb = 736
if size_mb * 1024 * 1024 < 82978 * 1024:
    r = "v1"
elif size_mb * 1024 * 1024 < 700 * 1024 * 1024:
    r = "v2"
else:
    r = "version=v2, model_version=v3  <-- 我们在这里"
print(f"  736 MB -> {r}")

print()
print("=" * 78)
print("3. init_vits_weights 的关键行（定位补丁插入点）")
print("=" * 78)
lines = src.split("\n")
for i, l in enumerate(lines, 1):
    if any(k in l for k in ("def init_vits_weights",
                            "get_sovits_version_from_path_fast(weights_path)",
                            "path_sovits = self.configs.default_configs",
                            "if if_lora_v3 == True and os.path.exists",
                            "dict_s2 = load_sovits_new(weights_path)",
                            "if if_lora_v3 == False:",
                            'lora_rank = dict_s2["lora_rank"]',
                            "get_peft_model(vits_model.cfm")):
        print(f"  L{i:5d}: {l.strip()[:88]}")

print()
print("=" * 78)
print("4. torchaudio.load 的调用点（错误 2）")
print("=" * 78)
for i, l in enumerate(lines, 1):
    if "torchaudio.load" in l:
        print(f"  L{i:5d}: {l.strip()[:88]}")
print("""
  云上 torchaudio 新版把 load 委派给 torchcodec（未安装）。
  修法：优先用 soundfile 后端，再退回直接读 soundfile。
""")

print("=" * 78)
print("5. 结论：需要两处补丁")
print("=" * 78)
print("""
  补丁 1（权重侧，离线做一次）
      fix_weight_header.py --dir SoVITS_weights_v5_full_e32 --apply
      -> 前 2 字节 PK -> b"07"，model_version 变 v5dev

  补丁 2（推理侧，改 TTS.py）
      2a. if_lora_v3 且 dict_s2 无 "lora_rank" -> 强制 if_lora_v3 = False
      2b. torchaudio.load -> 带回退的包装函数

  两者缺一不可：
      只做 1 -> KeyError: 'lora_rank'
      只做 2 -> 仍按大小猜成 v3 -> BigVGAN 报错
""")
