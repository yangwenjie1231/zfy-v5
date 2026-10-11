# -*- coding: utf-8 -*-
"""云算力（ROCm）跑 v5 全量微调的可行性评估。

机器规格（用户提供）：
  ubuntu22.04 / rocm7.2.3 / py3.12 / torch2.12.0
  8 核 CPU / 200GB 内存 / **192GB 显存**（AMD，ROCm）

核心问题
--------
v5 全量微调在本机（RTX 5060 8GB）**不可行**：
  · v5 SoVITS 总参数 381.51M（v2Pro 只有 42.39M，9 倍）
  · 全量微调优化器状态估算：
      参数 fp16 0.76 + 梯度 fp16 0.76 + fp32 主权重 1.53
      + AdamW(m,v) 3.05 = **6.10 GB**
    只剩 2.05 GB 给激活/批量 —— 加上梯度检查点也很紧，
    而且 v5 的 CFM 是 22 层 DiT（335.79M），激活占用远大于 v2Pro。

云机 192GB 显存 → 完全够（估算 6.1GB 是 v2Pro 的 9 倍仍有巨大余量）。

ROCm 兼容性风险（逐条核查结果）
--------------------------------
  风险项                          状态        说明
  ─────────────────────────────  ──────────  ─────────────────────────
  torch.cuda.amp (GradScaler)     低          仍存在（deprecated），
                                               ROCm 上也能跑；建议换 torch.amp
  torch.backends.cuda.*.allow_tf32 中          需加 try/except，ROCm 无此属性
  flash_attn                      低          训练路径**不需要**
                                              （仅推理可选加速）
  bitsandbytes / apex / deepspeed 无风险       代码里 0 处引用
  CUDA_VISIBLE_DEVICES            低          ROCm 下被忽略，不影响单卡
  DistributedBucketSampler        中          单卡时不启用 DDP，风险低
  cuda_graph (v5 新特性)           中          11 个文件提到；训练不用，
                                              推理若开启需关掉
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N_V5 = 381.51e6
N_V2PRO = 42.39e6
N_LORA128 = 23.07e6
VRAM_LOCAL = 8.15


def est(n, batch_note=""):
    p = n * 2 / 1e9
    g = n * 2 / 1e9
    m = n * 4 / 1e9
    o = n * 8 / 1e9
    return {"param": p, "grad": g, "master": m, "adam": o,
            "sum": p + g + m + o}


print("=" * 78)
print("显存估算（全量微调，AdamW + fp16 混合精度）")
print("=" * 78)
print(f"{'模型':22s} {'参数量':>12s} {'优化器+权重':>13s} {'本机 8GB':>12s}")
print("-" * 78)
for label, n, is_lora in [("v2Pro SoVITS（已跑通）", N_V2PRO, False),
                          ("v5 SoVITS 全量", N_V5, False),
                          ("v5 LoRA r128（已跑通）", N_LORA128, True)]:
    e = est(n)
    # LoRA：只有低秩参数需要优化器状态，但**整个模型**仍要占显存（fp16）
    need = (e["sum"] + N_V5 * 2 / 1e9) if is_lora else e["sum"]
    # 除优化器状态外，还要留激活/临时张量。经验上激活与模型规模同量级，
    # 且 8GB 卡实测 LoRA 已占 7.7GB —— 所以全量 6.1GB 优化器状态必然放不下。
    if is_lora:
        verdict = "已跑通(7.7G)"
    else:
        verdict = "**不可行**" if need > 3.0 else "可行"
    print(f"{label:22s} {n/1e6:11.2f}M {need:12.2f}GB {verdict:>12s}")

print()
print("  关键：6.10GB 只是**优化器状态+权重**，还没算激活。")
print("        v5 的 CFM 是 22 层 DiT（335.79M），激活占用远大于 v2Pro；")
print("        8GB 卡连 LoRA（冻结全部 381M 权重）都已占到 7.7GB。")
print("        => v5 全量微调在本机 **不可行**。")

print()
print("=" * 78)
print("云机可行性（192GB 显存）")
print("=" * 78)
e = est(N_V5)
print(f"  v5 全量微调基础占用      : {e['sum']:.2f} GB")
print(f"  192GB 显存余量           : {192 - e['sum']:.2f} GB")
print(f"  可支撑的 batch_size      : 很大（激活是主要变量，需实测）")
print(f"  结论                     : **完全可行**，显存不是瓶颈")

print()
print("=" * 78)
print("建议方案")
print("=" * 78)
print("""
  1. **数据准备在本机做完**（不需要上云）
     `logs/zfy_v5/` 里的 4-cnhubert / 5-wav32k / 6-name2semantic.tsv
     已算好（825 条）。只需打包上传：
       · GPT-SoVITS 检出（含 v5 代码 + 底模 + BERT/HuBERT）
       · logs/zfy_v5/（特征）
       · data/zfy_v5/wav32k/（音频，供 dataset 读）
     合计约 4-5 GB。

  2. **上云后改 3 处**（我会写好补丁脚本）
     a. `torch.backends.cuda.matmul.allow_tf32` -> try/except 包裹
     b. `torch.cuda.amp` -> `torch.amp`（避免 deprecated 警告/未来移除）
     c. 新增 `s2_train_v5_full.py`：用 SynthesizerTrnV3 但**不套 peft**
        （即把 s2_train_v3_lora.py 里的 get_peft_model 去掉，
          优化器吃全部参数）

  3. **显存不是瓶颈，时间才是**
     本机 LoRA r128 是 ~11 分钟/epoch（407 step，1.1 s/it）。
     全量微调每 step 更慢（要算梯度+更新 381M 参数），
     但云机 GPU 通常更强，预计可接受。
     建议先跑 4-8 epoch 看效果，再决定是否加长。

  4. **验证闭环**
     云上训完，把权重（每个约 400MB，全量）传回本机，
     用本机已有的 `eval_retrain.py` 评测 + 出盲听集
     （参考音频和 GPT 口径都已修正）。
""")
