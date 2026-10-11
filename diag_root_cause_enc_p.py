# -*- coding: utf-8 -*-
"""修复：全量微调权重缺 enc_p/ssl_proj（导致合成时长 9 倍异常）。

## 根因链（已完整确认）

1. `configs/s2.json` 里有 `"freeze_quantizer": true`
   （v5 的默认配置，因为官方只做 LoRA —— LoRA 时这些层本来就不训）

2. `models.py:1325-1328`（SynthesizerTrnV3.__init__）：
       if self.freeze_quantizer == True:
           set_no_grad(self.ssl_proj)
           set_no_grad(self.quantizer)
           set_no_grad(self.enc_p)      # <- requires_grad = False

3. `s2_train_v5_full.py:402-407` 导出权重时：
       no_grad_names = {所有 requires_grad=False 的参数名}
       for key in ckpt:
           if key not in no_grad_names:     # <- 跳过它们
               sim_ckpt[key] = ckpt[key].half().cpu()

   -> enc_p / ssl_proj / quantizer **不进权重文件**

4. 推理时 `load_state_dict(..., strict=False)` **静默保留随机初始化**
   -> `decode_encp()` 的先验分布 m_p/logs_p 全错
   -> 「让我想想。」生成 281 个 token（正常 ~30）-> 11.40s（正常 ~1.2s）

## 算术佐证
    完整模型            383.08M 参数
    enc_p + ssl_proj     15.33M
    云端权重 735.7MB/2  367.85M
    缺口                 15.23M   <- 差 0.09M（config 等开销）

## 为什么 v5 推理**必须**要 enc_p
    TTS.py:1752  fea_ref, ge = self.vits_model.decode_encp(...)
    TTS.py:1767  fea_todo, _ = self.vits_model.decode_encp(...)
    TTS.py:1768  mel = synthesize_v5_mel(...)
`decode_encp` -> `models.py:1361/1525` -> 用 `self.enc_p`。

## 本脚本做什么
  1. 确认 `set_no_grad` 与 `no_grad_names` 的行为
  2. 确认 s2.json 的 freeze_quantizer 值
  3. 输出**修复方案**（两条路）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")

print("=" * 78)
print("1. set_no_grad 的定义")
print("=" * 78)
mp = GSV / "GPT_SoVITS" / "module" / "models.py"
txt = mp.read_text(encoding="utf-8", errors="replace")
lines = txt.split("\n")
for i, line in enumerate(lines, 1):
    if "def set_no_grad" in line:
        for j in range(i - 1, min(i + 8, len(lines))):
            print(f"  models.py:{j+1}  {lines[j]}")
        break

print()
print("=" * 78)
print("2. s2.json 的 freeze_quantizer")
print("=" * 78)
for name in ("s2.json", "s2v2Pro.json", "s2v5.json"):
    p = GSV / "GPT_SoVITS" / "configs" / name
    if not p.exists():
        continue
    cfg = json.loads(p.read_text(encoding="utf-8"))
    fq = cfg.get("model", {}).get("freeze_quantizer")
    ver = cfg.get("model", {}).get("version")
    print(f"  {name:16s} version={ver!r:10s} freeze_quantizer={fq}")

print()
print("=" * 78)
print("3. 我们训练实际用的配置")
print("=" * 78)
tp = GSV / "tmp_full_e32.json"
if tp.exists():
    cfg = json.loads(tp.read_text(encoding="utf-8"))
    m = cfg.get("model", {})
    print(f"  tmp_full_e32.json: version={m.get('version')!r} "
          f"freeze_quantizer={m.get('freeze_quantizer')}")
else:
    print("  （本地无 tmp_full_e32.json）")

print()
print("=" * 78)
print("4. 修复方案")
print("=" * 78)
print("""
### 方案 A（推荐）：训练时把 freeze_quantizer 设为 False
  这样 enc_p/ssl_proj/quantizer 会**参与训练**并**进入权重文件**。
  代价：多 15.33M 参数要训（383M -> 383M，本来就在模型里，只是解冻）
  好处：语义/韵律信息能适配新语料 —— 这本来就是"全量微调"该做的

  改法：训练配置里 `"freeze_quantizer": false`
        （models.py:1325 只在 == True 时 set_no_grad）

### 方案 B：保留 freeze，但导出时**不裁剪**
  改 s2_train_v5_full.py:402-407，把 no_grad_names 的过滤去掉：
      for key in ckpt:
          sim_ckpt[key] = ckpt[key].half().cpu()
  这样权重含全部 681 键（含未训练的 enc_p 原值）。
  **注意**：freeze 时 enc_p 不更新，所以它的值就是底模的原值 ——
  对全量微调来说这是"没训到"，但至少不是随机初始化。

### 方案 C（应急）：用底模的 enc_p 补上
  从 s2Gv5dev.pth 取 enc_p/ssl_proj/quantizer 的权重，
  合并进我们训练出的权重文件。

### 我的建议
  如果目标是"全量微调、充分利用 63 分钟语料"，用 **A**：
  语义编码器也解冻，让它适配庄方宜的语料。
  如果只想快速验证"时长是否恢复正常"，用 **C**（不用重训）。

### 验证方法（修完必做）
  重新合成「让我想想。」，看：
    · 日志 T2S Decoding EOS [213 -> N]，N 应约 240~250（即 ~30 token）
    · 输出时长应约 1.2s
  当前是 [213 -> 494]（281 token，11.40s）。
""")

print("=" * 78)
print("5. 附：为什么之前没发现")
print("=" * 78)
print("""
  · `load_state_dict(strict=False)` 对 missing_keys **只打印不报错**
  · 云端日志里那一大串 missing_keys 被淹在 5000 字里，很容易当成噪声
  · 而 `unexpected_keys=[]` 让人误以为"键都对上了"
  · 实际是**反向**问题：我们的权重**少了**推理必需的层

  这是一个**无声错误**：能跑、能出音，但结果是错的。
  与之前"数据坏但训练成功"是同一类陷阱。
""")
