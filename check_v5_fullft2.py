# -*- coding: utf-8 -*-
"""确认 v5 能否做全量微调，以及需要改什么。

已核实的事实
------------
  · `s2_train.py`          import `SynthesizerTrn`      —— **不含 v5 支持**
                            （全文无 models_v5 / SynthesizerTrnV3）
  · `s2_train_v3_lora.py`  import `SynthesizerTrnV3`    —— **v5 就在这个类里**
                            （models.py L1306-1308：version in {v5dev,v5turbo}
                              -> self.cfm = CFMV5(...)）
                            但它硬编码用 peft 包 LoRA：
                              net_g.cfm = get_peft_model(net_g.cfm, lora_config)

结论：v5 的模型结构**支持全量**（SynthesizerTrnV3 就是完整模型），
      只是官方只给了 LoRA 训练脚本。要做全量微调需要：
        · 用 s2_train_v3_lora.py 的框架（它认识 v5）
        · 但**不**给 cfm 套 get_peft_model
        · 优化器吃全部参数（而非只吃 LoRA 参数）

本脚本验证 SynthesizerTrnV3 能正确构造 v5 模型，并报告参数量，
用来估算全量微调的显存需求。
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
for p in reversed([str(GSV), str(GSV / "GPT_SoVITS"),
                   str(GSV / "GPT_SoVITS" / "BigVGAN"),
                   str(GSV / "tools"), str(GSV / "tools" / "asr"),
                   str(GSV / "tools" / "uvr5")]):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
os.chdir(str(GSV))

import torch  # noqa: E402

print("=" * 82)
print("1) 两个训练脚本的模型类")
print("=" * 82)
for name in ("s2_train.py", "s2_train_v3_lora.py"):
    f = GSV / "GPT_SoVITS" / name
    src = f.read_text(encoding="utf-8", errors="replace")
    cls = ("SynthesizerTrnV3" if "SynthesizerTrnV3 as SynthesizerTrn" in src
           else "SynthesizerTrn")
    print(f"  {name:24s} 用 {cls:20s} "
          f"peft={str('peft' in src):5s} "
          f"全量参数优化器={str('model.parameters()' in src)}")

print()
print("=" * 82)
print("2) 构造 v5 的 SynthesizerTrnV3")
print("=" * 82)
from module.models import SynthesizerTrnV3  # noqa: E402
from module.models_v5 import CFMV5  # noqa: E402

cfg = json.loads((GSV / "GPT_SoVITS" / "configs" / "s2.json")
                 .read_text(encoding="utf-8"))
sig = inspect.signature(SynthesizerTrnV3.__init__)
print(f"  SynthesizerTrnV3.__init__ 参数: {list(sig.parameters.keys())[:14]} ...")

kwargs = dict(cfg["model"])
kwargs["version"] = "v5dev"
n_spec = cfg["data"]["filter_length"] // 2 + 1
n_seg = cfg["train"]["segment_size"] // cfg["data"]["hop_length"]

try:
    net = SynthesizerTrnV3(n_spec, n_seg, n_speakers=cfg["data"]["n_speakers"],
                           **kwargs)
    n_all = sum(p.numel() for p in net.parameters())
    n_cfm = sum(p.numel() for p in net.cfm.parameters())
    is_v5 = isinstance(net.cfm, CFMV5)
    print(f"  构造成功：总参数 {n_all/1e6:.2f} M，cfm {n_cfm/1e6:.2f} M，"
          f"CFMV5={is_v5}")

    # LoRA 会新增多少可训练参数（对照）
    try:
        from peft import LoraConfig, get_peft_model
        lc = LoraConfig(target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                        r=128, lora_alpha=128, init_lora_weights=True)
        cfm_l = get_peft_model(net.cfm, lc)
        n_lora = sum(p.numel() for p in cfm_l.parameters() if p.requires_grad)
        print(f"  LoRA(r=128) 可训练参数: {n_lora/1e6:.2f} M "
              f"({n_lora/n_all*100:.2f}% of total)")
    except Exception as e:  # noqa: BLE001
        print(f"  LoRA 对照失败: {type(e).__name__}")
    del net
except Exception as e:  # noqa: BLE001
    print(f"  构造失败: {type(e).__name__}: {e}")

print()
print("=" * 82)
print("结论")
print("=" * 82)
print("""
  v5 的**模型结构支持全量微调**（SynthesizerTrnV3 里就有 CFMV5），
  但官方只提供 LoRA 训练脚本（s2_train_v3_lora.py 里
  `net_g.cfm = get_peft_model(net_g.cfm, lora_config)` 是硬编码的）。

  而 s2_train.py（v2Pro 用的全量脚本）**不认识 v5**
  —— 它 import 的是 SynthesizerTrn，全文没有 models_v5。

  所以：
    · 已做的：v5 LoRA r128 / 24ep（把官方路径用满）
    · 想进一步：需要**改训练脚本**，用 SynthesizerTrnV3 + 全量参数优化器
      （去掉 get_peft_model）。这是可行的，但属于改动上游代码。
""")
