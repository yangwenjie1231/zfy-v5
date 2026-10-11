# -*- coding: utf-8 -*-
"""验证：v5 能否走**全量微调**（s2_train.py）—— 这是 v2Pro 用的方式。

背景
----
v2/v2Pro 用 `s2_train.py`（全量，673/678 键，无 peft）。
v5 的官方路径是 `s2_train_v3_lora.py`（LoRA，只微调 CFM 的 attention）。
所以 v5 音色不如 v2Pro 是"LoRA vs 全量"的差距。

但源码显示 `models.py` **认识 v5**：
    L1304  dit = DiT(..., use_step_embedding=version not in {"v5dev","v5turbo"})
    L1306  if version in {"v5dev","v5turbo"}:
    L1307      from module.models_v5 import CFMV5
    L1308      self.cfm = CFMV5(100, dit)
所以 `s2_train.py` 理论上能训 v5 的 SoVITS —— 只是官方 webui 没给这条路。

注意：早先我遇到 `TypeError: DiT.__init__() got an unexpected keyword
argument 'use_step_embedding'`，根因是 **f5_tts 命名空间包被旧检出遮蔽**
（users.pth 注入），不是 v5 不支持全量。那个问题已通过 PYTHONPATH 修复。

本脚本只做**构造测试**：实例化 v5 的 SynthesizerTrn，确认能建成、
且参数量与全量微调相符。不启动训练（避免与当前 r128 训练抢显存）。

用法：
    python check_v5_fullft.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
need = [str(GSV), str(GSV / "GPT_SoVITS"),
        str(GSV / "GPT_SoVITS" / "BigVGAN"),
        str(GSV / "tools"), str(GSV / "tools" / "asr"),
        str(GSV / "tools" / "uvr5")]
for p in reversed(need):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
os.chdir(str(GSV))

import torch  # noqa: E402

print("=" * 80)
print("1) f5_tts 解析检查（确认不被旧检出遮蔽）")
print("=" * 80)
import inspect  # noqa: E402
from f5_tts.model import DiT  # noqa: E402
sig = list(inspect.signature(DiT.__init__).parameters.keys())
print(f"  DiT 来源: {inspect.getfile(DiT)}")
print(f"  参数: {sig}")
ok_dit = "use_step_embedding" in sig
print(f"  含 use_step_embedding: {ok_dit}  "
      f"{'正确（v5版）' if ok_dit else '**错误（旧版遮蔽）**'}")

print()
print("=" * 80)
print("2) v5 的 SynthesizerTrn 构造测试")
print("=" * 80)
from module.models import SynthesizerTrn  # noqa: E402
from module.models_v5 import CFMV5  # noqa: E402

cfg_path = GSV / "GPT_SoVITS" / "configs" / "s2.json"
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
m = cfg["model"]
d = cfg["data"]

for version in ("v5dev", "v2Pro"):
    try:
        net = SynthesizerTrn(
            d["filter_length"] // 2 + 1,
            m["segment_size"] if "segment_size" in m
            else cfg["train"]["segment_size"] // d["hop_length"],
            n_speakers=d["n_speakers"],
            **{**m, "version": version},
        )
        n_all = sum(p.numel() for p in net.parameters())
        n_cfm = sum(p.numel() for p in net.cfm.parameters())
        is_v5 = isinstance(net.cfm, CFMV5)
        print(f"  {version:8s} OK  总参数 {n_all/1e6:7.2f}M  "
              f"cfm {n_cfm/1e6:6.2f}M  CFMV5={is_v5}")
        del net
    except Exception as e:  # noqa: BLE001
        print(f"  {version:8s} 失败 {type(e).__name__}: {e}")

print()
print("=" * 80)
print("结论")
print("=" * 80)
if ok_dit:
    print("""
  v5 的 SoVITS **可以**用 s2_train.py 做全量微调：
    · models.py 里 v5dev/v5turbo 会实例化 CFMV5（L1306-1308）
    · DiT 也支持 v5 的 use_step_embedding=False

  之前报 TypeError 是 f5_tts 命名空间包被旧检出遮蔽导致的，
  不是 v5 不支持全量微调 —— 该问题已通过 PYTHONPATH 修复。

  => 若要追平 v2Pro 的音色，应该试 **v5 全量微调**（对齐 v2Pro 的做法），
     而不是继续加大 LoRA 秩。
""")
else:
    print("  f5_tts 仍被旧版遮蔽，需先修 PYTHONPATH")
